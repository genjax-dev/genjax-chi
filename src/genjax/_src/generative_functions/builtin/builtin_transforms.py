# Copyright 2022 The MIT Probabilistic Computing Project & the oryx authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import dataclasses
import functools
import itertools
from typing import Any
from typing import List

import jax.core as core
import jax.tree_util as jtu

import genjax._src.core.interpreters.context as context
import genjax._src.core.interpreters.cps as cps
from genjax._src.core.datatypes import ChoiceMap
from genjax._src.core.datatypes import Trace
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.interpreters.context.diff_rules import Diff
from genjax._src.core.interpreters.context.diff_rules import NoChange
from genjax._src.core.interpreters.context.diff_rules import check_is_diff
from genjax._src.core.interpreters.context.diff_rules import check_no_change
from genjax._src.core.interpreters.context.diff_rules import tree_strip_diff
from genjax._src.core.interpreters.staging import is_concrete
from genjax._src.core.interpreters.staging import stage
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import PRNGKey
from genjax._src.generative_functions.builtin.builtin_primitives import cache_p
from genjax._src.generative_functions.builtin.builtin_primitives import gen_fn_p
from genjax._src.generative_functions.builtin.builtin_primitives import inline_p


##################################################
# Bare values (for interfaces sans change types) #
##################################################


@dataclasses.dataclass
class Bare(cps.Cell):
    val: Any
    meta: Any

    def flatten(self):
        return (self.val,), (self.meta,)

    @classmethod
    def new(cls, val):
        return Bare(val, None)

    def get_val(self):
        return self.val


######################################
#  Generative function interpreters  #
######################################

#####
# Transform address checks
#####

# Usage in transforms: checks for duplicate addresses.
@dataclasses.dataclass
class AddressVisitor(Pytree):
    visited: List

    def flatten(self):
        return (), (self.visited,)

    @classmethod
    def new(cls):
        return AddressVisitor([])

    def visit(self, addr):
        if addr in self.visited:
            raise Exception("Already visited this address.")
        else:
            self.visited.append(addr)

    def merge(self, other):
        new = AddressVisitor.new()
        for addr in itertools.chain(self.visited, other.visited):
            new.visit(addr)


#####
# Inlining
#####

# TODO: deprecate.
class DefaultInline:
    def inline(self, cell_type, prim, args, cont, in_tree, **kwargs):
        gen_fn, *args = jtu.tree_unflatten(in_tree, cps.static_map_unwrap(args))
        args = tuple(args)
        transformed = _inline_transform(cell_type, gen_fn.source, self)
        v = transformed(args)
        v = cps.flatmap_outcells(cell_type, v)
        return cont(*v)


def _inline_transform(cell_type, source_fn, handler, **kwargs):
    @functools.wraps(source_fn)
    def _inner(args):
        closed_jaxpr, (flat_args, _, out_tree) = stage(source_fn)(*args, **kwargs)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        with cps.Interpreter.new(cell_type, handler) as interpreter:
            flat_out = interpreter(
                jaxpr,
                [cell_type.new(v) for v in consts],
                list(map(cell_type.new, flat_args)),
            )
        flat_out = map(lambda v: v.get_val(), flat_out)
        retvals = jtu.tree_unflatten(out_tree, flat_out)
        return retvals

    return _inner


# NOTE: base context class for GFI transforms below.
@dataclasses.dataclass
class BuiltinInterfaceContext(context.Context):
    @abc.abstractmethod
    def handle_trace(self, *tracers, **params):
        pass

    @abc.abstractmethod
    def handle_cache(self, *tracers, **params):
        pass

    def can_handle(self, primitive):
        return False

    def process_primitive(self, primitive):
        raise NotImplementedError

    def get_custom_rule(self, primitive):
        if primitive is gen_fn_p:
            return self.handle_trace
        elif primitive is cache_p:
            return self.handle_cache
        else:
            return None


#####
# Inlining
#####


@dataclasses.dataclass
class InlineContext(context.Context):
    def flatten(self):
        return (), ()

    def yield_state(self):
        return ()

    def can_handle(self, primitive):
        return primitive is inline_p

    # Recursively inline - eliminate all `inline_p` primitive
    # bind calls.
    def process_inline(self, *args, **params):
        in_tree = params["in_tree"]
        gen_fn, *call_args = jtu.tree_unflatten(in_tree, args)
        retvals = inline_transform(gen_fn.source)(*call_args)
        return jtu.tree_leaves(retvals)

    def process_primitive(self, primitive, *args, **params):
        if primitive is inline_p:
            return self.process_inline(*args, **params)
        else:
            raise NotImplementedError

    def get_custom_rule(self, primitive):
        return None


def inline_transform(source_fn, **kwargs):
    @functools.wraps(source_fn)
    def wrapper(*args):
        ctx = InlineContext.new()
        retvals, _ = context.transform(source_fn, ctx)(*args, **kwargs)
        return retvals

    return wrapper


#####
# Simulate
#####


@dataclasses.dataclass
class SimulateContext(BuiltinInterfaceContext):
    key: PRNGKey
    score: FloatArray
    choice_state: Trie
    cache_state: Trie
    trace_visitor: AddressVisitor
    cache_visitor: AddressVisitor

    def flatten(self):
        return (
            self.key,
            self.score,
            self.choice_state,
            self.cache_state,
            self.trace_visitor,
            self.cache_visitor,
        ), ()

    @classmethod
    def new(cls, key: PRNGKey):
        score = 0.0
        choice_state = Trie.new()
        cache_state = Trie.new()
        trace_visitor = AddressVisitor.new()
        cache_visitor = AddressVisitor.new()
        return SimulateContext(
            key, score, choice_state, cache_state, trace_visitor, cache_visitor
        )

    def yield_state(self):
        return (self.key, self.choice_state, self.cache_state, self.score)

    def handle_trace(self, _, *args, **params):
        addr = params.get("addr")
        in_tree = params.get("in_tree")
        self.trace_visitor.visit(addr)
        gen_fn, *call_args = jtu.tree_unflatten(in_tree, args)
        call_args = tuple(call_args)
        self.key, tr = gen_fn.simulate(self.key, call_args)
        score = tr.get_score()
        self.choice_state[addr] = tr
        self.score += score
        v = tr.get_retval()
        return jtu.tree_leaves(v)

    def handle_cache(self, _, *args, **params):
        raise NotImplementedError


def simulate_transform(source_fn, **kwargs):
    inlined = inline_transform(source_fn, **kwargs)

    @functools.wraps(source_fn)
    def wrapper(key, args):
        ctx = SimulateContext.new(key)
        retvals, statefuls = context.transform(inlined, ctx)(*args, **kwargs)
        key, constraints, cache, score = statefuls
        return key, (source_fn, args, retvals, constraints, score), cache

    return wrapper


#####
# Importance
#####


@dataclasses.dataclass
class Importance(cps.Handler, DefaultInline):
    handles: List[core.Primitive]
    key: PRNGKey
    score: FloatArray
    weight: FloatArray
    constraints: ChoiceMap
    choice_state: Trie
    cache_state: Trie

    def flatten(self):
        return (
            self.key,
            self.score,
            self.weight,
            self.constraints,
            self.choice_state,
            self.cache_state,
        ), (self.handles,)

    @classmethod
    def new(cls, key, constraints):
        handles = [gen_fn_p, cache_p, inline_p]
        score = 0.0
        weight = 0.0
        choice_state = Trie.new()
        cache_state = Trie.new()
        return Importance(
            handles, key, score, weight, constraints, choice_state, cache_state
        )

    def trace(self, cell_type, prim, args, cont, addr, in_tree, **kwargs):
        gen_fn, *args = jtu.tree_unflatten(in_tree, cps.static_map_unwrap(args))
        sub_map = self.constraints.get_subtree(addr)
        args = tuple(args)
        self.key, (w, tr) = gen_fn.importance(self.key, sub_map, args)
        self.choice_state[addr] = tr
        self.score += tr.get_score()
        self.weight += w
        v = tr.get_retval()
        v = cps.flatmap_outcells(cell_type, v)
        return cont(*v)

    def cache(self, cell_type, prim, args, cont, addr, in_tree, **kwargs):
        fn, args = jtu.tree_unflatten(in_tree, cps.static_map_unwrap(args))
        retval = fn(*args)
        self.cache_state[addr] = retval
        retval = cps.flatmap_outcells(cell_type, retval)
        return cont(*retval)


def importance_transform(source_fn, **kwargs):
    @functools.wraps(source_fn)
    def _inner(key, constraints, args):
        closed_jaxpr, (flat_args, _, out_tree) = stage(source_fn)(*args, **kwargs)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        handler = Importance.new(key, constraints)
        with cps.Interpreter.new(Bare, handler) as interpreter:
            flat_out = interpreter(
                jaxpr,
                [Bare.new(v) for v in consts],
                list(map(Bare.new, flat_args)),
            )
        flat_out = map(lambda v: v.get_val(), flat_out)
        retvals = jtu.tree_unflatten(out_tree, flat_out)
        w = handler.weight
        score = handler.score
        constraints = handler.choice_state
        cache = handler.cache_state
        key = handler.key
        return key, (w, (source_fn, args, retvals, constraints, score)), cache

    return _inner


#####
# Update
#####

# NOTE: to support the `inline_p` primitive.
def _update_inline_transform(cell_type, source_fn, handler, **kwargs):
    @functools.wraps(source_fn)
    def _inner(argdiffs):
        vals = tree_strip_diff(argdiffs)
        closed_jaxpr, (_, _, out_tree) = stage(source_fn)(*vals, **kwargs)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        with cps.Interpreter.new(cell_type, handler) as interpreter:
            flat_argdiffs, _ = jtu.tree_flatten(argdiffs, is_leaf=check_is_diff)
            flat_retval_diffs = interpreter(
                jaxpr,
                [Diff.new(v, change=NoChange) for v in consts],
                flat_argdiffs,
            )
            retvals = tree_strip_diff(flat_retval_diffs)
            retvals = jtu.tree_unflatten(out_tree, retvals)

        return retvals

    return _inner


@dataclasses.dataclass
class Update(cps.Handler):
    handles: List[core.Primitive]
    key: PRNGKey
    score: FloatArray
    weight: FloatArray
    previous_trace: Trace
    constraints: ChoiceMap
    discard: Trie
    choice_state: Trie
    cache_state: Trie

    def flatten(self):
        return (
            self.key,
            self.score,
            self.weight,
            self.previous_trace,
            self.constraints,
            self.discard,
            self.choice_state,
            self.cache_state,
        ), (self.handles,)

    @classmethod
    def new(cls, key, previous_trace, constraints):
        handles = [gen_fn_p, cache_p, inline_p]
        score = 0.0
        weight = 0.0
        choice_state = Trie.new()
        cache_state = Trie.new()
        discard = Trie.new()
        return Update(
            handles,
            key,
            score,
            weight,
            previous_trace,
            constraints,
            discard,
            choice_state,
            cache_state,
        )

    def trace(self, cell_type, prim, args, cont, addr, in_tree, **kwargs):
        gen_fn, *argdiffs = jtu.tree_unflatten(in_tree, args)
        subtrace = self.previous_trace.choices.get_subtree(addr)
        subconstraints = self.constraints.get_subtree(addr)
        argdiffs = tuple(argdiffs)
        self.key, (retval_diff, w, tr, discard) = gen_fn.update(
            self.key, subtrace, subconstraints, argdiffs
        )

        self.weight += w
        self.choice_state[addr] = tr
        self.discard[addr] = discard
        retval_diff = cps.flatmap_outcells(cell_type, retval_diff)
        return cont(*retval_diff)

    def cache(self, cell_type, prim, args, cont, addr, in_tree, **kwargs):
        fn, args = jtu.tree_unflatten(in_tree, args)
        has_value = self.previous_trace.has_cached_value(addr)

        if is_concrete(has_value) and has_value and all(map(check_no_change, args)):
            cached_value = self.previous_trace.get_cached_value(addr)
            self.cache_state[addr] = cached_value
            retval = cps.flatmap_outcells(
                cell_type,
                cached_value,
                change=NoChange,
            )
            return cont(*retval)

        # Otherwise, we codegen by calling into the function.
        retval = fn(*args)
        self.cache_state[addr] = retval

        # Here, we keep the outcells dimension (e.g. the same as
        # the number of outgoing variables in the equation) invariant
        # under our expansion.
        retval = cps.flatmap_outcells(cell_type, retval)
        return cont(*retval)

    def inline(self, cell_type, prim, args, cont, in_tree, **kwargs):
        gen_fn, *argdiffs = jtu.tree_unflatten(in_tree, args)
        transformed = _update_inline_transform(cell_type, gen_fn.source, self)
        argdiffs = tuple(argdiffs)
        v = transformed(argdiffs)
        v = cps.flatmap_outcells(cell_type, v)
        return cont(*v)


def update_transform(source_fn, **kwargs):
    @functools.wraps(source_fn)
    def _inner(key, previous_trace, constraints, argdiffs):
        vals = tree_strip_diff(argdiffs)
        jaxpr, (_, _, out_tree) = stage(source_fn)(*vals, **kwargs)
        jaxpr, consts = jaxpr.jaxpr, jaxpr.literals
        handler = Update.new(key, previous_trace, constraints)
        with cps.Interpreter.new(Diff, handler) as interpreter:
            flat_argdiffs, _ = jtu.tree_flatten(argdiffs, is_leaf=check_is_diff)
            flat_retval_diffs = interpreter(
                jaxpr,
                [Diff.new(v, change=NoChange) for v in consts],
                flat_argdiffs,
            )

        retvals = tree_strip_diff(flat_retval_diffs)
        retvals = jtu.tree_unflatten(out_tree, retvals)
        retval_diffs = jtu.tree_unflatten(out_tree, flat_retval_diffs)
        w = handler.weight
        constraints = handler.choice_state
        cache = handler.cache_state
        discard = handler.discard
        key = handler.key
        return (
            key,
            (
                retval_diffs,
                w,
                (
                    source_fn,
                    vals,
                    retvals,
                    constraints,
                    previous_trace.get_score() + w,
                ),
                discard,
            ),
            cache,
        )

    return _inner


#####
# Assess
#####


@dataclasses.dataclass
class Assess(cps.Handler, DefaultInline):
    handles: List[core.Primitive]
    key: PRNGKey
    score: FloatArray
    constraints: ChoiceMap

    def flatten(self):
        return (
            self.key,
            self.score,
            self.constraints,
        ), (self.handles,)

    @classmethod
    def new(cls, key, constraints):
        handles = [gen_fn_p, cache_p, inline_p]
        score = 0.0
        return Assess(handles, key, score, constraints)

    def trace(self, cell_type, prim, args, cont, addr, in_tree, **kwargs):
        gen_fn, *args = jtu.tree_unflatten(in_tree, cps.static_map_unwrap(args))
        submap = self.constraints.get_subtree(addr)
        args = tuple(args)
        self.key, (v, score) = gen_fn.assess(self.key, submap, args)
        self.score += score
        v = cps.flatmap_outcells(cell_type, v)
        return cont(*v)

    def cache(self, cell_type, prim, args, cont, addr, in_tree, **kwargs):
        fn, args = jtu.tree_unflatten(in_tree, cps.static_map_unwrap(args))
        retval = fn(*args)
        retval = cps.flatmap_outcells(cell_type, retval)
        return cont(*retval)


def assess_transform(source_fn, **kwargs):
    @functools.wraps(source_fn)
    def _inner(key, constraints, args):
        closed_jaxpr, (flat_args, _, out_tree) = stage(source_fn)(*args, **kwargs)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        handler = Assess.new(key, constraints)
        with cps.Interpreter.new(Bare, handler) as interpreter:
            flat_out = interpreter(
                jaxpr,
                [Bare.new(v) for v in consts],
                list(map(Bare.new, flat_args)),
            )
        flat_out = map(lambda v: v.get_val(), flat_out)
        retvals = jtu.tree_unflatten(out_tree, flat_out)
        score = handler.score
        key = handler.key
        return key, (retvals, score)

    return _inner
