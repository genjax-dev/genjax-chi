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

import jax.core as jc
import jax.tree_util as jtu
from jax import util as jax_util

import genjax._src.core.interpreters.context as context
from genjax._src.core.datatypes import ChoiceMap
from genjax._src.core.datatypes import Trace
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.interpreters.staging import is_concrete
from genjax._src.core.pytree import Pytree
from genjax._src.core.transforms import adev
from genjax._src.core.transforms import incremental
from genjax._src.core.transforms.incremental import Diff
from genjax._src.core.transforms.incremental import DiffTrace
from genjax._src.core.transforms.incremental import static_check_no_change
from genjax._src.core.transforms.incremental import tree_diff_get_tracers
from genjax._src.core.transforms.incremental import tree_diff_primal
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import List
from genjax._src.core.typing import PRNGKey
from genjax._src.generative_functions.builtin.builtin_primitives import cache_p
from genjax._src.generative_functions.builtin.builtin_primitives import inline_p
from genjax._src.generative_functions.builtin.builtin_primitives import trace_p


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
# Builtin interpreter context
#####

# NOTE: base context class for GFI transforms below.
@dataclasses.dataclass
class BuiltinInterfaceContext(context.Context):
    @abc.abstractmethod
    def handle_trace(self, *tracers, **params):
        pass

    @abc.abstractmethod
    def handle_cache(self, *tracers, **params):
        pass

    def can_process(self, primitive):
        return False

    def process_primitive(self, primitive):
        raise NotImplementedError

    def get_custom_rule(self, primitive):
        if primitive is trace_p:
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

    def can_process(self, primitive):
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
class ImportanceContext(BuiltinInterfaceContext):
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
        ), ()

    def yield_state(self):
        return (self.key, self.score, self.weight, self.choice_state, self.cache_state)

    @classmethod
    def new(cls, key, constraints):
        score = 0.0
        weight = 0.0
        choice_state = Trie.new()
        cache_state = Trie.new()
        return ImportanceContext(
            key, score, weight, constraints, choice_state, cache_state
        )

    def handle_trace(self, _, *tracers, **params):
        addr = params["addr"]
        in_tree = params["in_tree"]
        gen_fn, *args = jtu.tree_unflatten(in_tree, tracers)
        sub_map = self.constraints.get_subtree(addr)
        args = tuple(args)
        self.key, (w, tr) = gen_fn.importance(self.key, sub_map, args)
        self.choice_state[addr] = tr
        self.score += tr.get_score()
        self.weight += w
        v = tr.get_retval()
        return jtu.tree_leaves(v)

    def handle_cache(self, _, *tracers, **params):
        addr = params["addr"]
        in_tree = params["in_tree"]
        fn, args = jtu.tree_unflatten(in_tree, *tracers)
        retval = fn(*args)
        self.cache_state[addr] = retval
        return jtu.tree_leaves(retval)


def importance_transform(source_fn, **kwargs):
    inlined = inline_transform(source_fn, **kwargs)

    @functools.wraps(source_fn)
    def wrapper(key, constraints, args):
        ctx = ImportanceContext.new(key, constraints)
        retvals, statefuls = context.transform(inlined, ctx)(*args, **kwargs)
        key, score, weight, choices, cache = statefuls
        return key, (weight, (source_fn, args, retvals, choices, score)), cache

    return wrapper


#####
# Update
#####


@dataclasses.dataclass
class UpdateContext(BuiltinInterfaceContext):
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
        ), ()

    def yield_state(self):
        return (
            self.key,
            self.weight,
            self.choice_state,
            self.cache_state,
            self.discard,
        )

    def get_tracers(self, diff):
        main = self.main_trace
        trace = DiffTrace(main, jc.cur_sublevel())
        out_tracers = tree_diff_get_tracers(diff, trace)
        return out_tracers

    @classmethod
    def new(cls, key, previous_trace, constraints):
        score = 0.0
        weight = 0.0
        choice_state = Trie.new()
        cache_state = Trie.new()
        discard = Trie.new()
        return UpdateContext(
            key,
            score,
            weight,
            previous_trace,
            constraints,
            discard,
            choice_state,
            cache_state,
        )

    def handle_trace(self, _, *tracers, **params):
        addr = params["addr"]
        in_tree = params["in_tree"]
        gen_fn, *tracer_argdiffs = jtu.tree_unflatten(in_tree, tracers)
        argdiffs = tuple(jax_util.safe_map(Diff.from_tracer, tracer_argdiffs))
        subtrace = self.previous_trace.choices.get_subtree(addr)
        subconstraints = self.constraints.get_subtree(addr)
        argdiffs = tuple(argdiffs)
        self.key, (retval_diff, w, tr, discard) = gen_fn.update(
            self.key, subtrace, subconstraints, argdiffs
        )

        self.weight += w
        self.choice_state[addr] = tr
        self.discard[addr] = discard
        # We have to convert the Diff back to tracers to return
        # from the primitive.
        out_tracers = self.get_tracers(retval_diff)
        return jtu.tree_leaves(out_tracers)

    # TODO: fix -- add Diff/tracer return.
    def handle_cache(self, _, *tracers, **params):
        addr = params["addr"]
        in_tree = params["in_tree"]
        fn, args = jtu.tree_unflatten(in_tree, tracers)
        has_value = self.previous_trace.has_cached_value(addr)

        if (
            is_concrete(has_value)
            and has_value
            and all(map(static_check_no_change, args))
        ):
            cached_value = self.previous_trace.get_cached_value(addr)
            self.cache_state[addr] = cached_value
            return jtu.tree_leaves(cached_value)

        retval = fn(*args)
        self.cache_state[addr] = retval
        return jtu.tree_leaves(retval)


def update_transform(source_fn, **kwargs):
    inlined = inline_transform(source_fn, **kwargs)

    @functools.wraps(source_fn)
    def wrapper(key, previous_trace, constraints, diffs):
        ctx = UpdateContext.new(key, previous_trace, constraints)
        retval_diffs, statefuls = incremental.transform(inlined, ctx)(*diffs, **kwargs)
        retval_primals = tree_diff_primal(retval_diffs)
        arg_primals = tree_diff_primal(diffs)
        key, weight, choices, cache, discard = statefuls
        return (
            key,
            (
                retval_diffs,
                weight,
                (
                    source_fn,
                    arg_primals,
                    retval_primals,
                    choices,
                    previous_trace.get_score() + weight,
                ),
                discard,
            ),
            cache,
        )

    return wrapper


#####
# Assess
#####


@dataclasses.dataclass
class AssessContext(BuiltinInterfaceContext):
    key: PRNGKey
    score: FloatArray
    constraints: ChoiceMap

    def flatten(self):
        return (
            self.key,
            self.score,
            self.constraints,
        ), ()

    def yield_state(self):
        return (self.key, self.score)

    @classmethod
    def new(cls, key, constraints):
        score = 0.0
        return AssessContext(key, score, constraints)

    def handle_trace(self, _, *tracers, **params):
        addr = params["addr"]
        in_tree = params["in_tree"]
        gen_fn, *args = jtu.tree_unflatten(in_tree, tracers)
        args = tuple(args)
        submap = self.constraints.get_subtree(addr)
        self.key, (v, score) = gen_fn.assess(self.key, submap, args)
        self.score += score
        return jtu.tree_leaves(v)

    def handle_cache(self, _, *tracers, **params):
        in_tree = params["in_tree"]
        fn, *args = jtu.tree_unflatten(in_tree, tracers)
        retval = fn(*args)
        return jtu.tree_leaves(retval)


def assess_transform(source_fn, **kwargs):
    inlined = inline_transform(source_fn, **kwargs)

    @functools.wraps(source_fn)
    def wrapper(key, constraints, args):
        ctx = AssessContext.new(key, constraints)
        retvals, statefuls = context.transform(inlined, ctx)(*args, **kwargs)
        key, score = statefuls
        return key, (retvals, score)

    return wrapper


###################
# ADEV and fusion #
###################

#####
# Probabilistic computation transform
#####


@dataclasses.dataclass
class ADEVConvertContext(BuiltinInterfaceContext):
    key: PRNGKey

    def flatten(self):
        return (self.key,), ()

    def yield_state(self):
        return (self.key,)

    @classmethod
    def new(cls, key):
        return ADEVConvertContext(
            key,
        )

    def handle_trace(self, _, *tracers, **params):
        in_tree = params["in_tree"]
        gen_fn, *args = jtu.tree_unflatten(in_tree, tracers)
        args = tuple(args)
        adev_term = adev.adev(gen_fn)
        self.key, v = adev.sample(adev_term, self.key, args)
        return jtu.tree_leaves(v)

    def handle_cache(self, _, *tracers, **params):
        in_tree = params["in_tree"]
        fn, args = jtu.tree_unflatten(in_tree, tracers)
        retval = fn(*args)
        return jtu.tree_leaves(retval)


def adev_conversion_transform(source_fn, **kwargs):
    inlined = inline_transform(source_fn, **kwargs)

    @functools.wraps(source_fn)
    def wrapper(key, args):
        ctx = ADEVConvertContext.new(key)
        retvals, statefuls = context.transform(inlined, ctx)(*args, **kwargs)
        (key,) = statefuls
        return key, retvals

    return wrapper


#####
# Fuse canonicalize transform
#####


@dataclasses.dataclass
class FuseContext(BuiltinInterfaceContext):
    key: PRNGKey
    choice_state: Trie

    def flatten(self):
        return (
            self.key,
            self.choice_state,
        ), ()

    def yield_state(self):
        return (self.key, self.choice_state)

    @classmethod
    def new(cls, key):
        choice_state = Trie.new()
        return FuseContext(key, choice_state)

    def handle_trace(self, _, *tracers, **params):
        addr = params["addr"]
        in_tree = params["in_tree"]
        gen_fn, *args = jtu.tree_unflatten(in_tree, tracers)
        args = tuple(args)
        self.key, (v, chm) = gen_fn.prepare_fuse(self.key, *args)
        self.choice_state[addr] = chm
        return jtu.tree_leaves(v)

    def handle_cache(self, _, *tracers, **params):
        in_tree = params["in_tree"]
        fn, args = jtu.tree_unflatten(in_tree, tracers)
        retval = fn(*args)
        return jtu.tree_leaves(retval)


def fuse_transform(source_fn, **kwargs):
    inlined = inline_transform(source_fn, **kwargs)

    @functools.wraps(source_fn)
    def wrapper(key, args):
        ctx = FuseContext.new(key)
        retvals, statefuls = context.transform(inlined, ctx)(*args, **kwargs)
        key, choices = statefuls
        return key, (retvals, choices)

    return wrapper
