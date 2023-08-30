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

import jax
import jax.core as jc
import jax.tree_util as jtu
from jax._src import core as jax_core

import genjax._src.core.interpreters.context as context
from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import HierarchicalTraceType
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.generative import tt_lift
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.interpreters.staging import is_concrete
from genjax._src.core.interpreters.staging import stage
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.transforms import incremental
from genjax._src.core.transforms.incremental import DiffTrace
from genjax._src.core.transforms.incremental import static_check_no_change
from genjax._src.core.transforms.incremental import tree_diff_from_tracer
from genjax._src.core.transforms.incremental import tree_diff_get_tracers
from genjax._src.core.transforms.incremental import tree_diff_primal
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import List
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import Any
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.builtin.builtin_primitives import cache_p
from genjax._src.generative_functions.builtin.builtin_primitives import trace_p
from genjax._src.generative_functions.builtin.builtin_primitives import PytreeAddress


safe_map = jax_core.safe_map
safe_zip = jax_core.safe_zip


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
            raise Exception(
                f"Already visited this address {addr}. Duplicate addresses are not allowed."
            )
        else:
            self.visited.append(addr)

    def merge(self, other):
        new = AddressVisitor.new()
        for addr in itertools.chain(self.visited, other.visited):
            new.visit(addr)


# Usage in transforms: checks for duplicate dynamic addresses.
@dataclasses.dataclass
class DynamicAddressVisitor(Pytree):
    subtree_visited: List
    index_visited: List[IntArray]

    def flatten(self):
        return (self.index_visited,), (self.subtree_visited,)

    @classmethod
    def new(cls):
        return DynamicAddressVisitor([], [])

    @typecheck
    def visit(self, index_addr: IntArray, rst: Tuple):
        self.index_visited.append(index_addr)
        self.subtree_visited.append(rst)

    @typecheck
    def merge(self, other: "DynamicAddressVisitor"):
        new = DynamicAddressVisitor.new()
        for index_addr, subtree_addr in zip(
            itertools.chain(self.index_visited, other.index_visited),
            itertools.chain(self.subtree_visited, other.subtree_visited),
        ):
            new.visit(index_addr, subtree_addr)

    # TODO: checkify.
    def verify(self):
        pass


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

    def yield_state(self):
        raise NotImplementedError

    def runtime_verify(self):
        return None


#####
# Simulate
#####


@dataclasses.dataclass
class SimulateContext(BuiltinInterfaceContext):
    key: PRNGKey
    score: FloatArray
    # Static addresses
    static_choice_state: Trie
    static_address_visitor: AddressVisitor
    # Dynamic addresses
    dynamic_addresses: List[IntArray]
    dynamic_choice_state: List[ChoiceMap]
    dynamic_address_visitor: AddressVisitor
    # Caching
    cache_state: Trie
    cache_visitor: AddressVisitor

    def flatten(self):
        return (
            self.key,
            self.score,
            self.static_choice_state,
            self.static_address_visitor,
            self.dynamic_addresses,
            self.dynamic_choice_state,
            self.dynamic_address_visitor,
            self.cache_state,
            self.cache_visitor,
        ), ()

    @classmethod
    def new(cls, key: PRNGKey):
        score = 0.0
        static_choice_state = Trie.new()
        static_address_visitor = AddressVisitor.new()
        dynamic_addresses = []
        dynamic_choice_state = []
        dynamic_address_visitor = DynamicAddressVisitor.new()
        cache_state = Trie.new()
        cache_visitor = AddressVisitor.new()
        return SimulateContext(
            key,
            score,
            static_choice_state,
            static_address_visitor,
            dynamic_addresses,
            dynamic_choice_state,
            dynamic_address_visitor,
            cache_state,
            cache_visitor,
        )

    def yield_state(self):
        return (
            self.static_choice_state,
            self.dynamic_addresses,
            self.dynamic_choice_state,
            self.cache_state,
            self.score,
        )

    def runtime_verify(self):
        self.dynamic_address_visitor.verify()

    @dispatch
    def visit(self, addr: Tuple):
        fst, *rest = addr
        if is_concrete(fst):
            self.static_address_visitor.visit(addr)
        else:
            self.dynamic_address_visitor.visit(fst, tuple(rest))

    @dispatch
    def visit(self, addr: Any):
        if is_concrete(addr):
            self.static_address_visitor.visit(addr)
        else:
            self.dynamic_address_visitor.visit(fst, ())

    @dispatch
    def visit(self, addr: PytreeAddress):
        tup = addr.to_tuple()
        if len(tup) == 1:
            self.visit(tup[0])
        else:
            self.visit(tup)

    @dispatch
    def set_choice_state(self, addr: Tuple, tr: Trace):
        fst, *rest = addr
        if is_concrete(fst):
            self.static_choice_state[addr] = tr
        else:
            self.dynamic_addresses.append(fst)
            sub_trie = Trie.new()
            sub_trie[tuple(rest)] = tr
            self.dynamic_choice_state.append(sub_trie)

    @dispatch
    def set_choice_state(self, addr: Any, tr: Trace):
        if is_concrete(addr):
            self.static_choice_state[addr] = tr
        else:
            self.dynamic_addresses.append(addr)
            self.dynamic_choice_state.append(tr)

    @dispatch
    def set_choice_state(self, addr: PytreeAddress, tr: Trace):
        tup = addr.to_tuple()
        if len(tup) == 1:
            self.set_choice_state(tup[0], tr)
        else:
            self.set_choice_state(tup, tr)

    def handle_trace(self, _, *tracers, **params):
        in_tree = params.get("in_tree")
        num_consts = params.get("num_consts")
        passed_in_tracers = tracers[num_consts:]
        gen_fn, addr, *call_args = jtu.tree_unflatten(in_tree, passed_in_tracers)
        self.visit(addr)
        call_args = tuple(call_args)
        self.key, sub_key = jax.random.split(self.key)
        tr = gen_fn.simulate(sub_key, call_args)
        score = tr.get_score()
        self.set_choice_state(addr, tr)
        self.score += score
        v = tr.get_retval()
        return jtu.tree_leaves(v)

    def handle_cache(self, _, *args, **params):
        raise NotImplementedError


def simulate_transform(source_fn, **kwargs):
    @functools.wraps(source_fn)
    def wrapper(key, args):
        ctx = SimulateContext.new(key)
        retvals, statefuls = context.transform(source_fn, ctx)(*args, **kwargs)
        ctx.runtime_verify()  # Produce runtime check for checkify.
        static_choices, dynamic_addresses, dynamic_choices, cache, score = statefuls
        return (
            source_fn,
            args,
            retvals,
            static_choices,
            dynamic_addresses,
            dynamic_choices,
            score,
        ), cache

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
    trace_visitor: AddressVisitor
    cache_visitor: AddressVisitor

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
        return (
            self.score,
            self.weight,
            self.choice_state,
            self.cache_state,
        )

    @classmethod
    def new(cls, key, constraints):
        score = 0.0
        weight = 0.0
        choice_state = Trie.new()
        cache_state = Trie.new()
        trace_visitor = AddressVisitor.new()
        cache_visitor = AddressVisitor.new()
        return ImportanceContext(
            key,
            score,
            weight,
            constraints,
            choice_state,
            cache_state,
            trace_visitor,
            cache_visitor,
        )

    def handle_trace(self, _, *tracers, **params):
        addr = params.get("addr")
        in_tree = params.get("in_tree")
        num_consts = params.get("num_consts")
        self.trace_visitor.visit(addr)
        passed_in_tracers = tracers[num_consts:]
        gen_fn, *args = jtu.tree_unflatten(in_tree, passed_in_tracers)
        sub_map = self.constraints.get_subtree(addr)
        args = tuple(args)
        self.key, sub_key = jax.random.split(self.key)
        (w, tr) = gen_fn.importance(sub_key, sub_map, args)
        self.choice_state[addr] = tr
        self.score += tr.get_score()
        self.weight += w
        v = tr.get_retval()
        return jtu.tree_leaves(v)

    def handle_cache(self, _, *tracers, **params):
        addr = params.get("addr")
        in_tree = params.get("in_tree")
        self.cache_visitor.visit(addr)
        fn, args = jtu.tree_unflatten(in_tree, *tracers)
        retval = fn(*args)
        self.cache_state[addr] = retval
        return jtu.tree_leaves(retval)


def importance_transform(source_fn, **kwargs):
    @functools.wraps(source_fn)
    def wrapper(key, constraints, args):
        ctx = ImportanceContext.new(key, constraints)
        retvals, statefuls = context.transform(source_fn, ctx)(*args, **kwargs)
        score, weight, choices, cache = statefuls
        return (weight, (source_fn, args, retvals, choices, score)), cache

    return wrapper


#####
# Update
#####


@dataclasses.dataclass
class UpdateContext(BuiltinInterfaceContext):
    key: PRNGKey
    weight: FloatArray
    previous_trace: Trace
    constraints: ChoiceMap
    discard: Trie
    choice_state: Trie
    cache_state: Trie
    trace_visitor: AddressVisitor
    cache_visitor: AddressVisitor

    def flatten(self):
        return (
            self.key,
            self.weight,
            self.previous_trace,
            self.constraints,
            self.discard,
            self.choice_state,
            self.cache_state,
            self.trace_visitor,
            self.cache_visitor,
        ), ()

    def yield_state(self):
        return (
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
        weight = 0.0
        choice_state = Trie.new()
        cache_state = Trie.new()
        discard = Trie.new()
        trace_visitor = AddressVisitor.new()
        cache_visitor = AddressVisitor.new()
        return UpdateContext(
            key,
            weight,
            previous_trace,
            constraints,
            discard,
            choice_state,
            cache_state,
            trace_visitor,
            cache_visitor,
        )

    def handle_trace(self, _, *tracers, **params):
        addr = params.get("addr")
        in_tree = params.get("in_tree")
        num_consts = params.get("num_consts")
        self.trace_visitor.visit(addr)
        passed_in_tracers = tracers[num_consts:]
        gen_fn, *tracer_argdiffs = jtu.tree_unflatten(in_tree, passed_in_tracers)
        # Convert DiffTracers into Diff values.
        argdiffs = tuple(tree_diff_from_tracer(tracer_argdiffs))

        # Run the update step.
        subtrace = self.previous_trace.choices.get_subtree(addr)
        subconstraints = self.constraints.get_subtree(addr)
        argdiffs = tuple(argdiffs)
        self.key, sub_key = jax.random.split(self.key)
        (retval_diff, w, tr, discard) = gen_fn.update(
            sub_key, subtrace, subconstraints, argdiffs
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
        addr = params.get("addr")
        in_tree = params.get("in_tree")
        self.cache_visitor.visit(addr)
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
    @functools.wraps(source_fn)
    def wrapper(key, previous_trace, constraints, diffs):
        ctx = UpdateContext.new(key, previous_trace, constraints)
        retval_diffs, statefuls = incremental.transform(source_fn, ctx)(
            *diffs, **kwargs
        )
        retval_primals = tree_diff_primal(retval_diffs)
        arg_primals = tree_diff_primal(diffs)
        weight, choices, cache, discard = statefuls
        return (
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
    trace_visitor: AddressVisitor
    cache_visitor: AddressVisitor

    def flatten(self):
        return (
            self.key,
            self.score,
            self.constraints,
            self.trace_visitor,
            self.cache_visitor,
        ), ()

    def yield_state(self):
        return (self.score,)

    @classmethod
    def new(cls, key, constraints):
        score = 0.0
        trace_visitor = AddressVisitor.new()
        cache_visitor = AddressVisitor.new()
        return AssessContext(key, score, constraints, trace_visitor, cache_visitor)

    def handle_trace(self, _, *tracers, **params):
        addr = params.get("addr")
        in_tree = params.get("in_tree")
        num_consts = params.get("num_consts")
        self.trace_visitor.visit(addr)
        passed_in_tracers = tracers[num_consts:]
        gen_fn, *args = jtu.tree_unflatten(in_tree, passed_in_tracers)
        args = tuple(args)
        submap = self.constraints.get_subtree(addr)
        self.key, sub_key = jax.random.split(self.key)
        (v, score) = gen_fn.assess(sub_key, submap, args)
        self.score += score
        return jtu.tree_leaves(v)

    def handle_cache(self, _, *tracers, **params):
        in_tree = params.get("in_tree")
        fn, *args = jtu.tree_unflatten(in_tree, tracers)
        retval = fn(*args)
        return jtu.tree_leaves(retval)


def assess_transform(source_fn, **kwargs):
    @functools.wraps(source_fn)
    def wrapper(key, constraints, args):
        ctx = AssessContext.new(key, constraints)
        retvals, statefuls = context.transform(source_fn, ctx)(*args, **kwargs)
        (score,) = statefuls
        return (retvals, score)

    return wrapper


#####
# Trace typing
#####


def trace_typing(jaxpr: jc.ClosedJaxpr, flat_in, consts):
    # Simple environment, nothing fancy required.
    env = {}
    inner_trace_type = Trie.new()

    def read(var):
        if type(var) is jc.Literal:
            return var.val
        return env[var]

    def write(var, val):
        env[var] = val

    safe_map(write, jaxpr.invars, flat_in)
    safe_map(write, jaxpr.constvars, consts)

    for eqn in jaxpr.eqns:
        if eqn.primitive == trace_p:
            in_tree = eqn.params["in_tree"]
            addr = eqn.params["addr"]
            invals = safe_map(read, eqn.invars)
            gen_fn, *args = jtu.tree_unflatten(in_tree, invals)
            ty = gen_fn.get_trace_type(*args, **eqn.params)
            inner_trace_type[addr] = ty
        outvals = safe_map(lambda v: v.aval, eqn.outvars)
        safe_map(write, eqn.outvars, outvals)

    return safe_map(read, jaxpr.outvars), inner_trace_type


def trace_type_transform(source_fn, **kwargs):
    @functools.wraps(source_fn)
    def _inner(*args):
        closed_jaxpr, (flat_in, _, out_tree) = stage(source_fn)(*args, **kwargs)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        flat_out, inner_tt = trace_typing(jaxpr, flat_in, consts)
        flat_out = list(map(lambda v: tt_lift(v), flat_out))
        if flat_out:
            rettypes = jtu.tree_unflatten(out_tree, flat_out)
        else:
            rettypes = tt_lift(None)
        return HierarchicalTraceType(inner_tt, rettypes)

    return _inner
