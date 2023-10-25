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

import dataclasses
import functools
import itertools
from dataclasses import dataclass

import jax
import jax.core as jc
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.util import safe_map

from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import DynamicHierarchicalChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import HierarchicalTraceType
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.generative import tt_lift
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.interpreters.forward import InitialStylePrimitive
from genjax._src.core.interpreters.forward import StatefulHandler
from genjax._src.core.interpreters.forward import forward
from genjax._src.core.interpreters.forward import initial_style_bind
from genjax._src.core.interpreters.incremental import incremental
from genjax._src.core.interpreters.incremental import static_check_no_change
from genjax._src.core.interpreters.incremental import tree_diff_primals
from genjax._src.core.interpreters.incremental import tree_diff_unpack_leaves
from genjax._src.core.interpreters.staging import stage
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import Any
from genjax._src.core.typing import Callable
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import List
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import String
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import static_check_is_concrete
from genjax._src.core.typing import typecheck


##############
# Primitives #
##############

# Generative function trace intrinsic.
trace_p = InitialStylePrimitive("trace")

# Cache intrinsic.
cache_p = InitialStylePrimitive("cache")


#####
# Static address checks
#####


# Usage in intrinsics: ensure that addresses do not contain JAX traced values.
def static_check_address_type(addr):
    check = all(jtu.tree_leaves(jtu.tree_map(static_check_is_concrete, addr)))
    if not check:
        raise Exception("Addresses must not contained JAX traced values.")


# Usage in intrinsics: ensure that addresses do not contain JAX traced values.
def static_check_concrete_or_dynamic_int_address(addr):
    def _check(v):
        if static_check_is_concrete(v):
            return True
        else:
            # TODO: fix to be more robust to different bit types.
            return v.dtype == jnp.int32

    check = all(jtu.tree_leaves(jtu.tree_map(_check, addr)))
    if not check:
        raise Exception(
            "Addresses must contain concrete (non-traced) values or traced integer values."
        )


#####
# Abstract generative function call
#####


# We defer the abstract call here so that, when we
# stage, any traced values stored in `gen_fn`
# get lifted to by `get_shaped_aval`.
def _abstract_gen_fn_call(gen_fn, _, *args):
    return gen_fn.__abstract_call__(*args)


############################################################
# Trace call (denotes invocation of a generative function) #
############################################################


@dataclass
class PytreeString(Pytree):
    string: String

    def flatten(self):
        return (), (self.string,)


def tree_convert_strings(v):
    def _convert(v):
        if isinstance(v, String):
            return PytreeString(v)
        else:
            return v

    return jtu.tree_map(_convert, v)


def _trace(gen_fn, addr, *args, **kwargs):
    addr = tree_convert_strings(addr)
    return initial_style_bind(trace_p)(_abstract_gen_fn_call)(
        gen_fn,
        addr,
        *args,
        **kwargs,
    )


@typecheck
def trace(addr: Any, gen_fn: GenerativeFunction, **kwargs) -> Callable:
    """Invoke a generative function, binding its generative semantics with the
    current caller.

    Arguments:
        addr: An address denoting the site of a generative function invocation.
        gen_fn: A generative function invoked as a callee of `StaticGenerativeFunction`.

    Returns:
        callable: A callable which wraps the `trace_p` primitive, accepting arguments (`args`) and binding the primitive with them. This raises the primitive to be handled by `StaticGenerativeFunction` transformations.
    """
    assert isinstance(gen_fn, GenerativeFunction)
    return lambda *args: _trace(gen_fn, addr, *args, **kwargs)


##############################################################
# Caching (denotes caching of deterministic subcomputations) #
##############################################################


def _cache(fn, addr, *args, **kwargs):
    return initial_style_bind(cache_p)(fn)(fn, *args, addr, **kwargs)


@typecheck
def cache(addr: Any, fn: Callable, *args: Any, **kwargs) -> Callable:
    """Invoke a generative function, binding its generative semantics with the
    current caller.

    Arguments:
        addr: An address denoting the site of a function invocation.
        fn: A deterministic function whose return value is cached under the arguments (memoization) inside `StaticGenerativeFunction` traces.

    Returns:
        callable: A callable which wraps the `cache_p` primitive, accepting arguments (`args`) and binding the primitive with them. This raises the primitive to be handled by `StaticGenerativeFunction` transformations.
    """
    # fn must be deterministic.
    assert not isinstance(fn, GenerativeFunction)
    static_check_address_type(addr)
    return lambda *args: _cache(fn, addr, *args, **kwargs)


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
# Static handler
#####


@dataclasses.dataclass
class StaticHandler(StatefulHandler):
    @dispatch
    def visit(self, addr: Tuple):
        fst, *rest = addr
        if static_check_is_concrete(fst):
            self.static_address_visitor.visit(addr)
        else:
            self.dynamic_address_visitor.visit(fst, tuple(rest))

    @dispatch
    def visit(self, addr: Any):
        if static_check_is_concrete(addr):
            self.static_address_visitor.visit(addr)
        else:
            self.dynamic_address_visitor.visit(addr, ())

    def set_choice_state(self, addr, tr):
        if isinstance(addr, PytreeString):
            addr = addr.string
        self.address_choices[addr] = tr

    def handles(self, prim):
        return prim == trace_p or prim == cache_p

    def dispatch(self, prim, *tracers, **params):
        if prim == trace_p:
            return self.handle_trace(*tracers, **params)
        elif prim == cache_p:
            return self.handle_cache(*tracers, **params)
        else:
            raise Exception("Illegal primitive: {}".format(prim))


############
# Simulate #
############


@dataclasses.dataclass
class SimulateHandler(StaticHandler):
    key: PRNGKey
    score: FloatArray
    static_address_visitor: AddressVisitor
    dynamic_address_visitor: AddressVisitor
    address_choices: DynamicHierarchicalChoiceMap
    cache_state: Trie
    cache_visitor: AddressVisitor

    def flatten(self):
        return (
            self.key,
            self.score,
            self.static_address_visitor,
            self.dynamic_address_visitor,
            self.address_choices,
            self.cache_state,
            self.cache_visitor,
        ), ()

    @classmethod
    def new(cls, key: PRNGKey):
        score = 0.0
        static_address_visitor = AddressVisitor.new()
        dynamic_address_visitor = DynamicAddressVisitor.new()
        address_choices = DynamicHierarchicalChoiceMap.new()
        cache_state = Trie.new()
        cache_visitor = AddressVisitor.new()
        return SimulateHandler(
            key,
            score,
            static_address_visitor,
            dynamic_address_visitor,
            address_choices,
            cache_state,
            cache_visitor,
        )

    def yield_state(self):
        return (
            self.address_choices,
            self.cache_state,
            self.score,
        )

    def runtime_verify(self):
        self.dynamic_address_visitor.verify()

    def handle_trace(self, *tracers, **params):
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

    def handle_cache(self, *args, **params):
        raise NotImplementedError


def simulate_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(key, args):
        stateful_handler = SimulateHandler.new(key)
        retval = forward(source_fn)(stateful_handler, *args)
        stateful_handler.runtime_verify()  # Produce runtime check for checkify.
        (
            address_choices,
            cache_state,
            score,
        ) = stateful_handler.yield_state()
        return (
            args,
            retval,
            address_choices,
            score,
        ), cache_state

    return wrapper


##############
# Importance #
##############


@dataclasses.dataclass
class ImportanceHandler(StaticHandler):
    key: PRNGKey
    score: FloatArray
    weight: FloatArray
    constraints: ChoiceMap
    # Static addresses
    static_address_visitor: AddressVisitor
    dynamic_address_visitor: AddressVisitor
    # Dynamic addresses
    dynamic_addresses: List[IntArray]
    address_choices: List[ChoiceMap]
    # Caching
    cache_state: Trie
    cache_visitor: AddressVisitor

    def flatten(self):
        return (
            self.key,
            self.score,
            self.weight,
            self.constraints,
            self.static_address_choices,
            self.static_address_visitor,
            self.dynamic_addresses,
            self.dynamic_address_choices,
            self.dynamic_address_visitor,
            self.cache_state,
            self.cache_visitor,
        ), ()

    def yield_state(self):
        return (
            self.score,
            self.weight,
            self.static_address_choices,
            self.dynamic_addresses,
            self.dynamic_address_choices,
            self.cache_state,
        )

    @classmethod
    def new(cls, key, constraints):
        score = 0.0
        weight = 0.0
        static_address_choices = Trie.new()
        static_address_visitor = AddressVisitor.new()
        dynamic_addresses = []
        dynamic_address_choices = []
        dynamic_address_visitor = DynamicAddressVisitor.new()
        cache_state = Trie.new()
        cache_visitor = AddressVisitor.new()
        return ImportanceHandler(
            key,
            score,
            weight,
            constraints,
            static_address_choices,
            static_address_visitor,
            dynamic_addresses,
            dynamic_address_choices,
            dynamic_address_visitor,
            cache_state,
            cache_visitor,
        )

    def runtime_verify(self):
        self.dynamic_address_visitor.verify()

    def get_submap(self, addr: Tuple):
        return self.constraints.get_subtree(addr)

    def handle_trace(self, *tracers, **params):
        in_tree = params.get("in_tree")
        num_consts = params.get("num_consts")
        passed_in_tracers = tracers[num_consts:]
        gen_fn, addr, *args = jtu.tree_unflatten(in_tree, passed_in_tracers)
        self.visit(addr)
        sub_map = self.get_submap(addr)
        args = tuple(args)
        self.key, sub_key = jax.random.split(self.key)
        (w, tr) = gen_fn.importance(sub_key, sub_map, args)
        self.set_choice_state(addr, tr)
        self.score += tr.get_score()
        self.weight += w
        v = tr.get_retval()
        return jtu.tree_leaves(v)

    def handle_cache(self, *tracers, **params):
        addr = params.get("addr")
        in_tree = params.get("in_tree")
        self.cache_visitor.visit(addr)
        fn, args = jtu.tree_unflatten(in_tree, *tracers)
        retval = fn(*args)
        self.cache_state[addr] = retval
        return jtu.tree_leaves(retval)


def importance_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(key, constraints, args):
        stateful_handler = ImportanceHandler.new(key, constraints)
        retval = forward(source_fn)(stateful_handler, *args)
        stateful_handler.runtime_verify()  # Produce runtime check for checkify.
        (
            score,
            weight,
            static_address_choices,
            dynamic_addresses,
            dynamic_address_choices,
            cache_state,
        ) = stateful_handler.yield_state()
        return (
            weight,
            (
                args,
                retval,
                static_address_choices,
                dynamic_addresses,
                dynamic_address_choices,
                score,
            ),
        ), cache_state

    return wrapper


##########
# Update #
##########


@dataclasses.dataclass
class UpdateHandler(StaticHandler):
    key: PRNGKey
    score: FloatArray
    weight: FloatArray
    previous_trace: Trace
    constraints: ChoiceMap
    static_discard: Trie
    dynamic_discard_addresses: List[IntArray]
    dynamic_discard_choices: List[ChoiceMap]
    # Static addresses
    static_address_choices: Trie
    static_address_visitor: AddressVisitor
    # Dynamic addresses
    dynamic_addresses: List[IntArray]
    dynamic_address_choices: List[ChoiceMap]
    dynamic_address_visitor: AddressVisitor
    # Caching
    cache_state: Trie
    cache_visitor: AddressVisitor

    def flatten(self):
        return (
            self.key,
            self.score,
            self.weight,
            self.previous_trace,
            self.constraints,
            self.static_discard,
            self.dynamic_discard_addresses,
            self.dynamic_discard_choices,
            self.static_address_choices,
            self.static_address_visitor,
            self.dynamic_addresses,
            self.dynamic_address_choices,
            self.dynamic_address_visitor,
            self.cache_state,
            self.cache_visitor,
        ), ()

    def yield_state(self):
        return (
            self.score,
            self.weight,
            self.static_discard,
            self.dynamic_discard_addresses,
            self.dynamic_discard_choices,
            self.static_address_choices,
            self.dynamic_addresses,
            self.dynamic_address_choices,
            self.cache_state,
        )

    @classmethod
    def new(cls, key, previous_trace, constraints):
        score = 0.0
        weight = 0.0
        static_discard = Trie.new()
        dynamic_discard_addresses = []
        dynamic_discard_choices = []
        static_address_choices = Trie.new()
        static_address_visitor = AddressVisitor.new()
        dynamic_addresses = []
        dynamic_address_choices = []
        dynamic_address_visitor = DynamicAddressVisitor.new()
        cache_state = Trie.new()
        cache_visitor = AddressVisitor.new()
        return UpdateHandler(
            key,
            score,
            weight,
            previous_trace,
            constraints,
            static_discard,
            dynamic_discard_addresses,
            dynamic_discard_choices,
            static_address_choices,
            static_address_visitor,
            dynamic_addresses,
            dynamic_address_choices,
            dynamic_address_visitor,
            cache_state,
            cache_visitor,
        )

    def runtime_verify(self):
        self.dynamic_address_visitor.verify()

    @dispatch
    def set_discard_state(self, addr: Tuple, chm: ChoiceMap):
        fst, *rest = addr
        if static_check_is_concrete(fst):
            self.static_discard[addr] = chm
        else:
            self.dynamic_discard_addresses.append(fst)
            sub_trie = Trie.new()
            sub_trie[tuple(rest)] = chm
            self.dynamic_discard_choices.append(sub_trie)

    @dispatch
    def set_discard_state(self, addr: Any, chm: ChoiceMap):
        if static_check_is_concrete(addr):
            self.static_discard[addr] = chm
        else:
            self.dynamic_discard_addresses.append(addr)
            self.dynamic_discard_choices.append(chm)

    def handle_trace(self, *tracers, **params):
        in_tree = params.get("in_tree")
        num_consts = params.get("num_consts")
        passed_in_tracers = tracers[num_consts:]
        gen_fn, addr, *argdiffs = jtu.tree_unflatten(in_tree, passed_in_tracers)
        self.visit(addr)

        # Run the update step.
        subtrace = self.get_prev_subtrace(addr)
        subconstraints = self.get_submap(addr)
        argdiffs = tuple(argdiffs)
        self.key, sub_key = jax.random.split(self.key)
        (retval_diff, w, tr, discard) = gen_fn.update(
            sub_key, subtrace, subconstraints, argdiffs
        )
        self.score += tr.get_score()
        self.weight += w
        self.set_choice_state(addr, tr)
        self.set_discard_state(addr, discard)

        # We have to convert the Diff back to tracers to return
        # from the primitive.
        return tree_diff_unpack_leaves(retval_diff)

    # TODO: fix -- add Diff/tracer return.
    def handle_cache(self, *tracers, **params):
        addr = params.get("addr")
        in_tree = params.get("in_tree")
        self.cache_visitor.visit(addr)
        fn, args = jtu.tree_unflatten(in_tree, tracers)
        has_value = self.previous_trace.has_cached_value(addr)

        if (
            static_check_is_concrete(has_value)
            and has_value
            and all(map(static_check_no_change, args))
        ):
            cached_value = self.previous_trace.get_cached_value(addr)
            self.cache_state[addr] = cached_value
            return jtu.tree_leaves(cached_value)

        retval = fn(*args)
        self.cache_state[addr] = retval
        return jtu.tree_leaves(retval)


def update_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(key, previous_trace, constraints, diffs):
        stateful_handler = UpdateHandler.new(key, previous_trace, constraints)
        retval_diffs = incremental(source_fn)(stateful_handler, *diffs)
        stateful_handler.runtime_verify()  # Produce runtime check for checkify.
        retval_primals = tree_diff_primals(retval_diffs)
        arg_primals = tree_diff_primals(diffs)
        (
            score,
            weight,
            static_discard,
            dynamic_discard_addresses,
            dynamic_discard_choices,
            static_address_choices,
            dynamic_addresses,
            dynamic_address_choices,
            cache_state,
        ) = stateful_handler.yield_state()
        return (
            (
                retval_diffs,
                weight,
                # Trace.
                (
                    arg_primals,
                    retval_primals,
                    static_address_choices,
                    dynamic_addresses,
                    dynamic_address_choices,
                    score,
                ),
                # Discard.
                (static_discard, dynamic_discard_addresses, dynamic_discard_choices),
            ),
            cache_state,
        )

    return wrapper


#####
# Assess
#####


@dataclasses.dataclass
class AssessHandler(StaticHandler):
    key: PRNGKey
    score: FloatArray
    constraints: ChoiceMap
    static_address_visitor: AddressVisitor
    dynamic_address_visitor: DynamicAddressVisitor
    cache_visitor: AddressVisitor

    def flatten(self):
        return (
            self.key,
            self.score,
            self.constraints,
            self.static_address_visitor,
            self.dynamic_address_visitor,
            self.cache_visitor,
        ), ()

    def yield_state(self):
        return (self.score,)

    @classmethod
    def new(cls, key, constraints):
        score = 0.0
        static_address_visitor = AddressVisitor.new()
        dynamic_address_visitor = DynamicAddressVisitor.new()
        cache_visitor = AddressVisitor.new()
        return AssessHandler(
            key,
            score,
            constraints,
            static_address_visitor,
            dynamic_address_visitor,
            cache_visitor,
        )

    def handle_trace(self, *tracers, **params):
        in_tree = params.get("in_tree")
        num_consts = params.get("num_consts")
        passed_in_tracers = tracers[num_consts:]
        gen_fn, addr, *args = jtu.tree_unflatten(in_tree, passed_in_tracers)
        self.visit(addr)
        args = tuple(args)
        submap = self.get_submap(addr)
        self.key, sub_key = jax.random.split(self.key)
        (v, score) = gen_fn.assess(sub_key, submap, args)
        self.score += score
        return jtu.tree_leaves(v)

    def handle_cache(self, *tracers, **params):
        in_tree = params.get("in_tree")
        fn, *args = jtu.tree_unflatten(in_tree, tracers)
        retval = fn(*args)
        return jtu.tree_leaves(retval)


def assess_transform(source_fn, **kwargs):
    @functools.wraps(source_fn)
    def wrapper(key, constraints, args):
        stateful_handler = AssessHandler.new(key, constraints)
        retval = forward(source_fn)(stateful_handler, *args, **kwargs)
        stateful_handler.runtime_verify()
        (score,) = stateful_handler.yield_state()
        return (retval, score)

    return wrapper


################
# Trace typing #
################


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
            invals = safe_map(read, eqn.invars)
            gen_fn, addr, *args = jtu.tree_unflatten(in_tree, invals)
            # Addr is `PytreeAddress`.
            tup = addr.to_tuple()
            ty = gen_fn.get_trace_type(*args, **eqn.params)
            inner_trace_type[tup] = ty
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
