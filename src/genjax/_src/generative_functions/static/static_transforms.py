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

import functools
import itertools
from dataclasses import dataclass

import jax
import jax.tree_util as jtu

from genjax._src.core.datatypes.generative import Choice
from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.interpreters.forward import InitialStylePrimitive
from genjax._src.core.interpreters.forward import StatefulHandler
from genjax._src.core.interpreters.forward import forward
from genjax._src.core.interpreters.forward import initial_style_bind
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.interpreters.incremental import incremental
from genjax._src.core.interpreters.incremental import static_check_no_change
from genjax._src.core.interpreters.incremental import tree_diff_primal
from genjax._src.core.interpreters.incremental import tree_diff_tangent
from genjax._src.core.pytree.const import tree_map_collapse_const
from genjax._src.core.pytree.const import tree_map_static_dynamic
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import Any
from genjax._src.core.typing import Callable
from genjax._src.core.typing import Dict
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import List
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import Union
from genjax._src.core.typing import static_check_is_concrete
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.static.static_datatypes import (
    StaticLanguageChoiceMap,
)


##############
# Primitives #
##############

# Generative function trace intrinsic.
trace_p = InitialStylePrimitive("trace")

# Cache intrinsic.
cache_p = InitialStylePrimitive("cache")


#####
# Saving deterministic results
#####

# This class is used to allow syntactic sugar (e.g. the `@` operator)
# in the static language for functions via the `cache` static_primitives.
@dataclass
class DeferredFunctionCall(Pytree):
    fn: Callable
    kwargs: Dict
    args: Union[None, Tuple]

    def flatten(self):
        return (self.args,), (self.fn, self.kwargs)

    @classmethod
    def new(cls, fn, **kwargs):
        assert not isinstance(fn, GenerativeFunction)
        return DeferredFunctionCall(fn, kwargs, None)

    def __call__(self, *args):
        return DeferredFunctionCall(self.fn, self.kwargs, args)

    def __matmul__(self, addr):
        return cache(addr, self.fn, **self.kwargs)(*self.args)


def save(fn, **kwargs):
    return DeferredFunctionCall.new(fn, **kwargs)


#####
# Address checks
#####


# Usage in intrinsics: ensure that addresses do not contain JAX traced values.
def static_check_address_type(addr):
    check = all(jtu.tree_leaves(jtu.tree_map(static_check_is_concrete, addr)))
    if not check:
        raise Exception("Static addresses must not contained JAX traced values.")


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


def _trace(gen_fn, addr, *args):
    addr = tree_map_static_dynamic(addr)
    return initial_style_bind(trace_p)(_abstract_gen_fn_call)(
        gen_fn,
        addr,
        *args,
    )


@typecheck
def trace(addr: Any, gen_fn: GenerativeFunction) -> Callable:
    """Invoke a generative function, binding its generative semantics with the
    current caller.

    Arguments:
        addr: An address denoting the site of a generative function invocation.
        gen_fn: A generative function invoked as a callee of `StaticLanguageGenerativeFunction`.

    Returns:
        callable: A callable which wraps the `trace_p` primitive, accepting arguments (`args`) and binding the primitive with them. This raises the primitive to be handled by `StaticLanguageGenerativeFunction` transformations.
    """
    assert isinstance(gen_fn, GenerativeFunction)
    return lambda *args: _trace(gen_fn, addr, *args)


##############################################################
# Caching (denotes caching of deterministic subcomputations) #
##############################################################


def _cache(fn, addr, *args):
    return initial_style_bind(cache_p)(fn)(fn, *args, addr)


@typecheck
def cache(addr: Any, fn: Callable, *args: Any) -> Callable:
    """Invoke a generative function, binding its generative semantics with the
    current caller.

    Arguments:
        addr: An address denoting the site of a function invocation.
        fn: A deterministic function whose return value is cached under the arguments (memoization) inside `StaticLanguageGenerativeFunction` traces.

    Returns:
        callable: A callable which wraps the `cache_p` primitive, accepting arguments (`args`) and binding the primitive with them. This raises the primitive to be handled by `StaticLanguageGenerativeFunction` transformations.
    """
    # fn must be deterministic.
    assert not isinstance(fn, GenerativeFunction)
    static_check_address_type(addr)
    return lambda *args: _cache(fn, addr, *args)


######################################
#  Generative function interpreters  #
######################################

#####
# Transform address checks
#####


# Usage in transforms: checks for duplicate addresses.
@dataclass
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


#####
# StaticLanguage handler
#####


# This explicitly makes assumptions about some common fields:
# e.g. it assumes if you are using `StaticLanguageHandler.get_submap`
# in your code, that your derived instance has a `constraints` field.
@dataclass
class StaticLanguageHandler(StatefulHandler):
    # By default, the interpreter handlers for this language
    # handle the two primitives we defined above
    # (`trace_p`, for random choices, and `cache_p`, for deterministic caching)
    def handles(self, prim):
        return prim == trace_p or prim == cache_p

    def visit(self, addr):
        self.address_visitor.visit(addr)

    def get_submap(self, addr):
        addr = tree_map_collapse_const(addr)
        return self.constraints.get_submap(addr)

    def get_subtrace(self, addr):
        addr = tree_map_collapse_const(addr)
        return self.previous_trace.get_subtrace(addr)

    @typecheck
    def set_choice_state(self, addr, tr: Trace):
        self.address_choices[addr] = tr

    @typecheck
    def set_discard_state(self, addr, ch: Choice):
        self.discard_choices[addr] = ch

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


@dataclass
class SimulateHandler(StaticLanguageHandler):
    key: PRNGKey
    score: FloatArray
    address_visitor: AddressVisitor
    address_choices: StaticLanguageChoiceMap
    cache_state: Trie
    cache_visitor: AddressVisitor

    def flatten(self):
        return (
            self.key,
            self.score,
            self.address_visitor,
            self.address_choices,
            self.cache_state,
            self.cache_visitor,
        ), ()

    @classmethod
    def new(cls, key: PRNGKey):
        score = 0.0
        address_visitor = AddressVisitor.new()
        address_choices = StaticLanguageChoiceMap.new()
        cache_state = Trie.new()
        cache_visitor = AddressVisitor.new()
        return SimulateHandler(
            key,
            score,
            address_visitor,
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


@dataclass
class ImportanceHandler(StaticLanguageHandler):
    key: PRNGKey
    score: FloatArray
    weight: FloatArray
    constraints: ChoiceMap
    address_visitor: AddressVisitor
    address_choices: StaticLanguageChoiceMap
    cache_state: Trie
    cache_visitor: AddressVisitor

    def flatten(self):
        return (
            self.key,
            self.score,
            self.weight,
            self.constraints,
            self.address_visitor,
            self.address_choices,
            self.cache_state,
            self.cache_visitor,
        ), ()

    def yield_state(self):
        return (
            self.score,
            self.weight,
            self.address_choices,
            self.cache_state,
        )

    @classmethod
    def new(cls, key, constraints):
        score = 0.0
        weight = 0.0
        address_visitor = AddressVisitor.new()
        address_choices = StaticLanguageChoiceMap.new()
        cache_state = Trie.new()
        cache_visitor = AddressVisitor.new()
        return ImportanceHandler(
            key,
            score,
            weight,
            constraints,
            address_visitor,
            address_choices,
            cache_state,
            cache_visitor,
        )

    def handle_trace(self, *tracers, **params):
        in_tree = params.get("in_tree")
        num_consts = params.get("num_consts")
        passed_in_tracers = tracers[num_consts:]
        gen_fn, addr, *args = jtu.tree_unflatten(in_tree, passed_in_tracers)
        self.visit(addr)
        sub_map = self.get_submap(addr)
        args = tuple(args)
        self.key, sub_key = jax.random.split(self.key)
        (tr, w) = gen_fn.importance(sub_key, sub_map, args)
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
        (
            score,
            weight,
            address_choices,
            cache_state,
        ) = stateful_handler.yield_state()
        return (
            weight,
            (
                args,
                retval,
                address_choices,
                score,
            ),
        ), cache_state

    return wrapper


##########
# Update #
##########


@dataclass
class UpdateHandler(StaticLanguageHandler):
    key: PRNGKey
    score: FloatArray
    weight: FloatArray
    previous_trace: Trace
    constraints: ChoiceMap
    address_visitor: AddressVisitor
    address_choices: StaticLanguageChoiceMap
    discard_choices: StaticLanguageChoiceMap
    cache_state: Trie
    cache_visitor: AddressVisitor

    def flatten(self):
        return (
            self.key,
            self.score,
            self.weight,
            self.previous_trace,
            self.constraints,
            self.address_visitor,
            self.address_choices,
            self.discard_choices,
            self.cache_state,
            self.cache_visitor,
        ), ()

    def yield_state(self):
        return (
            self.score,
            self.weight,
            self.address_choices,
            self.discard_choices,
            self.cache_state,
        )

    @classmethod
    def new(cls, key, previous_trace, constraints):
        score = 0.0
        weight = 0.0
        address_visitor = AddressVisitor.new()
        address_choices = StaticLanguageChoiceMap.new()
        discard_choices = StaticLanguageChoiceMap.new()
        cache_state = Trie.new()
        cache_visitor = AddressVisitor.new()
        return UpdateHandler(
            key,
            score,
            weight,
            previous_trace,
            constraints,
            address_visitor,
            address_choices,
            discard_choices,
            cache_state,
            cache_visitor,
        )

    def handle_trace(self, *tracers, **params):
        in_tree = params.get("in_tree")
        num_consts = params.get("num_consts")
        passed_in_tracers = tracers[num_consts:]
        gen_fn, addr, *argdiffs = jtu.tree_unflatten(in_tree, passed_in_tracers)
        self.visit(addr)

        # Run the update step.
        subtrace = self.get_subtrace(addr)
        subconstraints = self.get_submap(addr)
        argdiffs = tuple(argdiffs)
        self.key, sub_key = jax.random.split(self.key)
        (tr, w, retval_diff, discard) = gen_fn.update(
            sub_key, subtrace, subconstraints, argdiffs
        )
        self.score += tr.get_score()
        self.weight += w
        self.set_choice_state(addr, tr)
        self.set_discard_state(addr, discard)

        # We have to convert the Diff back to tracers to return
        # from the primitive.
        return jtu.tree_leaves(retval_diff, is_leaf=lambda v: isinstance(v, Diff))

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
    @typecheck
    def wrapper(key, previous_trace, constraints, diffs: Tuple):
        stateful_handler = UpdateHandler.new(key, previous_trace, constraints)
        diff_primals = tree_diff_primal(diffs)
        diff_tangents = tree_diff_tangent(diffs)
        retval_diffs = incremental(source_fn)(
            stateful_handler, diff_primals, diff_tangents
        )
        retval_primals = tree_diff_primal(retval_diffs)
        (
            score,
            weight,
            address_choices,
            discard_choices,
            cache_state,
        ) = stateful_handler.yield_state()
        return (
            (
                retval_diffs,
                weight,
                # Trace.
                (
                    diff_primals,
                    retval_primals,
                    address_choices,
                    score,
                ),
                # Discard.
                discard_choices,
            ),
            cache_state,
        )

    return wrapper


##########
# Assess #
##########


@dataclass
class AssessHandler(StaticLanguageHandler):
    score: FloatArray
    constraints: ChoiceMap
    address_visitor: AddressVisitor
    cache_visitor: AddressVisitor

    def flatten(self):
        return (
            self.score,
            self.constraints,
            self.address_visitor,
            self.cache_visitor,
        ), ()

    def yield_state(self):
        return (self.score,)

    @classmethod
    def new(cls, constraints):
        score = 0.0
        address_visitor = AddressVisitor.new()
        cache_visitor = AddressVisitor.new()
        return AssessHandler(
            score,
            constraints,
            address_visitor,
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
        (score, v) = gen_fn.assess(submap, args)
        self.score += score
        return jtu.tree_leaves(v)

    def handle_cache(self, *tracers, **params):
        in_tree = params.get("in_tree")
        fn, *args = jtu.tree_unflatten(in_tree, tracers)
        retval = fn(*args)
        return jtu.tree_leaves(retval)


def assess_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(constraints, args):
        stateful_handler = AssessHandler.new(constraints)
        retval = forward(source_fn)(stateful_handler, *args)
        (score,) = stateful_handler.yield_state()
        return (retval, score)

    return wrapper
