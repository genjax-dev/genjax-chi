# Copyright 2024 The MIT Probabilistic Computing Project & the oryx authors.
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

from genjax._src.core.datatypes.generative import (
    Choice,
    ChoiceMap,
    EmptyChoice,
    GenerativeFunction,
    Trace,
)
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.interpreters.forward import (
    InitialStylePrimitive,
    StatefulHandler,
    forward,
    initial_style_bind,
)
from genjax._src.core.interpreters.incremental import (
    Diff,
    incremental,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    Dict,
    FloatArray,
    List,
    PRNGKey,
    Tuple,
    static_check_is_concrete,
    typecheck,
)
from genjax._src.generative_functions.static.static_datatypes import Trie
from genjax.core.exceptions import AddressReuse, StaticAddressJAX

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
class DeferredFunctionCall(Pytree):
    fn: Callable
    args: Tuple
    kwargs: Dict

    def __call__(self, *args):
        return DeferredFunctionCall(self.fn, self.kwargs, args)

    def __matmul__(self, addr):
        return cache(addr, self.fn, **self.kwargs)(*self.args)


def save(fn, **kwargs):
    return DeferredFunctionCall(fn, **kwargs)


##################
# Address checks #
##################


# Usage in intrinsics: ensure that addresses do not contain JAX traced values.
def static_check_address_type(addr):
    check = all(jtu.tree_leaves(jtu.tree_map(static_check_is_concrete, addr)))
    if not check:
        raise StaticAddressJAX(addr)


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
    static_check_address_type(addr)
    addr = Pytree.tree_const(addr)
    return initial_style_bind(trace_p)(_abstract_gen_fn_call)(
        gen_fn,
        addr,
        *args,
    )


@typecheck
def trace(addr: Any, gen_fn: GenerativeFunction) -> Callable:
    """Invoke a generative function, binding its generative semantics with the current
    caller.

    Arguments:
        addr: An address denoting the site of a generative function invocation.
        gen_fn: A generative function invoked as a callee of `StaticGenerativeFunction`.

    Returns:
        callable: A callable which wraps the `trace_p` primitive, accepting arguments (`args`) and binding the primitive with them. This raises the primitive to be handled by `StaticGenerativeFunction` transformations.
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
    """Invoke a generative function, binding its generative semantics with the current
    caller.

    Arguments:
        addr: An address denoting the site of a function invocation.
        fn: A deterministic function whose return value is cached under the arguments (memoization) inside `StaticGenerativeFunction` traces.

    Returns:
        callable: A callable which wraps the `cache_p` primitive, accepting arguments (`args`) and binding the primitive with them. This raises the primitive to be handled by `StaticGenerativeFunction` transformations.
    """
    # fn must be deterministic.
    assert not isinstance(fn, GenerativeFunction)
    static_check_address_type(addr)
    return lambda *args: _cache(fn, addr, *args)


######################################
#  Generative function interpreters  #
######################################


# Usage in transforms: checks for duplicate addresses.
class AddressVisitor(Pytree):
    visited: List = Pytree.field(default_factory=list)

    def visit(self, addr):
        if addr in self.visited:
            raise AddressReuse(addr)
        else:
            self.visited.append(addr)

    def merge(self, other):
        new = AddressVisitor()
        for addr in itertools.chain(self.visited, other.visited):
            new.visit(addr)


###########################
# Static language handler #
###########################


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
        if isinstance(self.constraints, EmptyChoice):
            return self.constraints
        else:
            addr = Pytree.tree_unwrap_const(addr)
            return self.constraints.get_submap(addr)

    def get_subtrace(self, addr):
        addr = Pytree.tree_unwrap_const(addr)
        return self.previous_trace.get_subtrace(addr)

    @typecheck
    def set_choice_state(self, addr, tr: Trace):
        addr = Pytree.tree_unwrap_const(addr)
        self.address_choices = self.address_choices.trie_insert(addr, tr)

    @typecheck
    def set_discard_state(self, addr, ch: Choice):
        addr = Pytree.tree_unwrap_const(addr)
        self.discard_choices = self.discard_choices.trie_insert(addr, ch)

    def dispatch(self, prim, *tracers, **_params):
        if prim == trace_p:
            return self.handle_trace(*tracers, **_params)
        elif prim == cache_p:
            return self.handle_cache(*tracers, **_params)
        else:
            raise Exception("Illegal primitive: {}".format(prim))


############
# Simulate #
############


@dataclass
class SimulateHandler(StaticLanguageHandler):
    key: PRNGKey
    score: FloatArray = 0.0
    address_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)
    address_choices: Trie = Pytree.field(default_factory=Trie)
    cache_state: Trie = Pytree.field(default_factory=Trie)
    cache_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)

    def yield_state(self):
        return (
            self.address_choices,
            self.cache_state,
            self.score,
        )

    def handle_trace(self, *tracers, **_params):
        in_tree = _params.get("in_tree")
        num_consts = _params.get("num_consts")
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

    def handle_cache(self, *args, **_params):
        raise NotImplementedError


def simulate_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(key, args):
        stateful_handler = SimulateHandler(key)
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
    constraints: ChoiceMap
    score: FloatArray = 0.0
    weight: FloatArray = 0.0
    address_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)
    address_choices: Trie = Pytree.field(default_factory=Trie)
    cache_state: Trie = Pytree.field(default_factory=Trie)
    cache_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)

    def yield_state(self):
        return (
            self.score,
            self.weight,
            self.address_choices,
            self.cache_state,
        )

    def handle_trace(self, *tracers, **_params):
        in_tree = _params.get("in_tree")
        num_consts = _params.get("num_consts")
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

    def handle_cache(self, *tracers, **_params):
        addr = _params.get("addr")
        in_tree = _params.get("in_tree")
        self.cache_visitor.visit(addr)
        fn, args = jtu.tree_unflatten(in_tree, *tracers)
        retval = fn(*args)
        self.cache_state[addr] = retval
        return jtu.tree_leaves(retval)


def importance_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(key, constraints, args):
        stateful_handler = ImportanceHandler(key, constraints)
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
    previous_trace: Trace
    constraints: ChoiceMap
    address_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)
    score: FloatArray = 0.0
    weight: FloatArray = 0.0
    address_choices: Trie = Pytree.field(default_factory=Trie)
    discard_choices: Trie = Pytree.field(default_factory=Trie)
    cache_state: Trie = Pytree.field(default_factory=Trie)
    cache_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)

    def yield_state(self):
        return (
            self.score,
            self.weight,
            self.address_choices,
            self.discard_choices,
            self.cache_state,
        )

    def handle_trace(self, *tracers, **_params):
        in_tree = _params.get("in_tree")
        num_consts = _params.get("num_consts")
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
    def handle_cache(self, *tracers, **_params):
        addr = _params.get("addr")
        in_tree = _params.get("in_tree")
        self.cache_visitor.visit(addr)
        fn, args = jtu.tree_unflatten(in_tree, tracers)
        has_value = self.previous_trace.has_cached_value(addr)

        if (
            static_check_is_concrete(has_value)
            and has_value
            and Diff.static_check_no_change(args)
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
        stateful_handler = UpdateHandler(key, previous_trace, constraints)
        diff_primals = Diff.tree_primal(diffs)
        diff_tangents = Diff.tree_tangent(diffs)
        retval_diffs = incremental(source_fn)(
            stateful_handler, diff_primals, diff_tangents
        )
        retval_primals = Diff.tree_primal(retval_diffs)
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
    constraints: ChoiceMap
    score: FloatArray = 0.0
    address_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)
    cache_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)

    def yield_state(self):
        return (self.score,)

    def handle_trace(self, *tracers, **_params):
        in_tree = _params.get("in_tree")
        num_consts = _params.get("num_consts")
        passed_in_tracers = tracers[num_consts:]
        gen_fn, addr, *args = jtu.tree_unflatten(in_tree, passed_in_tracers)
        self.visit(addr)
        args = tuple(args)
        submap = self.get_submap(addr)
        (score, v) = gen_fn.assess(submap, args)
        self.score += score
        return jtu.tree_leaves(v)

    def handle_cache(self, *tracers, **_params):
        in_tree = _params.get("in_tree")
        fn, *args = jtu.tree_unflatten(in_tree, tracers)
        retval = fn(*args)
        return jtu.tree_leaves(retval)


def assess_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(constraints, args):
        stateful_handler = AssessHandler(constraints)
        retval = forward(source_fn)(stateful_handler, *args)
        (score,) = stateful_handler.yield_state()
        return (retval, score)

    return wrapper
