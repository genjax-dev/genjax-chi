# Copyright 2022 MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from dataclasses import dataclass

import jax.numpy as jnp

from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import DisjointUnionChoiceMap
from genjax._src.core.datatypes.generative import EmptyChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import HierarchicalChoiceMap
from genjax._src.core.datatypes.generative import IndexedChoiceMap
from genjax._src.core.datatypes.generative import JAXGenerativeFunction
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.generative import TraceType
from genjax._src.core.interpreters.incremental import static_check_tree_leaves_diff
from genjax._src.core.pytree.closure import DynamicClosure
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.pytree.static_checks import (
    static_check_tree_structure_equivalence,
)
from genjax._src.core.pytree.utilities import tree_stack
from genjax._src.core.typing import Any
from genjax._src.core.typing import Callable
from genjax._src.core.typing import Dict
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import Union
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.static.static_datatypes import (
    DynamicHierarchicalChoiceMap,
)
from genjax._src.generative_functions.static.static_datatypes import StaticTrace
from genjax._src.generative_functions.static.static_transforms import assess_transform
from genjax._src.generative_functions.static.static_transforms import cache
from genjax._src.generative_functions.static.static_transforms import (
    importance_transform,
)
from genjax._src.generative_functions.static.static_transforms import simulate_transform
from genjax._src.generative_functions.static.static_transforms import trace
from genjax._src.generative_functions.static.static_transforms import (
    trace_type_transform,
)
from genjax._src.generative_functions.static.static_transforms import update_transform


#####
# Static language syntactic sugar
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


# Denotes that a generative function should be inlined in the
# `@` syntactic sugar for addressing.
class INLINE_FLAG:
    pass


inline = INLINE_FLAG()


# This class is used to allow syntactic sugar (e.g. the `@` operator)
# in the static language for generative functions via the `trace` intrinsic.
@dataclass
class DeferredGenerativeFunctionCall(Pytree):
    gen_fn: GenerativeFunction
    kwargs: Dict
    args: Tuple

    def flatten(self):
        return (self.args,), (self.gen_fn, self.kwargs)

    @classmethod
    def new(cls, gen_fn, args, kwargs):
        return DeferredGenerativeFunctionCall(gen_fn, kwargs, args)

    def __matmul__(self, addr):
        if addr == inline:
            # To use inlining, the generative function must be a
            # `StaticGenerativeFunction`.
            assert isinstance(self.gen_fn, StaticGenerativeFunction)
            return self.gen_fn.inline(*self.args)
        else:
            return trace(addr, self.gen_fn, **self.kwargs)(*self.args)


# This mixin overloads the call functionality for this generative function
# and allows usage of shorthand notation in the static DSL.
class SupportsStaticSugar:
    @dispatch
    def __call__(self, *args: Any, **kwargs) -> DeferredGenerativeFunctionCall:
        return DeferredGenerativeFunctionCall.new(self, args, kwargs)

    @dispatch
    def __call__(self, key: PRNGKey, args: Tuple) -> Any:
        tr = self.simulate(key, args)
        retval = tr.get_retval()
        return retval


#####
# Generative function
#####


@dataclass
class StaticGenerativeFunction(
    JAXGenerativeFunction,
    SupportsStaticSugar,
):
    source: Callable

    def flatten(self):
        if isinstance(self.source, DynamicClosure):
            return (self.source,), ()
        else:
            return (), (self.source,)

    @typecheck
    @classmethod
    def new(cls, source: Callable):
        gen_fn = StaticGenerativeFunction(source)
        functools.update_wrapper(gen_fn, source)
        return gen_fn

    # To get the type of return value, just invoke
    # the source (with abstract tracer arguments).
    def __abstract_call__(self, *args) -> Any:
        return self.source(*args)

    @typecheck
    def get_trace_type(self, *args, **kwargs) -> TraceType:
        return trace_type_transform(self.source, **kwargs)(*args)

    @typecheck
    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> StaticTrace:
        (args, retval, address_choices, score), cache_state = simulate_transform(
            self.source
        )(key, args)
        return StaticTrace.new(
            self,
            args,
            retval,
            address_choices,
            cache_state,
            score,
        )

    @dispatch
    def importance(
        self,
        key: PRNGKey,
        chm: ChoiceMap,
        args: Tuple,
    ) -> Tuple[FloatArray, StaticTrace]:
        (
            w,
            (
                args,
                retval,
                static_address_choices,
                dynamic_addresses,
                dynamic_address_choices,
                score,
            ),
        ), cache_state = importance_transform(self.source)(key, chm, args)
        return (
            w,
            StaticTrace.new(
                self,
                args,
                retval,
                static_address_choices,
                dynamic_addresses,
                dynamic_address_choices,
                cache_state,
                score,
            ),
        )

    def _create_discard(
        self,
        static_address_choices,
        dynamic_addresses,
        dynamic_address_choices,
    ):
        # Handle coercion of the static address choices (a `Trie`)
        # to a choice map.
        if static_address_choices.is_empty():
            static_chm = EmptyChoiceMap()
        else:
            static_chm = HierarchicalChoiceMap.new(static_address_choices)

        # Now deal with the dynamic address choices.
        if not dynamic_addresses and not dynamic_address_choices:
            return static_chm
        else:
            # Specialized path: all structure is the same, we can coerce into
            # an `IndexedChoiceMap`.
            if static_check_tree_structure_equivalence(dynamic_address_choices):
                index_arr = jnp.stack(dynamic_addresses)
                stacked_inner = tree_stack(dynamic_address_choices)
                hierarchical = HierarchicalChoiceMap.new(stacked_inner)
                dynamic = IndexedChoiceMap.new(index_arr, hierarchical)

            # Fallback path: heterogeneous structure, we defer specialization
            # to other methods.
            else:
                dynamic = DynamicHierarchicalChoiceMap.new(
                    dynamic_addresses,
                    dynamic_address_choices,
                )

            if isinstance(static_chm, EmptyChoiceMap):
                return dynamic
            else:
                return DisjointUnionChoiceMap.new([static_chm, dynamic])

    @dispatch
    def update(
        self,
        key: PRNGKey,
        prev: Trace,
        constraints: ChoiceMap,
        argdiffs: Tuple,
    ) -> Tuple[Any, FloatArray, Trace, ChoiceMap]:
        assert static_check_tree_leaves_diff(argdiffs)
        (
            (
                retval_diffs,
                weight,
                (
                    arg_primals,
                    retval_primals,
                    static_address_choices,
                    dynamic_addresses,
                    dynamic_address_choices,
                    score,
                ),
                (static_discard, dynamic_discard_addresses, dynamic_discard_choices),
            ),
            cache_state,
        ) = update_transform(self.source)(key, prev, constraints, argdiffs)
        discard = self._create_discard(
            static_discard,
            dynamic_discard_addresses,
            dynamic_discard_choices,
        )
        return (
            retval_diffs,
            weight,
            StaticTrace.new(
                self,
                arg_primals,
                retval_primals,
                static_address_choices,
                dynamic_addresses,
                dynamic_address_choices,
                cache_state,
                score,
            ),
            discard,
        )

    @typecheck
    def assess(
        self,
        key: PRNGKey,
        chm: ChoiceMap,
        args: Tuple,
    ) -> Tuple[Any, FloatArray]:
        (retval, score) = assess_transform(self.source)(key, chm, args)
        return (retval, score)

    def inline(self, *args):
        return self.source(*args)

    def restore_with_aux(self, interface_data, aux):
        (original_args, retval, score, _) = interface_data
        (
            static_address_choices,
            dynamic_addresses,
            dynamic_address_choices,
            cache,
        ) = aux
        return StaticTrace.new(
            self,
            original_args,
            retval,
            static_address_choices,
            dynamic_addresses,
            dynamic_address_choices,
            cache,
            score,
        )

    ###################
    # Deserialization #
    ###################


#####
# Partial binding / currying
#####


def partial(gen_fn, *static_args):
    return StaticGenerativeFunction.new(
        lambda *args: gen_fn.inline(*args, *static_args),
    )


##############
# Shorthands #
##############

static_generative_function = StaticGenerativeFunction.new


# A decorator to pipe callables into our generative function.
@typecheck
def gen_fn(source: Callable):
    return StaticGenerativeFunction.new(source)


Static = gen_fn
