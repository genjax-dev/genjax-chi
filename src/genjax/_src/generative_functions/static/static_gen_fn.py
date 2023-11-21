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
from genjax._src.core.datatypes.generative import DynamicHierarchicalChoiceMap
from genjax._src.core.datatypes.generative import EmptyChoice
from genjax._src.core.datatypes.generative import HierarchicalChoiceMap
from genjax._src.core.datatypes.generative import IndexedChoiceMap
from genjax._src.core.datatypes.generative import JAXGenerativeFunction
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.interpreters.incremental import static_check_tree_leaves_diff
from genjax._src.core.pytree.checks import static_check_tree_structure_equivalence
from genjax._src.core.pytree.closure import DynamicClosure
from genjax._src.core.pytree.utilities import tree_stack
from genjax._src.core.typing import Any
from genjax._src.core.typing import Callable
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.static.static_datatypes import StaticTrace
from genjax._src.generative_functions.static.static_transforms import assess_transform
from genjax._src.generative_functions.static.static_transforms import (
    importance_transform,
)
from genjax._src.generative_functions.static.static_transforms import simulate_transform
from genjax._src.generative_functions.static.static_transforms import trace
from genjax._src.generative_functions.static.static_transforms import update_transform
from genjax._src.generative_functions.supports_callees import SupportsCalleeSugar
from genjax._src.generative_functions.supports_callees import push_trace_overload_stack


#######################
# Generative function #
#######################

# Callee syntactic sugar handler.
@typecheck
def handler_trace_with_static(
    addr,
    gen_fn: JAXGenerativeFunction,
    args: Tuple,
):
    return trace(addr, gen_fn)(*args)


@dataclass
class StaticGenerativeFunction(
    JAXGenerativeFunction,
    SupportsCalleeSugar,
):
    source: Callable

    def flatten(self):
        # NOTE: Experimental.
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
    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> StaticTrace:
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_static, self.source
        )
        (args, retval, address_choices, score), cache_state = simulate_transform(
            syntax_sugar_handled
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
            static_chm = EmptyChoice()
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

            if isinstance(static_chm, EmptyChoice):
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
        chm: ChoiceMap,
        args: Tuple,
    ) -> Tuple[Any, FloatArray]:
        (retval, score) = assess_transform(self.source)(chm, args)
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


StaticLanguage = StaticGenerativeFunction.new
