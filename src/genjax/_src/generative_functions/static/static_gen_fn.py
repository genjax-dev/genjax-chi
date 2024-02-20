# Copyright 2024 MIT Probabilistic Computing Project
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

from deprecated import deprecated
from equinox import module_update_wrapper

from genjax._src.core.datatypes.generative import (
    Choice,
    GenerativeFunction,
    HierarchicalChoiceMap,
    JAXGenerativeFunction,
    Trace,
)
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.interpreters.staging import stage
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    FloatArray,
    PRNGKey,
    Tuple,
    dispatch,
    typecheck,
)
from genjax._src.generative_functions.static.static_datatypes import StaticTrace
from genjax._src.generative_functions.static.static_transforms import (
    assess_transform,
    importance_transform,
    simulate_transform,
    trace,
    update_transform,
)
from genjax._src.generative_functions.supports_callees import (
    SupportsCalleeSugar,
    push_trace_overload_stack,
)

#######################
# Generative function #
#######################


# Callee syntactic sugar handler.
@typecheck
def handler_trace_with_static(
    addr,
    gen_fn: GenerativeFunction,
    args: Tuple,
):
    return trace(addr, gen_fn)(*args)


class StaticGenerativeFunction(
    JAXGenerativeFunction,
    SupportsCalleeSugar,
):
    """A `StaticGenerativeFunction` is a generative function which relies on program
    transformations applied to JAX traceable Python programs to implement the generative
    function interface.

    By virtue of the implementation, any source program which is provided to this generative function *must* be JAX traceable, meaning [all the footguns for programs that JAX exposes](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html) apply to the source program.

    In addition to the normal JAX footguns, there are a few more which are specific to the generative function interface semantics. Here is the full list of language restrictions (and capabilities):

    * One is allowed to use `jax.lax` control flow primitives _so long as the functions provided to the primitives do not contain `trace` invocations_. In other words, utilizing control flow primitives within the source of a `StaticGenerativeFunction`'s source program requires that the control flow primitives get *deterministic* computation.

    * The above restriction also applies to `jax.vmap`.

    !!! tip "Combinators for control flow"

        If you'd like to use control flow _on generative computation_, [the generative function combinators](../generative_functions/combinators) provide a way to do so in a way which is consistent with Gen's semantics and interfaces.

    * Source programs are allowed to utilize untraced randomness, with the usual Gen restrictions. In addition, it is highly recommended (meaning, for correctness, you absolutely should) to use [`jax.random`](https://jax.readthedocs.io/en/latest/jax.random.html) and JAX's PRNG capabilities. To utilize untraced randomness, you'll need to pass in an extra key as an argument to your model.

        ```python
        @static_gen_fn
        def model(key: PRNGKey):
            v = some_untraced_call(key)
            x = trace("x", genjax.normal)(v, 1.0)
            return x
        ```

    !!! warning "(RC later): The debugging UX"

        By virtue of the fact that JAX interpreters will run over arbitrary code used in this language, debugging the source code programs provided to generative functions in this language can be painful.

        *We're aware of it, and we're working on it!*
    """

    source: Callable = Pytree.static()

    # To get the type of return value, just invoke
    # the source (with abstract tracer arguments).
    def __abstract_call__(self, *args) -> Any:
        return self.source(*args)

    def _stage(self, *args):
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_static, self.source
        )
        return stage(syntax_sugar_handled)(*args)

    def _overload(self, *args):
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_static, self.source
        )
        return syntax_sugar_handled(*args)

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
        return StaticTrace(
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
        choice: Choice,
        args: Tuple,
    ) -> Tuple[StaticTrace, FloatArray]:
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_static, self.source
        )
        (
            (
                w,
                (
                    args,
                    retval,
                    address_choices,
                    score,
                ),
            ),
            cache_state,
        ) = importance_transform(syntax_sugar_handled)(key, choice, args)
        return (
            StaticTrace(
                self,
                args,
                retval,
                address_choices,
                cache_state,
                score,
            ),
            w,
        )

    @dispatch
    def update(
        self,
        key: PRNGKey,
        prev: Trace,
        constraints: Choice,
        argdiffs: Tuple,
    ) -> Tuple[Trace, FloatArray, Any, Choice]:
        assert Diff.static_check_tree_diff(argdiffs)
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_static, self.source
        )
        (
            (
                retval_diffs,
                weight,
                (
                    arg_primals,
                    retval_primals,
                    address_choices,
                    score,
                ),
                discard,
            ),
            cache_state,
        ) = update_transform(syntax_sugar_handled)(key, prev, constraints, argdiffs)
        return (
            StaticTrace(
                self,
                arg_primals,
                retval_primals,
                address_choices,
                cache_state,
                score,
            ),
            weight,
            retval_diffs,
            HierarchicalChoiceMap(discard),
        )

    @typecheck
    def assess(
        self,
        choice: Choice,
        args: Tuple,
    ) -> Tuple[FloatArray, Any]:
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_static, self.source
        )
        (retval, score) = assess_transform(syntax_sugar_handled)(choice, args)
        return (score, retval)

    def inline(self, *args):
        return self.source(*args)

    def restore_with_aux(self, interface_data, aux):
        (original_args, retval, score, _) = interface_data
        (
            address_choices,
            cache,
        ) = aux
        return StaticTrace(
            self,
            original_args,
            retval,
            address_choices,
            cache,
            score,
        )

    @property
    def __wrapped__(self):
        return self.source

    ###################
    # Deserialization #
    ###################


#############
# Decorator #
#############


def static_gen_fn(f) -> StaticGenerativeFunction:
    return module_update_wrapper(StaticGenerativeFunction(f))


@deprecated(version="0.0.2", reason="now called @static_gen_fn")
def static(f) -> StaticGenerativeFunction:
    return static_gen_fn(f)
