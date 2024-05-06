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

from genjax._src.core.generative import (
    Address,
    ChoiceMap,
    Constraint,
    GenerativeFunction,
    Retdiff,
    Trace,
    UpdateSpec,
    Weight,
)
from genjax._src.core.generative.core import push_trace_overload_stack
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Closure, Pytree
from genjax._src.core.traceback_util import register_exclusion
from genjax._src.core.typing import (
    Any,
    ArrayLike,
    Callable,
    FloatArray,
    List,
    PRNGKey,
    Tuple,
    typecheck,
)
from genjax._src.generative_functions.static.static_transforms import (
    AddressVisitor,
    assess_transform,
    importance_transform,
    simulate_transform,
    trace,
    update_transform,
)

register_exclusion(__file__)

#########
# Trace #
#########


@Pytree.dataclass
class StaticTrace(Trace):
    gen_fn: GenerativeFunction
    args: Tuple
    retval: Any
    addresses: AddressVisitor
    subtraces: List[Trace]
    score: FloatArray

    def get_args(self) -> Tuple:
        return self.args

    def get_retval(self) -> Any:
        return self.retval

    def get_gen_fn(self) -> GenerativeFunction:
        return self.gen_fn

    def get_sample(self) -> ChoiceMap:
        addresses = self.addresses.get_visited()
        chm = ChoiceMap.n
        for addr, subtrace in zip(addresses, self.subtraces):
            chm = chm ^ ChoiceMap.a(addr, subtrace.get_sample())

        return chm

    def get_score(self) -> FloatArray:
        return self.score

    def get_subtrace(self, addr: Address):
        addresses = self.addresses.get_visited()
        idx = addresses.index(addr)
        return self.subtraces[idx]


#######################
# Generative function #
#######################


# Callee syntactic sugar handler.
@typecheck
def handler_trace_with_static(
    addr: Address,
    gen_fn: GenerativeFunction,
    args: Tuple,
):
    return trace(addr, gen_fn, args)


@Pytree.dataclass
class StaticGenerativeFunction(GenerativeFunction):
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

    source: Closure

    # To get the type of return value, just invoke
    # the source (with abstract tracer arguments).
    def __abstract_call__(self, *args) -> Any:
        return self.source(*args)

    def handle_kwargs(self) -> GenerativeFunction:
        @Pytree.partial()
        def kwarged_source(args, kwargs):
            return self.source(*args, **kwargs)

        return StaticGenerativeFunction(kwarged_source)

    @typecheck
    @GenerativeFunction.gfi_boundary
    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> StaticTrace:
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_static, self.source
        )
        (args, retval, address_visitor, address_traces, score) = simulate_transform(
            syntax_sugar_handled
        )(key, args)
        return StaticTrace(
            self,
            args,
            retval,
            address_visitor,
            address_traces,
            score,
        )

    @typecheck
    def importance_choice_map(
        self,
        key: PRNGKey,
        chm: ChoiceMap,
        args: Tuple,
    ) -> Tuple[StaticTrace, Weight, UpdateSpec]:
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_static, self.source
        )
        (
            w,
            (
                args,
                retval,
                address_visitor,
                address_traces,
                score,
            ),
            bwd_specs,
        ) = importance_transform(syntax_sugar_handled)(key, chm, args)

        def make_bwd_spec(visitor, subspecs):
            addresses = visitor.get_visited()
            addresses = Pytree.tree_unwrap_const(addresses)
            chm = ChoiceMap.n
            for addr, subspec in zip(addresses, subspecs):
                chm = chm ^ ChoiceMap.a(addr, subspec)
            return chm

        bwd_spec = make_bwd_spec(address_visitor, bwd_specs)
        return (
            StaticTrace(
                self,
                args,
                retval,
                address_visitor,
                address_traces,
                score,
            ),
            w,
            bwd_spec,
        )

    @typecheck
    @GenerativeFunction.gfi_boundary
    def importance(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: Tuple,
    ) -> Tuple[StaticTrace, Weight, UpdateSpec]:
        match constraint:
            case ChoiceMap():
                return self.importance_choice_map(key, constraint, args)
            case _:
                raise Exception("Not implemented")

    @typecheck
    @GenerativeFunction.gfi_boundary
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        update_spec: UpdateSpec,
        argdiffs: Tuple,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateSpec]:
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
                    address_visitor,
                    address_traces,
                    score,
                ),
                bwd_specs,
            ),
        ) = update_transform(syntax_sugar_handled)(key, trace, update_spec, argdiffs)

        def make_bwd_spec(visitor, subspecs):
            addresses = visitor.get_visited()
            addresses = Pytree.tree_unwrap_const(addresses)
            chm = ChoiceMap.n
            for addr, subspec in zip(addresses, subspecs):
                chm = chm ^ ChoiceMap.a(addr, subspec)
            return chm

        bwd_spec = make_bwd_spec(address_visitor, bwd_specs)
        return (
            StaticTrace(
                self,
                arg_primals,
                retval_primals,
                address_visitor,
                address_traces,
                score,
            ),
            weight,
            retval_diffs,
            bwd_spec,
        )

    @typecheck
    @GenerativeFunction.gfi_boundary
    def assess(
        self,
        sample: ChoiceMap,
        args: Tuple,
    ) -> Tuple[ArrayLike, Any]:
        syntax_sugar_handled = push_trace_overload_stack(
            handler_trace_with_static, self.source
        )
        (retval, score) = assess_transform(syntax_sugar_handled)(sample, args)
        return (score, retval)

    def inline(self, *args):
        return self.source(*args)


#############
# Decorator #
#############


@typecheck
def static_gen_fn(f: Callable) -> GenerativeFunction:
    if isinstance(f, Closure):
        return StaticGenerativeFunction(f)
    else:
        closure = Pytree.partial()(f)
        return StaticGenerativeFunction(closure)
