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


from genjax._src.core.compiler.interpreters.incremental import Diff, incremental
from genjax._src.core.generative import (
    GFI,
    Argdiffs,
    EditRequest,
    Retdiff,
    Score,
    Trace,
    Weight,
)
from genjax._src.core.generative.choice_map import (
    Address,
    ChoiceMap,
    Selection,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    Generic,
    TypeVar,
)

ArgTuple = TypeVar("ArgTuple", bound=tuple[Any, ...])
R = TypeVar("R")
S = TypeVar("S")


@Pytree.dataclass
class DimapTrace(Generic[R, S], Trace[S]):
    gen_fn: "Dimap[Any, R, S]"
    inner: Trace[R]
    args: tuple[Any, ...]
    retval: S

    def get_args(self) -> tuple[Any, ...]:
        return self.args

    def get_gen_fn(self) -> GFI[S]:
        return self.gen_fn

    def get_choices(self) -> ChoiceMap:
        return self.inner.get_choices()

    def get_retval(self) -> S:
        return self.retval

    def get_score(self) -> Score:
        return self.inner.get_score()

    def get_inner_trace(self, address: Address) -> Trace[R]:
        return self.inner.get_inner_trace(address)


@Pytree.dataclass
class Dimap(Generic[ArgTuple, R, S], GFI[S]):
    """
    A combinator that transforms both the arguments and return values of a [`genjax.GFI`][].

    This combinator allows for the modification of input arguments and return values through specified mapping functions, enabling the adaptation of a generative function to different contexts or requirements.

    Attributes:
        inner: The inner generative function to which the transformations are applied.
        argument_mapping: A function that maps the original arguments to the modified arguments that are passed to the inner generative function.
        retval_mapping: A function that takes a pair of `(args, return_value)` of the inner generative function and returns a mapped return value.
        info: Optional information or description about the specific instance of the combinator.

    Examples:
        Transforming the arguments and return values of a normal distribution draw via the [`genjax.dimap`][] decorator:
        ```python exec="yes" html="true" source="material-block" session="dimap"
        import genjax, jax


        @genjax.dimap(
            # double the mean and halve the std
            pre=lambda mean, std: (mean * 2, std / 2),
            post=lambda _args, retval: retval * 10,
        )
        @genjax.gen
        def transformed_normal_draw(mean, std):
            return genjax.normal(mean, std) @ "x"


        key = jax.random.key(314159)
        tr = jax.jit(transformed_normal_draw.simulate)(
            key,
            (
                0.0,  # Original mean
                1.0,  # Original std
            ),
        )
        print(tr.render_html())
        ```
    """

    inner: GFI[R]
    argument_mapping: Callable[..., ArgTuple] = Pytree.static()
    retval_mapping: Callable[[tuple[Any, ...], ArgTuple, R], S] = Pytree.static()

    def simulate(
        self,
        args: tuple[Any, ...],
    ) -> DimapTrace[R, S]:
        inner_args = self.argument_mapping(*args)
        tr = self.inner.simulate(inner_args)
        inner_retval = tr.get_retval()
        retval = self.retval_mapping(args, inner_args, inner_retval)
        return DimapTrace(self, tr, args, retval)

    def generate(
        self,
        constraint: ChoiceMap,
        args: tuple[Any, ...],
    ) -> tuple[DimapTrace[R, S], Weight]:
        inner_args = self.argument_mapping(*args)
        tr, weight = self.inner.generate(constraint, inner_args)
        inner_retval = tr.get_retval()
        retval = self.retval_mapping(args, inner_args, inner_retval)
        return DimapTrace(self, tr, args, retval), weight

    def project(
        self,
        trace: Trace[S],
        selection: Selection,
    ) -> Weight:
        assert isinstance(trace, DimapTrace)
        return trace.inner.project(selection)

    def edit_change_target(
        self,
        trace: Trace[S],
        request: EditRequest,
        argdiffs: Argdiffs,
    ) -> tuple[DimapTrace[R, S], Weight, Retdiff[S], EditRequest]:
        assert isinstance(trace, DimapTrace)

        primals = Diff.tree_primal(argdiffs)
        tangents = Diff.tree_tangent(argdiffs)

        inner_argdiffs = incremental(self.argument_mapping)(
            None,
            primals,
            tangents,
        )
        inner_trace: Trace[R] = trace.inner

        tr, w, inner_retdiff, bwd_request = self.inner.edit(
            inner_trace,
            request,
            inner_argdiffs,
        )

        inner_retval_primals = Diff.tree_primal(inner_retdiff)
        inner_retval_tangents = Diff.tree_tangent(inner_retdiff)

        def closed_mapping(args: tuple[Any, ...], retval: R) -> S:
            xformed_args = self.argument_mapping(*args)
            return self.retval_mapping(args, xformed_args, retval)

        retval_diff = incremental(closed_mapping)(
            None,
            (primals, inner_retval_primals),
            (tangents, inner_retval_tangents),
        )

        retval_primal: S = Diff.tree_primal(retval_diff)
        return (
            DimapTrace(self, tr, primals, retval_primal),
            w,
            retval_diff,
            bwd_request,
        )

    def edit(
        self,
        trace: Trace[S],
        edit_request: EditRequest,
        argdiffs: Argdiffs,
    ) -> tuple[DimapTrace[R, S], Weight, Retdiff[S], EditRequest]:
        return self.edit_change_target(trace, edit_request, argdiffs)

    def assess(
        self,
        chm: ChoiceMap,
        args: tuple[Any, ...],
    ) -> tuple[Score, S]:
        inner_args = self.argument_mapping(*args)
        w, inner_retval = self.inner.assess(chm, inner_args)
        retval = self.retval_mapping(args, inner_args, inner_retval)
        return w, retval


#############
# Decorator #
#############


def dimap(
    *,
    pre: Callable[..., ArgTuple] = lambda *args: args,
    post: Callable[[tuple[Any, ...], ArgTuple, R], S] = lambda _,
    _xformed,
    retval: retval,
) -> Callable[[GFI[R]], Dimap[ArgTuple, R, S]]:
    """
    Returns a decorator that wraps a [`genjax.GFI`][] and applies pre- and post-processing functions to its arguments and return value.

    !!! info
        Prefer [`genjax.map`][] if you only need to transform the return value, or [`genjax.contramap`][] if you need to transform the arguments.

    Args:
        pre: A callable that preprocesses the arguments before passing them to the wrapped function. Note that `pre` must return a _tuple_ of arguments, not a bare argument. Default is the identity function.
        post: A callable that postprocesses the return value of the wrapped function. Default is the identity function.

    Returns:
        A decorator that takes a [`genjax.GFI`][] and returns a new [`genjax.GFI`][] with the same behavior but with the arguments and return value transformed according to `pre` and `post`.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="dimap"
        import jax, genjax


        # Define pre- and post-processing functions
        def pre_process(x, y):
            return (x + 1, y * 2)


        def post_process(args, xformed, retval):
            return retval**2


        # Apply dimap to a generative function
        @genjax.dimap(pre=pre_process, post=post_process)
        @genjax.gen
        def dimap_model(x, y):
            return genjax.normal(x, y) @ "z"


        # Use the dimap model
        key = jax.random.key(0)
        trace = dimap_model.simulate(key, (2.0, 3.0))

        print(trace.render_html())
        ```
    """

    def decorator(f: GFI[R]) -> Dimap[ArgTuple, R, S]:
        return Dimap(f, pre, post)

    return decorator


def map(
    f: Callable[[R], S],
) -> Callable[[GFI[R]], Dimap[tuple[Any, ...], R, S]]:
    """
    Returns a decorator that wraps a [`genjax.GFI`][] and applies a post-processing function to its return value.

    This is a specialized version of [`genjax.dimap`][] where only the post-processing function is applied.

    Args:
        f: A callable that postprocesses the return value of the wrapped function.

    Returns:
        A decorator that takes a [`genjax.GFI`][] and returns a new [`genjax.GFI`][] with the same behavior but with the return value transformed according to `f`.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="map"
        import jax, genjax


        # Define a post-processing function
        def square(x):
            return x**2


        # Apply map to a generative function
        @genjax.map(square)
        @genjax.gen
        def map_model(x):
            return genjax.normal(x, 1.0) @ "z"


        # Use the map model
        key = jax.random.key(0)
        trace = map_model.simulate(key, (2.0,))

        print(trace.render_html())
        ```
    """

    def post(_args, _xformed, x: R) -> S:
        return f(x)

    return dimap(pre=lambda *args: args, post=post)


def contramap(
    f: Callable[..., ArgTuple],
) -> Callable[[GFI[R]], Dimap[ArgTuple, R, R]]:
    """
    Returns a decorator that wraps a [`genjax.GFI`][] and applies a pre-processing function to its arguments.

    This is a specialized version of [`genjax.dimap`][] where only the pre-processing function is applied.

    Args:
        f: A callable that preprocesses the arguments of the wrapped function. Note that `f` must return a _tuple_ of arguments, not a bare argument.

    Returns:
        A decorator that takes a [`genjax.GFI`][] and returns a new [`genjax.GFI`][] with the same behavior but with the arguments transformed according to `f`.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="contramap"
        import jax, genjax


        # Define a pre-processing function.
        # Note that this function must return a tuple of arguments!
        def add_one(x):
            return (x + 1,)


        # Apply contramap to a generative function
        @genjax.contramap(add_one)
        @genjax.gen
        def contramap_model(x):
            return genjax.normal(x, 1.0) @ "z"


        # Use the contramap model
        key = jax.random.key(0)
        trace = contramap_model.simulate(key, (2.0,))

        print(trace.render_html())
        ```
    """
    return dimap(pre=f, post=lambda _args, _xformed, ret: ret)
