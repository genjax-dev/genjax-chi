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
    Argdiffs,
    EmptyTrace,
    GenerativeFunction,
    GenericProblem,
    Retdiff,
    Sample,
    Score,
    Trace,
    UpdateProblem,
    Weight,
)
from genjax._src.core.interpreters.incremental import Diff, incremental
from genjax._src.core.pytree import Pytree
from genjax._src.core.traceback_util import register_exclusion
from genjax._src.core.typing import (
    Any,
    Callable,
    Optional,
    PRNGKey,
    String,
    Tuple,
    typecheck,
)

register_exclusion(__file__)


@Pytree.dataclass
class DimapTrace(Trace):
    dimap_combinator: "DimapCombinator"
    inner: Trace
    args: Tuple
    retval: Any

    def get_args(self):
        return self.args

    def get_gen_fn(self):
        return self.dimap_combinator

    def get_sample(self):
        return self.inner.get_sample()

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.inner.get_score()


@Pytree.dataclass
class DimapCombinator(GenerativeFunction):
    """
    A combinator that transforms both the arguments and return values of a [`genjax.GenerativeFunction`][].

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


        key = jax.random.PRNGKey(314159)
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

    inner: GenerativeFunction
    argument_mapping: Callable = Pytree.static()
    retval_mapping: Callable = Pytree.static()
    info: Optional[String] = Pytree.static(default=None)

    @GenerativeFunction.gfi_boundary
    @typecheck
    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> DimapTrace:
        inner_args = self.argument_mapping(*args)
        tr = self.inner.simulate(key, inner_args)
        inner_retval = tr.get_retval()
        retval = self.retval_mapping(inner_args, inner_retval)
        return DimapTrace(self, tr, args, retval)

    @typecheck
    def update_change_target(
        self,
        key: PRNGKey,
        trace: Trace,
        update_problem: UpdateProblem,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        assert isinstance(trace, EmptyTrace | DimapTrace)
        primals = Diff.tree_primal(argdiffs)
        tangents = Diff.tree_tangent(argdiffs)
        inner_argdiffs = incremental(self.argument_mapping)(
            None,
            primals,
            tangents,
        )
        match trace:
            case DimapTrace():
                inner_trace = trace.inner
            case EmptyTrace():
                inner_trace = EmptyTrace(self.inner)
        tr, w, inner_retdiff, bwd_problem = self.inner.update(
            key, inner_trace, GenericProblem(inner_argdiffs, update_problem)
        )
        inner_retval_primals = Diff.tree_primal((inner_retdiff,))
        inner_retval_tangents = Diff.tree_tangent((inner_retdiff,))

        def closed_mapping(args, retval):
            return self.retval_mapping(args, retval)

        retval_diff = incremental(closed_mapping)(
            None,
            (primals, inner_retval_primals),
            (tangents, inner_retval_tangents),
        )
        retval_primal = Diff.tree_primal(retval_diff)
        return (
            DimapTrace(self, tr, primals, retval_primal),
            w,
            retval_diff,
            bwd_problem,
        )

    @typecheck
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        update_problem: UpdateProblem,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        match update_problem:
            case GenericProblem(argdiffs, subproblem):
                return self.update_change_target(key, trace, subproblem, argdiffs)
            case _:
                return self.update_change_target(
                    key, trace, update_problem, Diff.no_change(trace.get_args())
                )

    @typecheck
    def assess(
        self,
        sample: Sample,
        args: Tuple,
    ) -> Tuple[Score, Any]:
        inner_args = self.argument_mapping(*args)
        w, inner_retval = self.inner.assess(sample, inner_args)
        retval = self.retval_mapping(args, inner_retval)
        return w, retval


#############
# Decorator #
#############


def dimap(
    *,
    pre: Callable = lambda *args: args,
    post: Callable = lambda _args, retval: retval,
    info: Optional[String] = None,
) -> Callable[[GenerativeFunction], DimapCombinator]:
    """
    Returns a decorator that wraps a [`genjax.GenerativeFunction`][] and applies pre- and post-processing functions to its arguments and return value.

    !!! info
        Prefer [`genjax.map`][] if you only need to transform the return value, or [`genjax.contramap`][] if you need to transform the arguments.

    Args:
        pre: A callable that preprocesses the arguments before passing them to the wrapped function. Note that `pre` must return a _tuple_ of arguments, not a bare argument. Default is the identity function.
        post: A callable that postprocesses the return value of the wrapped function. Default is the identity function.
        info: An optional string providing additional information about the `dimap` operation.

    Returns:
        A decorator that takes a [`genjax.GenerativeFunction`][] and returns a new [`genjax.GenerativeFunction`][] with the same behavior but with the arguments and return value transformed according to `pre` and `post`.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="dimap"
        import jax, genjax


        # Define pre- and post-processing functions
        def pre_process(x, y):
            return (x + 1, y * 2)


        def post_process(args, retval):
            return retval**2


        # Apply dimap to a generative function
        @genjax.dimap(pre=pre_process, post=post_process, info="Square of normal")
        @genjax.gen
        def dimap_model(x, y):
            return genjax.normal(x, y) @ "z"


        # Use the dimap model
        key = jax.random.PRNGKey(0)
        trace = dimap_model.simulate(key, (2.0, 3.0))

        print(trace.render_html())
        ```
    """

    def decorator(f) -> DimapCombinator:
        return DimapCombinator(f, pre, post, info)

    return decorator


def map(
    f: Callable,
    *,
    info: Optional[String] = None,
) -> Callable[[GenerativeFunction], DimapCombinator]:
    """
    Returns a decorator that wraps a [`genjax.GenerativeFunction`][] and applies a post-processing function to its return value.

    This is a specialized version of [`genjax.dimap`][] where only the post-processing function is applied.

    Args:
        f: A callable that postprocesses the return value of the wrapped function.
        info: An optional string providing additional information about the `map` operation.

    Returns:
        A decorator that takes a [`genjax.GenerativeFunction`][] and returns a new [`genjax.GenerativeFunction`][] with the same behavior but with the return value transformed according to `f`.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="map"
        import jax, genjax


        # Define a post-processing function
        def square(x):
            return x**2


        # Apply map to a generative function
        @genjax.map(square, info="Square of normal")
        @genjax.gen
        def map_model(x):
            return genjax.normal(x, 1.0) @ "z"


        # Use the map model
        key = jax.random.PRNGKey(0)
        trace = map_model.simulate(key, (2.0,))

        print(trace.render_html())
        ```
    """
    return dimap(pre=lambda *args: args, post=lambda _, ret: f(ret), info=info)


def contramap(
    f: Callable,
    *,
    info: Optional[String] = None,
) -> Callable[[GenerativeFunction], DimapCombinator]:
    """
    Returns a decorator that wraps a [`genjax.GenerativeFunction`][] and applies a pre-processing function to its arguments.

    This is a specialized version of [`genjax.dimap`][] where only the pre-processing function is applied.

    Args:
        f: A callable that preprocesses the arguments of the wrapped function. Note that `f` must return a _tuple_ of arguments, not a bare argument.
        info: An optional string providing additional information about the `contramap` operation.

    Returns:
        A decorator that takes a [`genjax.GenerativeFunction`][] and returns a new [`genjax.GenerativeFunction`][] with the same behavior but with the arguments transformed according to `f`.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="contramap"
        import jax, genjax


        # Define a pre-processing function.
        # Note that this function must return a tuple of arguments!
        def add_one(x):
            return (x + 1,)


        # Apply contramap to a generative function
        @genjax.contramap(add_one, info="Add one to input")
        @genjax.gen
        def contramap_model(x):
            return genjax.normal(x, 1.0) @ "z"


        # Use the contramap model
        key = jax.random.PRNGKey(0)
        trace = contramap_model.simulate(key, (2.0,))

        print(trace.render_html())
        ```
    """
    return dimap(pre=f, post=lambda _, ret: ret, info=info)
