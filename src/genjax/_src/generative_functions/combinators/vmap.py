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
"""The `VmapCombinator` is a generative function combinator which exposes
vectorization on the input arguments of a provided generative function callee.

This vectorization is implemented using `jax.vmap`, and the combinator expects the user to specify `in_axes` as part of the construction of an instance of this combinator.

"""

import jax
import jax.numpy as jnp

from genjax._src.core.generative import (
    Arguments,
    ChoiceMap,
    ChoiceMapConstraint,
    ChoiceMapProjection,
    ChoiceMapSample,
    Constraint,
    CouldPanic,
    EditRequest,
    GenerativeFunction,
    Projection,
    Retdiff,
    Retval,
    Sample,
    Score,
    SelectionProjection,
    Trace,
    Weight,
)
from genjax._src.core.generative.core import SampleCoercableToChoiceMap
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    Generic,
    InAxes,
    PRNGKey,
    TypeVar,
    tuple,
)

G = TypeVar("G", bound=GenerativeFunction)
A = TypeVar("A", bound=Arguments)
S = TypeVar("S", bound=Sample)
R = TypeVar("R", bound=Retval)
G = TypeVar("G", bound=GenerativeFunction)
C = TypeVar("C", bound=Constraint)
S = TypeVar("S", bound=Sample)
P = TypeVar("P", bound=Projection)
Tr = TypeVar("Tr", bound=Trace)
U = TypeVar("U", bound=EditRequest)


@Pytree.dataclass
class VmapTrace(
    Generic[G, A, S, R],
    SampleCoercableToChoiceMap,
    Trace["VmapCombinator", A, ChoiceMapSample, R],
):
    gen_fn: "VmapCombinator"
    inner: Trace[G, A, S, R]
    arguments: A
    retval: R
    score: Score

    def get_args(self) -> A:
        return self.arguments

    def get_retval(self) -> R:
        return self.retval

    def get_gen_fn(self) -> "VmapCombinator":
        return self.gen_fn

    def get_sample(self) -> ChoiceMapSample:
        return ChoiceMapSample(
            jax.vmap(lambda idx, subtrace: ChoiceMap.idx(idx, subtrace.get_sample()))(
                jnp.arange(len(self.inner.get_score())),
                self.inner,
            )
        )

    def get_choices(self) -> ChoiceMap:
        assert isinstance(self.inner, SampleCoercableToChoiceMap), type(self.inner)
        return jax.vmap(
            lambda idx, subtrace: ChoiceMap.idx(idx, subtrace.get_choices())
        )(
            jnp.arange(len(self.inner.get_score())),
            self.inner,
        )

    def get_score(self) -> Score:
        return self.score


@Pytree.dataclass
class VmapCombinator(
    Generic[Tr, A, S, R, C, P, U],
    GenerativeFunction[
        VmapTrace,
        A,
        ChoiceMapSample,
        R,
        ChoiceMapConstraint,
        ChoiceMapProjection | SelectionProjection,
        U,
    ],
):
    """`VmapCombinator` is a generative function which lifts another generative
    function to support `vmap`-based patterns of parallel (and generative)
    computation.

    In contrast to the full set of options which [`jax.vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html), this combinator expects an `in_axes: Tuple` configuration argument, which indicates how the underlying `vmap` patterns should be broadcast across the input arguments to the generative function.

    Attributes:
        gen_fn: A [`genjax.GenerativeFunction`][] to be vectorized.

        in_axes: A tuple specifying which input arguments (or indices into them) should be vectorized. `in_axes` must match (or prefix) the `Pytree` type of the argument tuple for the underlying `gen_fn`. Defaults to 0, i.e., the first argument. See [this link](https://jax/readthedocs.io/en/latest/pytrees.html#applying-optional-parameters-to-pytrees) for more detail.

    Examples:
        Create a `VmapCombinator` using the [`genjax.vmap`][] decorator:
        ```python exec="yes" html="true" source="material-block" session="vmap"
        import jax, genjax
        import jax.numpy as jnp


        @genjax.vmap(in_axes=(0,))
        @genjax.gen
        def mapped(x):
            noise1 = genjax.normal(0.0, 1.0) @ "noise1"
            noise2 = genjax.normal(0.0, 1.0) @ "noise2"
            return x + noise1 + noise2


        key = jax.random.PRNGKey(314159)
        arr = jnp.ones(100)

        tr = jax.jit(mapped.simulate)(key, (arr,))
        print(tr.render_html())
        ```

        Use the [`genjax.GenerativeFunction.vmap`][] method:
        ```python exec="yes" html="true" source="material-block" session="vmap"
        @genjax.gen
        def add_normal_noise(x):
            noise1 = genjax.normal(0.0, 1.0) @ "noise1"
            noise2 = genjax.normal(0.0, 1.0) @ "noise2"
            return x + noise1 + noise2


        mapped = add_normal_noise.vmap(in_axes=(0,))

        tr = jax.jit(mapped.simulate)(key, (arr,))
        print(tr.render_html())
        ```

    """

    gen_fn: GenerativeFunction[Tr, A, S, R, ChoiceMapConstraint, P, U]
    in_axes: InAxes = Pytree.static()

    def __abstract_call__(self, *arguments) -> Any:
        def inner(*arguments):
            return self.gen_fn.__abstract_call__(*arguments)

        return jax.vmap(inner, in_axes=self.in_axes)(*arguments)

    def _static_check_broadcastable(self, arguments: tuple) -> None:
        # Argument broadcast semantics must be fully specified
        # in `in_axes`.
        if self.in_axes is not None:
            axes_length = 1 if isinstance(self.in_axes, int) else len(self.in_axes)
            if not len(arguments) == axes_length:
                raise Exception(
                    f"VmapCombinator requires that length of the provided in_axes kwarg match the number of arguments provided to the invocation.\nA mismatch occured with len(arguments) = {len(arguments)} and len(self.in_axes) = {axes_length}"
                )

    def _static_broadcast_dim_length(self, arguments):
        def find_axis_size(axis, x):
            if axis is not None:
                leaves = jax.tree_util.tree_leaves(x)
                if leaves:
                    return leaves[0].shape[axis]
            return ()

        axis_sizes = jax.tree_util.tree_map(find_axis_size, self.in_axes, arguments)
        axis_sizes = set(jax.tree_util.tree_leaves(axis_sizes))
        if len(axis_sizes) == 1:
            (d_axis_size,) = axis_sizes
        else:
            raise ValueError(f"Inconsistent batch axis sizes: {axis_sizes}")
        return d_axis_size

    def simulate(
        self,
        key: PRNGKey,
        arguments: A,
    ) -> VmapTrace:
        self._static_check_broadcastable(arguments)
        broadcast_dim_length = self._static_broadcast_dim_length(arguments)
        sub_keys = jax.random.split(key, broadcast_dim_length)
        tr = jax.vmap(self.gen_fn.simulate, (0, self.in_axes))(sub_keys, arguments)
        retval = tr.get_retval()
        scores = tr.get_score()
        map_tr = VmapTrace(self, tr, arguments, retval, jnp.sum(scores))
        return map_tr

    def assess(
        self,
        key: PRNGKey,
        sample: ChoiceMapSample,
        arguments: A,
    ) -> CouldPanic[tuple[Score, R]]:
        self._static_check_broadcastable(arguments)
        broadcast_dim_length = self._static_broadcast_dim_length(arguments)
        idx_array = jnp.arange(0, broadcast_dim_length)

        def _assess(key, idx, arguments):
            submap: ChoiceMapSample = sample(idx)  # type: ignore
            return self.gen_fn.assess(key, submap, arguments)

        sub_keys = jax.random.split(key, broadcast_dim_length)
        maybe_err: CouldPanic = jax.vmap(_assess, in_axes=(0, 0, self.in_axes))(
            sub_keys, idx_array, arguments
        )
        return CouldPanic.bind(
            maybe_err.collapse(),
            lambda v: CouldPanic.pure((jnp.sum(v[0]), v[1])),
        )

    def importance_edit(
        self,
        key: PRNGKey,
        constraint: ChoiceMapConstraint,
        arguments: Arguments,
    ) -> tuple[VmapTrace, Weight, ChoiceMapProjection]:
        self._static_check_broadcastable(arguments)
        broadcast_dim_length = self._static_broadcast_dim_length(arguments)
        idx_array = jnp.arange(0, broadcast_dim_length)
        sub_keys = jax.random.split(key, broadcast_dim_length)

        def _importance(key, idx, choice_map, arguments):
            subconstraint: ChoiceMapConstraint = constraint(idx)  # type:ignore
            tr, w, bwd_projection = self.gen_fn.importance_edit(
                key,
                subconstraint,
                arguments,
            )
            return tr, w, ChoiceMapProjection(ChoiceMap.idx(idx, bwd_projection))

        (tr, w, bwd_projection) = jax.vmap(
            _importance, in_axes=(0, 0, None, self.in_axes)
        )(sub_keys, idx_array, constraint, arguments)
        w = jnp.sum(w)
        retval = tr.get_retval()
        scores = tr.get_score()
        map_tr = VmapTrace(self, tr, arguments, retval, jnp.sum(scores))
        return map_tr, w, bwd_projection

    def project_edit(
        self,
        key: PRNGKey,
        trace: VmapTrace,
        projection: SelectionProjection | ChoiceMapProjection,
    ) -> tuple[Weight, ChoiceMapConstraint]:
        raise NotImplementedError

    def choice_map_edit(
        self,
        key: PRNGKey,
        prev: VmapTrace,
        update_request: ChoiceMap,
        arguments: Arguments,
    ) -> tuple[Trace, Weight, Retdiff, ChoiceMap]:
        primals = Diff.tree_primal(arguments)
        self._static_check_broadcastable(primals)
        broadcast_dim_length = self._static_broadcast_dim_length(primals)
        idx_array = jnp.arange(0, broadcast_dim_length)
        sub_keys = jax.random.split(key, broadcast_dim_length)

        def _update(key, idx, subtrace, argdiffs):
            subrequest = update_request(idx)
            new_subtrace, w, retdiff, bwd_problem = self.gen_fn.update(
                key, subtrace, IncrementalEditRequest(argdiffs, subrequest)
            )
            return new_subtrace, w, retdiff, ChoiceMap.idx(idx, bwd_problem)

        new_subtraces, w, retdiff, bwd_problems = jax.vmap(
            _update, in_axes=(0, 0, 0, self.in_axes)
        )(sub_keys, idx_array, prev.inner, argdiffs)
        w = jnp.sum(w)
        retval = new_subtraces.get_retval()
        scores = new_subtraces.get_score()
        map_tr = VmapTrace(self, new_subtraces, primals, retval, jnp.sum(scores))
        return map_tr, w, retdiff, bwd_problems

    def edit(
        self,
        key: PRNGKey,
        trace: VmapTrace,
        update_request: EditRequest,
    ) -> tuple[Trace, Weight, Retdiff, EditRequest]:
        match update_request:
            case IncrementalEditRequest(argdiffs, subrequest):
                return self.update_change_target(key, trace, subrequest, argdiffs)
            case _:
                return self.update_change_target(
                    key, trace, update_request, Diff.no_change(trace.get_args())
                )


#############
# Decorator #
#############


def vmap(*, in_axes: InAxes = 0) -> Callable[[GenerativeFunction], GenerativeFunction]:
    """Returns a decorator that wraps a
    [`GenerativeFunction`][genjax.GenerativeFunction] and returns a new
    `GenerativeFunction` that performs a vectorized map over the argument
    specified by `in_axes`. Traced values are nested under an index, and the
    retval is vectorized.

    Args:
        in_axes: Selector specifying which input arguments (or index into them) should be vectorized. `in_axes` must match (or prefix) the `Pytree` type of the argument tuple for the underlying `gen_fn`. Defaults to 0, i.e., the first argument. See [this link](https://jax.readthedocs.io/en/latest/pytrees.html#applying-optional-parameters-to-pytrees) for more detail.

    Returns:
        A decorator that converts a [`genjax.GenerativeFunction`][] into a new [`genjax.GenerativeFunction`][] that accepts an argument of one-higher dimension at the position specified by `in_axes`.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="vmap"
        import jax, genjax
        import jax.numpy as jnp


        @genjax.vmap(in_axes=0)
        @genjax.gen
        def vmapped_model(x):
            v = genjax.normal(x, 1.0) @ "v"
            return genjax.normal(v, 0.01) @ "q"


        key = jax.random.PRNGKey(314159)
        arr = jnp.ones(100)

        # `vmapped_model` accepts an array of numbers:
        tr = jax.jit(vmapped_model.simulate)(key, (arr,))

        print(tr.render_html())
        ```

    """

    def decorator(gen_fn) -> GenerativeFunction:
        return VmapCombinator(gen_fn, in_axes)

    return decorator
