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
"""The `VmapCombinator` is a generative function combinator which exposes vectorization
on the input arguments of a provided generative function callee.

This vectorization is implemented using `jax.vmap`, and the combinator expects the user to specify `in_axes` as part of the construction of an instance of this combinator.
"""

import jax
import jax.numpy as jnp

from genjax._src.core.generative import (
    Argdiffs,
    ChoiceMap,
    ChoiceMapConstraint,
    Constraint,
    EditRequest,
    GenerativeFunction,
    IncrementalGenericRequest,
    Projection,
    R,
    Retdiff,
    Score,
    Trace,
    Weight,
)
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    FloatArray,
    Generic,
    InAxes,
    PRNGKey,
)


@Pytree.dataclass
class VmapTrace(Generic[R], Trace[R]):
    gen_fn: GenerativeFunction[R]
    inner: Trace[R]
    args: tuple[Any, ...]
    retval: R
    score: FloatArray

    def get_args(self) -> tuple[Any, ...]:
        return self.args

    def get_retval(self):
        return self.retval

    def get_gen_fn(self):
        return self.gen_fn

    def get_sample(self) -> ChoiceMap:
        return self.get_choices()

    def get_choices(self) -> ChoiceMap:
        return jax.vmap(
            lambda idx, subtrace: ChoiceMap.entry(subtrace.get_choices(), idx)
        )(
            jnp.arange(len(self.inner.get_score())),
            self.inner,
        )

    def get_score(self):
        return self.score


@Pytree.dataclass
class VmapCombinator(Generic[R], GenerativeFunction[R]):
    """`VmapCombinator` is a generative function which lifts another generative function to support `vmap`-based patterns of parallel (and generative) computation.

    In contrast to the full set of options which [`jax.vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html), this combinator expects an `in_axes: tuple` configuration argument, which indicates how the underlying `vmap` patterns should be broadcast across the input arguments to the generative function.

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

    gen_fn: GenerativeFunction[R]
    in_axes: InAxes = Pytree.static()

    def __abstract_call__(self, *args) -> Any:
        def inner(*args):
            return self.gen_fn.__abstract_call__(*args)

        return jax.vmap(inner, in_axes=self.in_axes)(*args)

    def _static_check_broadcastable(self, args: tuple[Any, ...]) -> None:
        # Argument broadcast semantics must be fully specified
        # in `in_axes`.
        if self.in_axes is not None:
            axes_length = 1 if isinstance(self.in_axes, int) else len(self.in_axes)
            if not len(args) == axes_length:
                raise Exception(
                    f"VmapCombinator requires that length of the provided in_axes kwarg match the number of arguments provided to the invocation.\nA mismatch occured with len(args) = {len(args)} and len(self.in_axes) = {axes_length}"
                )

    def _static_broadcast_dim_length(self, args):
        def find_axis_size(axis, x):
            if axis is not None:
                leaves = jax.tree_util.tree_leaves(x)
                if leaves:
                    return leaves[0].shape[axis]
            return ()

        axis_sizes = jax.tree_util.tree_map(
            lambda x, y: None if x is None else find_axis_size(x, y),
            self.in_axes,
            args,
            is_leaf=lambda x: x is None,
        )
        axis_sizes = set(jax.tree_util.tree_leaves(axis_sizes))
        if len(axis_sizes) == 1:
            (d_axis_size,) = axis_sizes
        else:
            raise ValueError(f"Inconsistent batch axis sizes: {axis_sizes}")
        return d_axis_size

    def simulate(
        self,
        key: PRNGKey,
        args: tuple[Any, ...],
    ) -> VmapTrace[R]:
        self._static_check_broadcastable(args)
        broadcast_dim_length = self._static_broadcast_dim_length(args)
        sub_keys = jax.random.split(key, broadcast_dim_length)
        tr = jax.vmap(self.gen_fn.simulate, (0, self.in_axes))(sub_keys, args)
        retval = tr.get_retval()
        scores = tr.get_score()
        map_tr = VmapTrace(self, tr, args, retval, jnp.sum(scores))
        return map_tr

    def generate(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: tuple[Any, ...],
    ) -> tuple[VmapTrace[R], Weight]:
        assert isinstance(constraint, ChoiceMapConstraint)
        self._static_check_broadcastable(args)
        broadcast_dim_length = self._static_broadcast_dim_length(args)
        idx_array = jnp.arange(0, broadcast_dim_length)
        sub_keys = jax.random.split(key, broadcast_dim_length)

        def _importance(key, idx, choice_map, args):
            submap = choice_map(idx)
            tr, w = self.gen_fn.generate(
                key,
                submap,
                args,
            )
            return tr, w

        (tr, w) = jax.vmap(_importance, in_axes=(0, 0, None, self.in_axes))(
            sub_keys, idx_array, constraint, args
        )
        w = jnp.sum(w)
        retval = tr.get_retval()
        scores = tr.get_score()
        map_tr = VmapTrace(self, tr, args, retval, jnp.sum(scores))
        return map_tr, w

    def project(
        self,
        key: PRNGKey,
        trace: Trace[R],
        projection: Projection[Any],
    ) -> Weight:
        raise NotImplementedError

    def edit_choice_map_constraint(
        self,
        key: PRNGKey,
        trace: VmapTrace[R],
        constraint: ChoiceMapConstraint,
        argdiffs: Argdiffs,
    ) -> tuple[VmapTrace[R], Weight, Retdiff[R], EditRequest]:
        primals = Diff.tree_primal(argdiffs)
        self._static_check_broadcastable(primals)
        broadcast_dim_length = self._static_broadcast_dim_length(primals)
        idx_array = jnp.arange(0, broadcast_dim_length)
        sub_keys = jax.random.split(key, broadcast_dim_length)

        def _update(key, idx, subtrace, argdiffs):
            subconstraint = constraint(idx)
            assert isinstance(subconstraint, ChoiceMapConstraint), type(subconstraint)
            new_subtrace, w, retdiff, bwd_request = self.gen_fn.edit(
                key,
                subtrace,
                IncrementalGenericRequest(subconstraint),
                argdiffs,
            )
            assert isinstance(bwd_request, IncrementalGenericRequest)
            inner_chm_constraint = bwd_request.constraint
            return (
                new_subtrace,
                w,
                retdiff,
                ChoiceMapConstraint(ChoiceMap.entry(inner_chm_constraint, idx)),
            )

        new_subtraces, w, retdiff, bwd_constraints = jax.vmap(
            _update, in_axes=(0, 0, 0, self.in_axes)
        )(sub_keys, idx_array, trace.inner, argdiffs)
        w = jnp.sum(w)
        retval = new_subtraces.get_retval()
        scores = new_subtraces.get_score()
        map_tr = VmapTrace(self, new_subtraces, primals, retval, jnp.sum(scores))
        return (
            map_tr,
            w,
            retdiff,
            IncrementalGenericRequest(bwd_constraints),
        )

    def edit(
        self,
        key: PRNGKey,
        trace: Trace[R],
        edit_request: EditRequest,
        argdiffs: Argdiffs,
    ) -> tuple[VmapTrace[R], Weight, Retdiff[R], EditRequest]:
        assert isinstance(trace, VmapTrace)
        assert isinstance(edit_request, IncrementalGenericRequest), type(edit_request)
        constraint = edit_request.constraint
        assert isinstance(constraint, ChoiceMapConstraint)
        return self.edit_choice_map_constraint(
            key,
            trace,
            constraint,
            argdiffs,
        )

    def assess(
        self,
        sample: ChoiceMap,
        args: tuple[Any, ...],
    ) -> tuple[Score, R]:
        self._static_check_broadcastable(args)
        broadcast_dim_length = self._static_broadcast_dim_length(args)
        idx_array = jnp.arange(0, broadcast_dim_length)

        def _assess(idx, args):
            submap = sample(idx)
            return self.gen_fn.assess(submap, args)

        scores, retvals = jax.vmap(_assess, in_axes=(0, self.in_axes))(idx_array, args)
        return jnp.sum(scores), retvals


#############
# Decorator #
#############


def vmap(
    *, in_axes: InAxes = 0
) -> Callable[[GenerativeFunction[R]], VmapCombinator[R]]:
    """
    Returns a decorator that wraps a [`GenerativeFunction`][genjax.GenerativeFunction] and returns a new `GenerativeFunction` that performs a vectorized map over the argument specified by `in_axes`. Traced values are nested under an index, and the retval is vectorized.

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

    def decorator(gen_fn: GenerativeFunction[R]) -> VmapCombinator[R]:
        return VmapCombinator(gen_fn, in_axes)

    return decorator
