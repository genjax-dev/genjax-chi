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

from typing import Callable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox import module_update_wrapper

from genjax._src.core.generative import (
    ChoiceMap,
    GenerativeFunction,
    JAXGenerativeFunction,
    Selection,
    Trace,
)
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    ArrayLike,
    FloatArray,
    IntArray,
    PRNGKey,
    Tuple,
    dispatch,
    typecheck,
)
from genjax._src.generative_functions.combinators.drop_arguments import (
    DropArgumentsTrace,
)
from genjax._src.generative_functions.static.static_gen_fn import SupportsCalleeSugar


class VmapTrace(Trace):
    gen_fn: GenerativeFunction
    inner: Trace
    args: Tuple
    retval: Any
    score: FloatArray

    def get_args(self):
        return self.args

    def get_choices(self):
        return jax.vmap(lambda idx, submap: ChoiceMap.a(idx, submap))(
            jnp.arange(len(self.inner.get_score())), self.inner.get_choices()
        )

    def get_gen_fn(self):
        return self.gen_fn

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def try_restore_arguments(self):
        original_arguments = self.get_args()
        return (
            self.inner.restore(original_arguments)
            if isinstance(self.inner, DropArgumentsTrace)
            else self.inner
        )

    def project(
        self,
        key: PRNGKey,
        selection: Selection,
    ) -> FloatArray:
        inner = self.try_restore_arguments()

        def idx_check(idx, inner_slice):
            remaining = selection.step(idx)
            sub_key = jax.random.fold_in(key, idx)
            inner_weight = inner_slice.project(sub_key, remaining)
            return inner_weight

        idxs = jnp.arange(0, len(inner.get_score()))
        ws = jax.vmap(idx_check)(idxs, inner)
        return jnp.sum(ws, axis=0)


class VmapCombinator(JAXGenerativeFunction, SupportsCalleeSugar):
    """> `VmapCombinator` accepts a generative function as input and provides
    `vmap`-based implementations of the generative function interface methods.

    Examples:
        ```python exec="yes" source="tabbed-left"
        import jax
        import jax.numpy as jnp
        import genjax

        console = genjax.console()

        #############################################################
        # One way to create a `VmapCombinator`: using the decorator. #
        #############################################################


        @genjax.vmap_combinator(in_axes=(0,))
        @genjax.static_gen_fn
        def mapped(x):
            noise1 = genjax.normal(0.0, 1.0) @ "noise1"
            noise2 = genjax.normal(0.0, 1.0) @ "noise2"
            return x + noise1 + noise2


        ################################################
        # The other way: use `vmap_combinator` directly #
        ################################################


        @genjax.static_gen_fn
        def add_normal_noise(x):
            noise1 = genjax.normal(0.0, 1.0) @ "noise1"
            noise2 = genjax.normal(0.0, 1.0) @ "noise2"
            return x + noise1 + noise2


        mapped = genjax.vmap_combinator(in_axes=(0,))(add_normal_noise)

        key = jax.random.PRNGKey(314159)
        arr = jnp.ones(100)
        tr = jax.jit(mapped.simulate)(key, (arr,))

        print(console.render(tr))
        ```
    """

    kernel: JAXGenerativeFunction
    in_axes: Tuple = Pytree.static()

    def __abstract_call__(self, *args) -> Any:
        return jax.vmap(self.kernel.__abstract_call__, in_axes=self.in_axes)(*args)

    def _static_check_broadcastable(self, args):
        # Argument broadcast semantics must be fully specified
        # in `in_axes`.
        if not len(args) == len(self.in_axes):
            raise Exception(
                f"VmapCombinator requires that length of the provided in_axes kwarg match the number of arguments provided to the invocation.\nA mismatch occured with len(args) = {len(args)} and len(self.in_axes) = {len(self.in_axes)}"
            )

    def _static_broadcast_dim_length(self, args):
        def find_axis_size(axis, x):
            if axis is not None:
                leaves = jax.tree_util.tree_leaves(x)
                if leaves:
                    return leaves[0].shape[axis]
            return ()

        axis_sizes = jax.tree_util.tree_map(find_axis_size, self.in_axes, args)
        axis_sizes = set(jax.tree_util.tree_leaves(axis_sizes))
        if len(axis_sizes) == 1:
            (d_axis_size,) = axis_sizes
        else:
            raise ValueError(f"Inconsistent batch axis sizes: {axis_sizes}")
        return d_axis_size

    @typecheck
    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> VmapTrace:
        self._static_check_broadcastable(args)
        broadcast_dim_length = self._static_broadcast_dim_length(args)
        sub_keys = jax.random.split(key, broadcast_dim_length)
        tr = jax.vmap(self.kernel.simulate, in_axes=(0, self.in_axes))(sub_keys, args)
        retval = tr.get_retval()
        scores = tr.get_score()
        map_tr = VmapTrace(self, tr, args, retval, jnp.sum(scores))
        return map_tr

    def importance(
        self,
        key: PRNGKey,
        choice: ChoiceMap,
        args: Tuple,
    ) -> Tuple[VmapTrace, FloatArray]:
        self._static_check_broadcastable(args)
        broadcast_dim_length = self._static_broadcast_dim_length(args)
        index_array = jnp.arange(0, broadcast_dim_length)
        sub_keys = jax.random.split(key, broadcast_dim_length)

        def _importance(key, index, choice, args):
            submap = choice.get_submap(index)
            return self.kernel.importance(key, submap, args)

        (tr, w) = jax.vmap(_importance, in_axes=(0, 0, None, self.in_axes))(
            sub_keys, index_array, choice, args
        )
        w = jnp.sum(w)
        retval = tr.get_retval()
        scores = tr.get_score()
        map_tr = VmapTrace(self, tr, args, retval, jnp.sum(scores))
        return (map_tr, w)

    @dispatch
    def maybe_restore_arguments_kernel_update(
        self,
        key: PRNGKey,
        prev: DropArgumentsTrace,
        submap: Any,
        original_arguments: Tuple,
        argdiffs: Tuple,
    ):
        restored = prev.restore(original_arguments)
        return self.kernel.update(key, restored, submap, argdiffs)

    @dispatch
    def maybe_restore_arguments_kernel_update(
        self,
        key: PRNGKey,
        prev: Trace,
        submap: Any,
        original_arguments: Tuple,
        argdiffs: Tuple,
    ):
        return self.kernel.update(key, prev, submap, argdiffs)

    @dispatch
    def update(
        self,
        key: PRNGKey,
        prev: VmapTrace,
        choice: ChoiceMap,
        argdiffs: Tuple,
    ) -> Tuple[VmapTrace, FloatArray, Any, ChoiceMap]:
        args = Diff.tree_primal(argdiffs)
        original_args = prev.get_args()
        self._static_check_broadcastable(args)
        broadcast_dim_length = self._static_broadcast_dim_length(args)
        index_array = jnp.arange(0, broadcast_dim_length)
        sub_keys = jax.random.split(key, broadcast_dim_length)
        inner_trace = prev.inner

        @typecheck
        def _update_inner(
            key: PRNGKey,
            index: IntArray,
            prev: Trace,
            choice: ChoiceMap,
            original_args: Tuple,
            argdiffs: Tuple,
        ):
            submap = choice.get_submap(index)
            return self.maybe_restore_arguments_kernel_update(
                key, prev, submap, original_args, argdiffs
            )

        (tr, w, retval_diff, discard) = jax.vmap(
            _update_inner,
            in_axes=(0, 0, 0, None, self.in_axes, self.in_axes),
        )(sub_keys, index_array, inner_trace, choice, original_args, argdiffs)
        w = jnp.sum(w)
        retval = tr.get_retval()
        scores = tr.get_score()
        map_tr = VmapTrace(self, tr, args, retval, jnp.sum(scores))
        return (map_tr, w, retval_diff, discard)

    @dispatch
    def update(
        self,
        key: PRNGKey,
        prev: VmapTrace,
        choice: ChoiceMap,
        argdiffs: Tuple,
    ) -> Tuple[VmapTrace, FloatArray, Any, ChoiceMap]:
        args = Diff.tree_primal(argdiffs)
        original_args = prev.get_args()
        self._static_check_broadcastable(args)
        broadcast_dim_length = self._static_broadcast_dim_length(args)
        prev_inaxes_tree = jtu.tree_map(
            lambda v: None if v.shape == () else 0, prev.inner
        )
        sub_keys = jax.random.split(key, broadcast_dim_length)

        (tr, w, retval_diff, discard) = jax.vmap(
            self.maybe_restore_arguments_kernel_update,
            in_axes=(0, prev_inaxes_tree, 0, self.in_axes, self.in_axes),
        )(sub_keys, prev.inner, choice.inner, original_args, argdiffs)
        w = jnp.sum(w)
        retval = tr.get_retval()
        scores = tr.get_score()
        map_tr = VmapTrace(self, tr, args, retval, jnp.sum(scores))
        return (map_tr, w, retval_diff, discard)

    @typecheck
    def assess(
        self,
        choice: ChoiceMap,
        args: Tuple,
    ) -> Tuple[ArrayLike, Any]:
        self._static_check_broadcastable(args)
        broadcast_dim_length = self._static_broadcast_dim_length(args)
        choice_dim = Pytree.static_check_tree_leaves_have_matching_leading_dim(choice)

        # The argument leaves and choice map leaves must have matching
        # broadcast dimension.
        #
        # Otherwise, a user may have passed in an invalid (not fully constrained)
        # VectorChoiceMap (or messed up the arguments in some way).
        assert choice_dim == broadcast_dim_length

        inner = choice.inner
        (score, retval) = jax.vmap(self.kernel.assess, in_axes=(0, self.in_axes))(
            inner, args
        )
        return (jnp.sum(score), retval)

    @property
    def __wrapped__(self):
        return self.kernel


#############
# Decorator #
#############


def vmap_combinator(in_axes: Tuple) -> Callable[[Callable], VmapCombinator]:
    def decorator(f) -> VmapCombinator:
        return module_update_wrapper(VmapCombinator(f, in_axes))

    return decorator
