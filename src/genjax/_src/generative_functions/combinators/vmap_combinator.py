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
    ChoiceMap,
    Constraint,
    GenerativeFunction,
    GenerativeFunctionClosure,
    Retdiff,
    Trace,
    UpdateSpec,
    Weight,
)
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    ArrayLike,
    Callable,
    FloatArray,
    IntArray,
    PRNGKey,
    Tuple,
    typecheck,
)


@Pytree.dataclass
class VmapTrace(Trace):
    gen_fn: GenerativeFunction
    inner: Trace
    retval: Any
    score: FloatArray

    def get_sample(self):
        return jax.vmap(lambda idx, submap: ChoiceMap.a(idx, submap))(
            jnp.arange(len(self.inner.get_score())), self.inner.get_sample()
        )

    def get_gen_fn(self):
        return self.gen_fn

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score


@Pytree.dataclass
class VmapCombinator(GenerativeFunction):
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

    args: Tuple
    kernel: GenerativeFunctionClosure
    in_axes: Tuple = Pytree.static()

    def __abstract_call__(self) -> Any:
        def inner(*args):
            kernel = self.kernel(*args)
            return kernel.__abstract_call__()

        return jax.vmap(inner, in_axes=self.in_axes)(*self.args)

    def _static_check_broadcastable(self):
        # Argument broadcast semantics must be fully specified
        # in `in_axes`.
        if not len(self.args) == len(self.in_axes):
            raise Exception(
                f"VmapCombinator requires that length of the provided in_axes kwarg match the number of arguments provided to the invocation.\nA mismatch occured with len(args) = {len(self.args)} and len(self.in_axes) = {len(self.in_axes)}"
            )

    def _static_broadcast_dim_length(self):
        def find_axis_size(axis, x):
            if axis is not None:
                leaves = jax.tree_util.tree_leaves(x)
                if leaves:
                    return leaves[0].shape[axis]
            return ()

        axis_sizes = jax.tree_util.tree_map(find_axis_size, self.in_axes, self.args)
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
    ) -> VmapTrace:
        self._static_check_broadcastable()
        broadcast_dim_length = self._static_broadcast_dim_length()
        sub_keys = jax.random.split(key, broadcast_dim_length)

        def inner(key, args):
            kernel_gen_fn = self.kernel(*args)
            tr = kernel_gen_fn.simulate(key)
            return tr

        tr = jax.vmap(inner, (0, self.in_axes))(sub_keys, self.args)

        retval = tr.get_retval()
        scores = tr.get_score()
        map_tr = VmapTrace(self, tr, retval, jnp.sum(scores))
        return map_tr

    def importance_choice_map(
        self,
        key: PRNGKey,
        choice_map: ChoiceMap,
    ) -> Tuple[VmapTrace, FloatArray]:
        self._static_check_broadcastable()
        broadcast_dim_length = self._static_broadcast_dim_length()
        index_array = jnp.arange(0, broadcast_dim_length)
        sub_keys = jax.random.split(key, broadcast_dim_length)

        def _importance(key, index, choice_map, args):
            submap = choice_map.get_submap(index)
            kernel_gen_fn = self.kernel(*args)
            return kernel_gen_fn.importance(key, submap)

        (tr, w) = jax.vmap(_importance, in_axes=(0, 0, None, self.in_axes))(
            sub_keys, index_array, choice_map, self.args
        )
        w = jnp.sum(w)
        retval = tr.get_retval()
        scores = tr.get_score()
        map_tr = VmapTrace(self, tr, retval, jnp.sum(scores))
        return (map_tr, w)

    def importance(
        self,
        key: PRNGKey,
        constraint: Constraint,
    ) -> Tuple[Trace, Weight]:
        match constraint:
            case ChoiceMap():
                choice_map: ChoiceMap = constraint
                return self.importance_choice_map(key, choice_map)

            case _:
                raise NotImplementedError

    def update_choice_map(
        self,
        key: PRNGKey,
        prev: VmapTrace,
        choice: ChoiceMap,
        argdiffs: Tuple,
    ) -> Tuple[Trace, Weight, Retdiff, ChoiceMap]:
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

    @typecheck
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        update_spec: UpdateSpec,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateSpec]:
        match update_spec:
            case ChoiceMap():
                choice_map: ChoiceMap = update_spec
                return self.update_choice_map(key, trace, choice_map)

            case _:
                raise NotImplementedError

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


#############
# Decorator #
#############


def vmap_combinator(
    in_axes: Tuple,
) -> Callable[[Callable], Callable[[Any], VmapCombinator]]:
    def decorator(f) -> Callable[[Any], VmapCombinator]:
        return lambda *args: VmapCombinator(args, f, in_axes)

    return decorator
