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
    Sample,
    ChoiceMap,
    Constraint,
    GenerativeFunction,
    GenerativeFunctionClosure,
    Retdiff,
    Score,
    Selection,
    Trace,
    UpdateSpec,
    Weight,
)
from genjax._src.core.generative.choice_map import RemoveSelectionUpdateSpec
from genjax._src.core.pytree import Pytree
from genjax._src.core.traceback_util import register_exclusion
from genjax._src.core.typing import (
    Any,
    Callable,
    FloatArray,
    Optional,
    PRNGKey,
    Tuple,
    typecheck,
)

register_exclusion(__file__)


@Pytree.dataclass
class VmapTrace(Trace):
    gen_fn: GenerativeFunction
    inner: Trace
    args: Tuple
    retval: Any
    score: FloatArray

    def get_args(self) -> Tuple:
        return self.args

    def get_retval(self):
        return self.retval

    def get_gen_fn(self):
        return self.gen_fn

    def get_sample(self):
        return jax.vmap(lambda idx, subtrace: ChoiceMap.a(idx, subtrace.get_sample()))(
            jnp.arange(len(self.inner.get_score())),
            self.inner,
        )

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

        ##############################################################
        # One way to create a `VmapCombinator`: using the decorator. #
        ##############################################################


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

    gen_fn: GenerativeFunction
    in_axes: Tuple = Pytree.static()

    def __abstract_call__(self, *args) -> Any:
        def inner(*args):
            return self.gen_fn.__abstract_call__(*args)

        return jax.vmap(inner, in_axes=self.in_axes)(*args)

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

        def inner(key, args):
            tr = self.gen_fn.simulate(key, args)
            return tr

        tr = jax.vmap(inner, (0, self.in_axes))(sub_keys, args)

        retval = tr.get_retval()
        scores = tr.get_score()
        map_tr = VmapTrace(self, tr, args, retval, jnp.sum(scores))
        return map_tr

    def importance_choice_map(
        self,
        key: PRNGKey,
        choice_map: ChoiceMap,
        args: Tuple,
    ) -> Tuple[VmapTrace, FloatArray, UpdateSpec]:
        self._static_check_broadcastable(args)
        broadcast_dim_length = self._static_broadcast_dim_length(args)
        idx_array = jnp.arange(0, broadcast_dim_length)
        sub_keys = jax.random.split(key, broadcast_dim_length)

        def _importance(key, idx, choice_map, args):
            submap = choice_map.get_submap(idx)
            tr, w, bwd_spec = self.gen_fn.importance(key, submap, args)
            return tr, w, ChoiceMap.a(idx, bwd_spec)

        (tr, w, bwd_spec) = jax.vmap(_importance, in_axes=(0, 0, None, self.in_axes))(
            sub_keys, idx_array, choice_map, args
        )
        w = jnp.sum(w)
        retval = tr.get_retval()
        scores = tr.get_score()
        map_tr = VmapTrace(self, tr, args, retval, jnp.sum(scores))
        return map_tr, w, bwd_spec

    def importance(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: Tuple,
    ) -> Tuple[Trace, Weight, UpdateSpec]:
        match constraint:
            case ChoiceMap():
                choice_map: ChoiceMap = constraint
                return self.importance_choice_map(key, choice_map, args)

            case _:
                raise NotImplementedError

    def update_choice_map(
        self,
        key: PRNGKey,
        prev: VmapTrace,
        choice: ChoiceMap,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, ChoiceMap]:
        pass

    def update_remove_selection(
        self,
        key: PRNGKey,
        trace: VmapTrace,
        selection: Selection,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, ChoiceMap]:
        self._static_check_broadcastable()
        broadcast_dim_length = self._static_broadcast_dim_length()
        idx_array = jnp.arange(0, broadcast_dim_length)
        sub_keys = jax.random.split(key, broadcast_dim_length)

        def _update(key, idx, args):
            subselection = selection.step(idx)
            kernel_gen_fn = self.kernel(*args)
            sub_spec = RemoveSelectionUpdateSpec(subselection)
            tr, w, retdiff, bwd_spec = kernel_gen_fn.update(key, trace, sub_spec)
            return tr, w, retdiff, ChoiceMap.a(idx, bwd_spec)

        tr, w, retdiff, bwd_specs = jax.vmap(_update, in_axes=(0, 0, self.in_axes))(
            sub_keys, idx_array, self.args
        )
        w = jnp.sum(w)
        retval = tr.get_retval()
        scores = tr.get_score()
        map_tr = VmapTrace(self, tr, retval, jnp.sum(scores))
        return map_tr, w, retdiff, bwd_specs

    @typecheck
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        update_spec: UpdateSpec,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateSpec]:
        match update_spec:
            case ChoiceMap():
                return self.update_choice_map(key, trace, update_spec, argdiffs)

            case RemoveSelectionUpdateSpec(selection):
                return self.update_remove_selection(key, trace, selection, argdiffs)

            case _:
                raise Exception(f"Not implemented spec: {update_spec}")

    @typecheck
    def assess(
        self,
        sample: Sample,
        args: Tuple,
    ) -> Tuple[Score, Any]:
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
    gen_fn: Optional[GenerativeFunctionClosure] = None,
    /,
    *,
    in_axes: Tuple,
) -> Callable | VmapCombinator:
    def decorator(gen_fn) -> VmapCombinator:
        return VmapCombinator(gen_fn, in_axes)

    if gen_fn:
        return decorator(gen_fn)
    else:
        return decorator
