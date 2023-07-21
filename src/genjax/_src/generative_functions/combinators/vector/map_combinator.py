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
"""This module implements a generative function combinator which allows
broadcasting for generative functions -- mapping over vectorial versions of
their arguments."""

from dataclasses import dataclass

import jax
import jax.experimental.host_callback as hcb
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from genjax._src.core.datatypes.generative import AllSelection
from genjax._src.core.datatypes.generative import EmptyChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import NoneSelection
from genjax._src.core.datatypes.generative import Selection
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.tracetypes import TraceType
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.interpreters.staging import concrete_cond
from genjax._src.core.transforms.incremental import tree_diff_primal
from genjax._src.core.typing import Any
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import Union
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.builtin.builtin_gen_fn import SupportsBuiltinSugar
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    ComplementIndexSelection,
)
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    IndexChoiceMap,
)
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    IndexSelection,
)
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    VectorChoiceMap,
)
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    VectorSelection,
)
from genjax._src.generative_functions.combinators.vector.vector_tracetypes import (
    VectorTraceType,
)
from genjax._src.utilities import slash


#####
# Map trace
#####


@dataclass
class MapTrace(Trace):
    gen_fn: GenerativeFunction
    inner: Trace
    args: Tuple
    retval: Any
    score: FloatArray

    def flatten(self):
        return (
            self.gen_fn,
            self.inner,
            self.args,
            self.retval,
            self.score,
        ), ()

    def get_args(self):
        return self.args

    def get_choices(self):
        return VectorChoiceMap.new(self.inner)

    def get_gen_fn(self):
        return self.gen_fn

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def project(self, selection: Selection) -> FloatArray:
        if isinstance(selection, VectorSelection):
            return jnp.sum(self.inner.project(selection.inner))
        elif isinstance(selection, IndexSelection) or isinstance(
            selection, ComplementIndexSelection
        ):
            inner_project = self.inner.project(selection.inner)
            return jnp.sum(
                jnp.take(inner_project, selection.indices, mode="fill", fill_value=0.0)
            )
        elif isinstance(selection, AllSelection):
            return self.score
        elif isinstance(selection, NoneSelection):
            return 0.0
        else:
            selection = VectorSelection.new(selection)
            return self.project(selection)


#####
# Map
#####


@dataclass
class MapCombinator(GenerativeFunction, SupportsBuiltinSugar):
    """> `MapCombinator` accepts a generative function as input and provides
    `vmap`-based implementations of the generative function interface methods.

    Examples:

        ```python exec="yes" source="tabbed-left"
        import jax
        import jax.numpy as jnp
        import genjax
        console = genjax.pretty()

        @genjax.gen
        def add_normal_noise(x):
            noise1 = genjax.normal(0.0, 1.0) @ "noise1"
            noise2 = genjax.normal(0.0, 1.0) @ "noise2"
            return x + noise1 + noise2


        # Creating a `MapCombinator` via the preferred `new` class method.
        mapped = genjax.MapCombinator.new(add_normal_noise, in_axes=(0,))

        key = jax.random.PRNGKey(314159)
        arr = jnp.ones(100)
        key, tr = jax.jit(genjax.simulate(mapped))(key, (arr, ))

        print(console.render(tr))
        ```
    """

    in_axes: Tuple
    repeats: Union[None, IntArray]
    kernel: GenerativeFunction

    def flatten(self):
        return (self.kernel,), (self.in_axes, self.repeats)

    @typecheck
    @classmethod
    def new(
        cls,
        kernel: GenerativeFunction,
        in_axes: Union[None, Tuple] = None,
        repeats: Union[None, IntArray] = None,
    ) -> "MapCombinator":
        """The preferred constructor for `MapCombinator` generative function
        instances. The shorthand symbol is `Map = MapCombinator.new`.

        Arguments:
            kernel: A single `GenerativeFunction` instance.
            in_axes: A tuple specifying which `args` to broadcast over.
            repeats: An integer specifying the length of repetitions (ignored if `in_axes` is specified, if `in_axes` is not specified - required).

        Returns:
            instance: A `MapCombinator` instance.
        """
        assert isinstance(kernel, GenerativeFunction)
        if in_axes is None or all(map(lambda v: v is None, in_axes)):
            assert repeats is not None
        return MapCombinator(in_axes, repeats, kernel)

    def _static_check_broadcastable(self, args):
        # Argument broadcast semantics must be fully specified
        # in `in_axes` or via `self.repeats`.
        assert self.repeats or (len(args) == len(self.in_axes))

    def _static_broadcast_dim_length(self, args):
        def find_axis_size(axis, x):
            if axis is not None:
                leaves = jax.tree_util.tree_leaves(x)
                if leaves:
                    return leaves[0].shape[axis]
            return ()

        axis_sizes = jax.tree_util.tree_map(find_axis_size, self.in_axes, args)
        axis_sizes = set(jax.tree_util.tree_leaves(axis_sizes))
        if self.repeats is None and len(axis_sizes) == 1:
            (d_axis_size,) = axis_sizes
        elif len(axis_sizes) > 1:
            raise ValueError(f"Inconsistent batch axis sizes: {axis_sizes}")
        elif self.repeats is None:
            raise ValueError("repeats should be specified manually.")
        else:
            d_axis_size = self.repeats
        return d_axis_size

    @typecheck
    def get_trace_type(self, *args, **kwargs) -> TraceType:
        broadcast_dim_length = self._static_broadcast_dim_length(args)
        kernel_tt = self.kernel.get_trace_type(*args)
        return VectorTraceType(kernel_tt, broadcast_dim_length)

    @typecheck
    def simulate(self, key: PRNGKey, args: Tuple, **kwargs) -> Tuple[PRNGKey, MapTrace]:
        self._static_check_broadcastable(args)
        broadcast_dim_length = self._static_broadcast_dim_length(args)
        key, sub_keys = slash(key, broadcast_dim_length)
        _, tr = jax.vmap(self.kernel.simulate, in_axes=(0, self.in_axes))(
            sub_keys, args
        )
        retval = tr.get_retval()
        scores = tr.get_score()
        map_tr = MapTrace(self, tr, args, retval, jnp.sum(scores))
        return key, map_tr

    # Implementation specialized to `VectorChoiceMap`.
    @dispatch
    def importance(
        self,
        key: PRNGKey,
        chm: VectorChoiceMap,
        args: Tuple,
        **_,
    ):
        def _importance(key, chm, args):
            return self.kernel.importance(key, chm, args)

        def _simulate(key, _, args):
            key, tr = self.kernel.simulate(key, args)
            return key, (0.0, tr)

        def _switch(key, flag, chm, args):
            return concrete_cond(flag, _importance, _simulate, key, chm, args)

        self._static_check_broadcastable(args)
        broadcast_dim_length = self._static_broadcast_dim_length(args)
        key, sub_keys = slash(key, broadcast_dim_length)

        inner = chm.inner
        _, (w, tr) = jax.vmap(_importance, in_axes=(0, 0, self.in_axes))(
            sub_keys, inner, args
        )

        w = jnp.sum(w)
        retval = tr.get_retval()
        scores = tr.get_score()
        map_tr = MapTrace(self, tr, args, retval, scores)
        return key, (w, map_tr)

    # Implementation specialized to `IndexChoiceMap`.
    @dispatch
    def importance(
        self,
        key: PRNGKey,
        chm: IndexChoiceMap,
        args: Tuple,
        **_,
    ):
        self._static_check_broadcastable(args)
        broadcast_dim_length = self._static_broadcast_dim_length(args)
        index_array = jnp.arange(0, broadcast_dim_length)
        flag_array = (index_array[:, None] == chm.indices).sum(axis=-1)
        key, sub_keys = slash(key, broadcast_dim_length)

        def _simulate(key, index, chm, args):
            key, tr = self.kernel.simulate(key, args)
            return key, (0.0, tr)

        def _importance(key, index, chm, args):
            (slice_index,) = jnp.nonzero(index == chm, size=1)
            slice_index = slice_index[0]
            inner_chm = chm.inner.slice(slice_index)
            return self.kernel.importance(key, inner_chm, args)

        def _switch(key, flag, index, chm, args):
            return concrete_cond(flag, _importance, _simulate, key, index, chm, args)

        _, (w, tr) = jax.vmap(_switch, in_axes=(0, 0, 0, None, self.in_axes))(
            sub_keys, flag_array, index_array, chm, args
        )
        w = jnp.sum(w)
        retval = tr.get_retval()
        scores = tr.get_score()
        map_tr = MapTrace(self, tr, args, retval, jnp.sum(scores))
        return key, (w, map_tr)

    @dispatch
    def importance(
        self,
        key: PRNGKey,
        chm: EmptyChoiceMap,
        args: Tuple,
        **_,
    ) -> Tuple[PRNGKey, Tuple[FloatArray, MapTrace]]:
        key, map_tr = self.simulate(key, args)
        w = 0.0
        return key, (w, map_tr)

    # Fallback implementation if a generic Trie is passed in.
    @dispatch
    def importance(
        self,
        key: PRNGKey,
        chm: Trie,
        args: Tuple,
        **_,
    ) -> Tuple[PRNGKey, Tuple[FloatArray, MapTrace]]:
        indchm = IndexChoiceMap.convert(chm)
        return self.importance(key, indchm, args)

    # The choice map passed in here is a vector choice map.
    @dispatch
    def update(
        self,
        key: PRNGKey,
        prev: MapTrace,
        chm: VectorChoiceMap,
        argdiffs: Tuple,
    ):
        args = tree_diff_primal(argdiffs)
        self._static_check_broadcastable(args)
        broadcast_dim_length = self._static_broadcast_dim_length(args)
        prev_inaxes_tree = jtu.tree_map(
            lambda v: None if v.shape == () else 0, prev.inner
        )
        key, sub_keys = slash(key, broadcast_dim_length)

        _, (retdiff, w, tr, discard) = jax.vmap(
            self.kernel.update, in_axes=(0, prev_inaxes_tree, 0, self.in_axes)
        )(sub_keys, prev.inner, chm.inner, argdiffs)
        w = jnp.sum(w)
        retval = tr.get_retval()
        scores = tr.get_score()
        map_tr = MapTrace(self, tr, args, retval, jnp.sum(scores))
        discard = VectorChoiceMap(discard)
        return key, (retdiff, w, map_tr, discard)

    # The choice map passed in here is empty, but perhaps
    # the arguments have changed.
    @dispatch
    def update(
        self,
        key: PRNGKey,
        prev: MapTrace,
        chm: EmptyChoiceMap,
        argdiffs: Tuple,
    ):
        def _fallback(key, prev, chm, argdiffs):
            key, (retdiff, w, tr, d) = self.kernel.update(
                key, prev, EmptyChoiceMap(), argdiffs
            )
            return key, (retdiff, w, tr, d)

        prev_inaxes_tree = jtu.tree_map(
            lambda v: None if v.shape == () else 0, prev.inner
        )
        args = jtu.tree_leaves(argdiffs)
        self._static_check_broadcastable(args)
        broadcast_dim_length = self._static_broadcast_dim_length(args)
        key, sub_keys = slash(key, broadcast_dim_length)
        _, (retdiff, w, tr, discard) = jax.vmap(
            _fallback, in_axes=(0, prev_inaxes_tree, 0, self.in_axes)
        )(sub_keys, prev.inner, chm, argdiffs)
        w = jnp.sum(w)
        retval = tr.get_retval()
        map_tr = MapTrace(self, tr, args, retval, jnp.sum(tr.get_score()))
        return key, (retdiff, w, map_tr, discard)

    # TODO: I've had so many issues with getting this to work correctly
    # and not throw - and I'm not sure why it's been so finicky.
    # Investigate if it occurs again.
    def _throw_index_check_host_exception(
        self, check, truth: IntArray, index: IntArray
    ):
        def _inner(args, _):
            truth = args[0]
            index = args[1]
            check = args[2]
            if not np.all(check):
                raise Exception(
                    f"\nMapCombinator {self} received a choice map with mismatched indices in assess.\nReference:\n{truth}\nPassed in:\n{index}"
                )

        hcb.id_tap(
            _inner,
            (truth, index, check),
            result=None,
        )
        return None

    @typecheck
    def assess(
        self, key: PRNGKey, chm: VectorChoiceMap, args: Tuple, **kwargs
    ) -> Tuple[PRNGKey, Tuple[Any, FloatArray]]:
        self._static_check_broadcastable(args)
        broadcast_dim_length = self._static_broadcast_dim_length(args)
        indices = jnp.array([i for i in range(0, broadcast_dim_length)])
        check = jnp.count_nonzero(indices - chm.get_index()) == 0

        # This inserts a host callback check for bounds checking.
        # If there is an index failure, `assess` must fail
        # because we must provide a constraint for every generative
        # function call.
        self._throw_index_check_host_exception(check, indices, chm.get_index())

        inner = chm.inner
        key, sub_keys = slash(key, broadcast_dim_length)
        _, (retval, score) = jax.vmap(self.kernel.assess, in_axes=(0, 0, self.in_axes))(
            sub_keys, inner, args
        )
        return key, (retval, jnp.sum(score))


##############
# Shorthands #
##############

Map = MapCombinator.new
