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
statically unrolled control flow for generative functions which can act as
kernels (a kernel generative function can accept their previous output as
input)."""

from dataclasses import dataclass

import jax
import jax.experimental.host_callback as hcb
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import EmptyChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.interpreters.staging import concrete_cond
from genjax._src.core.interpreters.staging import make_zero_trace
from genjax._src.core.transforms.incremental import tree_diff_primal
from genjax._src.core.typing import Any
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import Int
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import Union
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.builtin.builtin_gen_fn import (
    DeferredGenerativeFunctionCall,
)
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    VectorChoiceMap,
)
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    VectorTrace,
)
from genjax._src.generative_functions.combinators.vector.vector_tracetypes import (
    VectorTraceType,
)


#####
# UnfoldCombinator
#####


@dataclass
class UnfoldCombinator(GenerativeFunction):
    """
    > `UnfoldCombinator` accepts a kernel generative function, as well as a static maximum unroll length, and provides a scan-like pattern of generative computation.

    !!! info "Kernel generative functions"
        A kernel generative function is one which accepts and returns the same signature of arguments. Under the hood, `UnfoldCombinator` is implemented using `jax.lax.scan` - which has the same requirements.

    Examples:

        ```python exec="yes" source="tabbed-left"
        import jax
        import genjax
        console = genjax.pretty()

        # A kernel generative function.
        @genjax.gen
        def random_walk(prev):
            x = genjax.normal(prev, 1.0) @ "x"
            return x

        # Creating a `SwitchCombinator` via the preferred `new` class method.
        unfold = genjax.UnfoldCombinator.new(random_walk, 1000)

        init = 0.5
        key = jax.random.PRNGKey(314159)
        key, tr = jax.jit(genjax.simulate(unfold))(key, (999, init))

        print(console.render(tr))
        ```
    """

    max_length: IntArray
    kernel: GenerativeFunction

    def flatten(self):
        return (self.kernel,), (self.max_length,)

    @typecheck
    @classmethod
    def new(cls, kernel: GenerativeFunction, max_length: Int) -> "UnfoldCombinator":
        """
        The preferred constructor for `UnfoldCombinator` generative function instances. The shorthand symbol is `Unfold = UnfoldCombinator.new`.

        Arguments:
            kernel: A kernel `GenerativeFunction` instance.
            max_length: A static maximum possible unroll length.

        Returns:
            instance: An `UnfoldCombinator` instance.
        """
        return UnfoldCombinator(max_length, kernel)

    # This overloads the call functionality for this generative function
    # and allows usage of shorthand notation in the builtin DSL.
    def __call__(self, *args, **kwargs) -> DeferredGenerativeFunctionCall:
        return DeferredGenerativeFunctionCall.new(self, args, kwargs)

    @typecheck
    def get_trace_type(self, *args, **kwargs) -> VectorTraceType:
        _ = args[0]
        args = args[1:]
        inner_type = self.kernel.get_trace_type(*args, **kwargs)
        return VectorTraceType(inner_type, self.max_length)

    def _throw_bounds_host_exception(self, count: int):
        def _inner(count, _):
            raise Exception(
                f"\nUnfoldCombinator {self} received a length argument ({count}) longer than specified max length ({self.max_length})"
            )

        hcb.id_tap(
            lambda *args: _inner(*args),
            count,
            result=None,
        )
        return None

    @typecheck
    def simulate(self, key: PRNGKey, args: Tuple, **_) -> Tuple[PRNGKey, VectorTrace]:
        length = args[0]
        state = args[1]
        static_args = args[2:]

        # This inserts a host callback check for bounds checking.
        check = jnp.less(self.max_length, length + 1)
        concrete_cond(
            check,
            lambda *_: self._throw_bounds_host_exception(length + 1),
            lambda *_: None,
        )

        zero_trace = make_zero_trace(
            self.kernel,
            key,
            (state, *static_args),
        )

        def _inner_simulate(key, state, static_args, count):
            key, tr = self.kernel.simulate(key, (state, *static_args))
            state = tr.get_retval()
            score = tr.get_score()
            return (key, tr, state, count, count + 1, score)

        def _inner_zero_fallback(key, state, _, count):
            state = state
            score = 0.0
            return (key, zero_trace, state, -1, count, score)

        def _inner(carry, _):
            count, key, state = carry
            check = jnp.less(count, length + 1)
            key, tr, state, index, count, score = concrete_cond(
                check,
                _inner_simulate,
                _inner_zero_fallback,
                key,
                state,
                static_args,
                count,
            )

            return (count, key, state), (tr, index, state, score)

        (_, key, state), (tr, indices, retval, scores) = jax.lax.scan(
            _inner,
            (0, key, state),
            None,
            length=self.max_length,
        )

        unfold_tr = VectorTrace(self, indices, tr, args, retval, jnp.sum(scores))

        return key, unfold_tr

    # This checks the leaves of a choice map,
    # to determine if it is "out of bounds" for
    # the max static length of this combinator.
    def _static_bounds_check(self, v):
        lengths = []

        def _inner(v):
            if v.shape[-1] > self.max_length:
                raise Exception("Length of leaf longer than max length.")
            else:
                lengths.append(v.shape[-1])
                return v

        ret = jtu.tree_map(_inner, v)
        fixed_len = set(lengths)
        assert len(fixed_len) == 1
        return ret, fixed_len.pop()

    # This pads the leaves of a choice map up to
    # `self.max_length` -- so that we can scan
    # over the leading axes of the leaves.
    def _static_padder(self, v):
        ndim = len(v.shape)
        pad_axes = list(
            (0, self.max_length - len(v)) if k == 0 else (0, 0) for k in range(0, ndim)
        )
        return (
            np.pad(v, pad_axes) if isinstance(v, np.ndarray) else jnp.pad(v, pad_axes)
        )

    def _importance_indexed(self, key, chm, args):
        length = args[0]
        state = args[1]
        static_args = args[2:]

        # Unwrap the index mask.
        inner_choice_map = chm.inner
        target_index = chm.get_index()

        # Complicated - refactor in future.
        def _inner(carry, _):
            count, key, state = carry

            def _importance(key, state):
                return self.kernel.importance(
                    key, inner_choice_map, (state, *static_args)
                )

            def _simulate(key, state):
                key, tr = self.kernel.simulate(key, (state, *static_args))
                return key, (0.0, tr)

            check = count == target_index
            key, (w, tr) = concrete_cond(check, _importance, _simulate, key, state)
            check = jnp.less(count, length + 1)
            index = concrete_cond(check, lambda *_: count, lambda *_: -1)
            count, state, score, w = concrete_cond(
                check,
                lambda *args: (count + 1, tr.get_retval(), tr.get_score(), w),
                lambda *args: (count, state, 0.0, 0.0),
            )
            return (count, key, state), (w, score, tr, index, state)

        (count, key, state), (w, score, tr, indices, retval) = jax.lax.scan(
            _inner,
            (0, key, state),
            None,
            length=self.max_length,
        )

        unfold_tr = VectorTrace(self, indices, tr, args, retval, jnp.sum(score))

        w = jnp.sum(w)
        return key, (w, unfold_tr)

    def _importance_vcm(self, key, chm, args):
        length = args[0]
        state = args[1]
        static_args = args[2:]

        def _inner(carry, slice):
            count, key, state = carry
            chm = slice

            def _importance(key, chm, state):
                return self.kernel.importance(key, chm, (state, *static_args))

            def _simulate(key, chm, state):
                key, tr = self.kernel.simulate(key, (state, *static_args))
                return key, (0.0, tr)

            check = count == chm.get_index()
            key, (w, tr) = concrete_cond(
                check,
                _importance,
                _simulate,
                key,
                chm,
                state,
            )

            check = jnp.less(count, length + 1)
            index = concrete_cond(
                check,
                lambda *args: count,
                lambda *args: -1,
            )
            count, state, score, w = concrete_cond(
                check,
                lambda *args: (
                    count + 1,
                    tr.get_retval(),
                    tr.get_score(),
                    w,
                ),
                lambda *args: (count, state, 0.0, 0.0),
            )
            return (count, key, state), (w, score, tr, index, state)

        (count, key, state), (w, score, tr, indices, retval) = jax.lax.scan(
            _inner,
            (0, key, state),
            chm,
            length=self.max_length,
        )

        unfold_tr = VectorTrace(self, indices, tr, args, retval, jnp.sum(score))

        w = jnp.sum(w)
        return key, (w, unfold_tr)

    def _importance_fallback(self, key, chm, args):
        length = args[0]
        state = args[1]
        static_args = args[2:]

        # Check incoming choice map, and coerce to `VectorChoiceMap`
        # before passing into scan calls.
        chm, fixed_len = self._static_bounds_check(chm)
        chm = jtu.tree_map(
            self._static_padder,
            chm,
        )
        if not isinstance(chm, VectorChoiceMap):
            indices = np.array(
                [ind if ind < fixed_len else -1 for ind in range(0, self.max_length)]
            )
            chm = VectorChoiceMap(
                indices,
                chm,
            )

        def _inner(carry, slice):
            count, key, state = carry
            chm = slice

            def _importance(key, chm, state):
                return self.kernel.importance(key, chm, (state, *static_args))

            def _simulate(key, chm, state):
                key, tr = self.kernel.simulate(key, (state, *static_args))
                return key, (0.0, tr)

            check = count == chm.get_index()
            key, (w, tr) = concrete_cond(
                check,
                _importance,
                _simulate,
                key,
                chm,
                state,
            )

            check = jnp.less(count, length + 1)
            index = concrete_cond(
                check,
                lambda *args: count,
                lambda *args: -1,
            )
            count, state, score, weight = concrete_cond(
                check,
                lambda *args: (
                    count + 1,
                    tr.get_retval(),
                    tr.get_score(),
                    w,
                ),
                lambda *args: (count, state, 0.0, 0.0),
            )
            return (count, key, state), (w, score, tr, index, state)

        (count, key, state), (w, score, tr, indices, retval) = jax.lax.scan(
            _inner,
            (0, key, state),
            chm,
            length=self.max_length,
        )

        unfold_tr = VectorTrace(self, indices, tr, args, retval, jnp.sum(score))

        w = jnp.sum(w)
        return key, (w, unfold_tr)

    @typecheck
    def importance(
        self, key: PRNGKey, chm: ChoiceMap, args: Tuple, **kwargs
    ) -> Tuple[PRNGKey, Tuple[FloatArray, VectorTrace]]:
        length = args[0]

        # This inserts a host callback check for bounds checking.
        # At runtime, if the bounds are exceeded -- an error
        # will be emitted.
        check = jnp.less(self.max_length, length + 1)
        concrete_cond(
            check,
            lambda *args: self._throw_bounds_host_exception(length + 1),
            lambda *args: None,
        )

        if isinstance(chm, VectorChoiceMap):
            return self._importance_vcm(key, chm, args)
        else:
            return self._importance_fallback(key, chm, args)

    # The choice map is a vector choice map.
    def _update_vcm(self, key, prev, chm, argdiffs):
        length = argdiffs[0]
        state = argdiffs[1]
        static_args = argdiffs[2:]
        args = tree_diff_primal(argdiffs)

        def _inner(carry, slice):
            count, key, state = carry
            (prev, chm) = slice

            def _update(key, prev, chm, state):
                return self.kernel.update(key, prev, chm, (state, *static_args))

            def _fallthrough(key, prev, chm, state):
                return self.kernel.update(
                    key, prev, EmptyChoiceMap(), (state, *static_args)
                )

            check = count == chm.get_index()
            key, (retdiff, w, tr, discard) = concrete_cond(
                check, _update, _fallthrough, key, prev, chm, state
            )

            check = jnp.less(count, length + 1)
            index = concrete_cond(check, lambda *args: count, lambda *args: -1)
            count, state, score, weight = concrete_cond(
                check,
                lambda *args: (count + 1, retdiff, tr.get_score(), w),
                lambda *args: (count, state, 0.0, 0.0),
            )
            return (count, key, state), (state, score, w, tr, discard, index)

        (count, key, state), (
            retdiff,
            score,
            w,
            tr,
            discard,
            indices,
        ) = jax.lax.scan(_inner, (0, key, state), (prev, chm), length=self.max_length)

        unfold_tr = VectorTrace(
            self, indices, tr, args, retdiff.get_val(), jnp.sum(score)
        )

        w = jnp.sum(w)
        return key, (retdiff, w, unfold_tr, discard)

    # The choice map doesn't carry optimization info.
    def _update_fallback(self, key, prev, chm, argdiffs):
        length = argdiffs[0]
        state = argdiffs[1]
        static_args = argdiffs[2:]
        args = tuple(map(lambda v: v.get_val(), argdiffs))

        # Check incoming choice map, and coerce to `VectorChoiceMap`
        # before passing into scan calls.
        self._static_bounds_check(chm)
        chm = jtu.tree_map(
            self._static_padder,
            chm,
        )
        chm = VectorChoiceMap(
            np.array([ind for ind in range(0, self.max_length)]),
            chm,
        )

        # The actual semantics of update are carried out by a scan
        # call.

        def _inner(carry, slice):
            count, key, state = carry
            (prev, chm) = slice

            def _update(key, prev, chm, state):
                return self.kernel.update(key, prev, chm, (state, *static_args))

            def _fallthrough(key, prev, chm, state):
                return self.kernel.update(
                    key, prev, EmptyChoiceMap(), (state, *static_args)
                )

            check = count == chm.get_index()
            key, (retdiff, w, tr, discard) = concrete_cond(
                check, _update, _fallthrough, key, prev, chm, state
            )

            check = jnp.less(count, length + 1)
            index = concrete_cond(check, lambda *args: count, lambda *args: -1)
            count, state, score, weight = concrete_cond(
                check,
                lambda *args: (count + 1, retdiff, tr.get_score(), w),
                lambda *args: (count, state, 0.0, 0.0),
            )
            return (count, key, state), (state, score, w, tr, discard, index)

        (_, key, _), (retdiff, score, w, tr, discard, indices) = jax.lax.scan(
            _inner, (0, key, state), (prev, chm), length=self.max_length
        )

        unfold_tr = VectorTrace(
            self, indices, tr, args, retdiff.get_val(), jnp.sum(score)
        )

        w = jnp.sum(w)
        return key, (retdiff, w, unfold_tr, discard)

    @typecheck
    def update(
        self,
        key: PRNGKey,
        prev: Trace,
        chm: Union[EmptyChoiceMap, VectorChoiceMap],
        argdiffs: Tuple,
        **_,
    ) -> Tuple[PRNGKey, Tuple[Any, FloatArray, VectorTrace, ChoiceMap]]:
        length = argdiffs[0].get_val()

        # Unwrap the previous trace at this address
        # we should get a `VectorChoiceMap`.
        # We don't need the index indicators from the trace,
        # so we can just unwrap it.
        prev = prev.inner

        # This inserts a host callback check for bounds checking.
        # If we go out of bounds on device, it throws to the
        # Python runtime -- which will raise.
        check = jnp.less(self.max_length, length + 1)
        concrete_cond(
            check,
            lambda *args: self._throw_bounds_host_exception(length + 1),
            lambda *args: None,
        )

        # Branches here implement certain optimizations when more
        # information about the passed in choice map is available.
        #
        # The fallback just inflates a choice map to the right shape
        # and runs a generic update.
        if isinstance(chm, VectorChoiceMap):
            return self._update_vcm(key, prev, chm, argdiffs)
        else:
            return self._update_empty(key, prev, chm, argdiffs)

    def _throw_index_check_host_exception(self, count: IntArray, index: IntArray):
        def _inner(pair, transforms):
            (count, index) = pair
            raise Exception(
                f"\nUnfoldCombinator {self} received a choice map with mismatched indices (count {count}, at index {index}) in assess."
            )

        hcb.id_tap(
            lambda *args: _inner(*args),
            (count, index),
            result=None,
        )
        return None

    @typecheck
    def assess(
        self, key: PRNGKey, chm: ChoiceMap, args: Tuple, **kwargs
    ) -> Tuple[PRNGKey, Tuple[Any, FloatArray]]:
        assert isinstance(chm, VectorChoiceMap)
        length = args[0]

        # This inserts a host callback check for bounds checking.
        # At runtime, if the bounds are exceeded -- an error
        # will be emitted.
        check = jnp.less(self.max_length, length + 1)
        concrete_cond(
            check,
            lambda *args: self._throw_bounds_host_exception(length + 1),
            lambda *args: None,
        )

        length = args[0]
        state = args[1]
        static_args = args[2:]

        def _inner(carry, slice):
            count, key, state = carry
            chm = slice

            check = count == chm.get_index()

            # TODO: the below doesn't work when `vmap` is used.
            # This inserts a host callback check for bounds checking.
            # If there is an index failure, `assess` must fail
            # because we must provide a constraint for every generative
            # function call.
            # jax.lax.cond(
            #    check,
            #    lambda *args: None,
            #    lambda *args: self._throw_index_check_host_exception(*args),
            #    count,
            #    chm.get_index(),
            # )

            key, (retval, score) = self.kernel.assess(key, chm, (state, *static_args))

            check = jnp.less(count, length + 1)
            index = concrete_cond(
                check,
                lambda *args: count,
                lambda *args: -1,
            )
            count, state, score = concrete_cond(
                check,
                lambda *args: (count + 1, retval, score),
                lambda *args: (count, state, 0.0),
            )
            return (count, key, state), (state, score, index)

        (_, key, state), (retval, score, _) = jax.lax.scan(
            _inner,
            (0, key, state),
            chm,
            length=self.max_length,
        )

        score = jnp.sum(score)
        return key, (retval, score)


##############
# Shorthands #
##############

Unfold = UnfoldCombinator.new
