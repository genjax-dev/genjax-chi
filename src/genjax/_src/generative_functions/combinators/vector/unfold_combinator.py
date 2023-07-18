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

from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import EmptyChoiceMap
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.interpreters.staging import concrete_cond
from genjax._src.core.interpreters.staging import make_zero_trace
from genjax._src.core.transforms.incremental import Diff
from genjax._src.core.transforms.incremental import UnknownChange
from genjax._src.core.transforms.incremental import diff
from genjax._src.core.transforms.incremental import static_check_no_change
from genjax._src.core.transforms.incremental import tree_diff_primal
from genjax._src.core.typing import Any
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import Int
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.builtin.builtin_gen_fn import SupportsBuiltinSugar
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    IndexChoiceMap,
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
class UnfoldCombinator(GenerativeFunction, SupportsBuiltinSugar):
    """> `UnfoldCombinator` accepts a kernel generative function, as well as a
    static maximum unroll length, and provides a scan-like pattern of
    generative computation.

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
        """The preferred constructor for `UnfoldCombinator` generative function
        instances. The shorthand symbol is `Unfold = UnfoldCombinator.new`.

        Arguments:
            kernel: A kernel `GenerativeFunction` instance.
            max_length: A static maximum possible unroll length.

        Returns:
            instance: An `UnfoldCombinator` instance.
        """
        return UnfoldCombinator(max_length, kernel)

    @typecheck
    def get_trace_type(self, *args, **kwargs) -> VectorTraceType:
        _ = args[0]
        args = args[1:]
        inner_type = self.kernel.get_trace_type(*args, **kwargs)
        return VectorTraceType(inner_type, self.max_length)

    def _runtime_throw_bounds_exception(self, count: int):
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

    def _runtime_check_bounds(self, args):
        length = args[0]

        # This inserts a host callback check for bounds checking.
        # At runtime, if the bounds are exceeded -- an error
        # will be emitted.
        check = jnp.less(self.max_length, length + 1)
        concrete_cond(
            check,
            lambda *args: self._runtime_throw_bounds_exception(length + 1),
            lambda *args: None,
        )

    @typecheck
    def get_trace_type(self, *args, **kwargs) -> VectorTraceType:
        _ = args[0]
        args = args[1:]
        inner_type = self.kernel.get_trace_type(*args, **kwargs)
        return VectorTraceType(inner_type, self.max_length)

    @typecheck
    def simulate(self, key: PRNGKey, args: Tuple, **_) -> Tuple[PRNGKey, VectorTrace]:
        self._runtime_check_bounds(args)
        length = args[0]
        state = args[1]
        static_args = args[2:]

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

        unfold_tr = VectorTrace(self, tr, args, retval, jnp.sum(scores))

        return key, unfold_tr

    @dispatch
    def importance(
        self,
        key: PRNGKey,
        chm: IndexChoiceMap,
        args: Tuple,
    ):
        self._runtime_check_bounds(args)
        length = args[0]
        state = args[1]
        static_args = args[2:]

        def _inner(carry, _):
            count, key, state = carry
            sub_choice_map = chm.get_subtree(count)
            key, (w, tr) = self.kernel.importance(
                key, sub_choice_map, (state, *static_args)
            )

            check = jnp.less(count, length + 1)
            count, state, score, w = concrete_cond(
                check,
                lambda *args: (count + 1, tr.get_retval(), tr.get_score(), w),
                lambda *args: (count, state, 0.0, 0.0),
            )
            return (count, key, state), (w, score, tr, state)

        (count, key, state), (w, score, tr, retval) = jax.lax.scan(
            _inner,
            (0, key, state),
            None,
            length=self.max_length,
        )

        unfold_tr = VectorTrace(self, tr, args, retval, jnp.sum(score))

        w = jnp.sum(w)
        return key, (w, unfold_tr)

    @dispatch
    def importance(
        self,
        key: PRNGKey,
        chm: VectorChoiceMap,
        args: Tuple,
    ):
        self._runtime_check_bounds(args)
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

            check_count = jnp.less(count, length + 1)
            key, (w, tr) = concrete_cond(
                check_count,
                _importance,
                _simulate,
                key,
                chm,
                state,
            )

            count, state, score, w = concrete_cond(
                check_count,
                lambda *args: (
                    count + 1,
                    tr.get_retval(),
                    tr.get_score(),
                    w,
                ),
                lambda *args: (count, state, 0.0, 0.0),
            )
            return (count, key, state), (w, score, tr, state)

        (_, key, state), (w, score, tr, retval) = jax.lax.scan(
            _inner,
            (0, key, state),
            chm,
            length=self.max_length,
        )

        unfold_tr = VectorTrace(self, tr, args, retval, jnp.sum(score))

        w = jnp.sum(w)
        return key, (w, unfold_tr)

    @dispatch
    def importance(
        self,
        key: PRNGKey,
        _: EmptyChoiceMap,
        args: Tuple,
    ):
        self._runtime_check_bounds(args)
        key, unfold_tr = self.simulate(key, args)
        w = 0.0
        return key, (w, unfold_tr)

    @dispatch
    def _update_fallback(
        self,
        key: PRNGKey,
        prev: VectorTrace,
        chm: IndexChoiceMap,
        length: Diff,
        state: Diff,
        *static_args: Diff,
    ):
        raise NotImplementedError

    @dispatch
    def _update_fallback(
        self,
        key: PRNGKey,
        prev: VectorTrace,
        chm: VectorChoiceMap,
        length: Diff,
        state: Diff,
        *static_args: Diff,
    ):
        length, state, static_args = tree_diff_primal((length, state, static_args))

        def _inner(carry, slice):
            count, key, state = carry
            (prev, chm) = slice

            key, (retdiff, w, tr, discard) = self.kernel.update(
                key, prev, chm, (state, *static_args)
            )

            check = jnp.less(count, length + 1)
            count, state, score, weight = concrete_cond(
                check,
                lambda *args: (count + 1, retdiff, tr.get_score(), w),
                lambda *args: (count, state, 0.0, 0.0),
            )
            return (count, key, state), (state, score, weight, tr, discard)

        (count, key, state), (retdiff, score, w, tr, discard) = jax.lax.scan(
            _inner, (0, key, state), (prev, chm), length=self.max_length
        )

        unfold_tr = VectorTrace(
            self,
            tr,
            (length, state, *static_args),
            tree_diff_primal(retdiff),
            jnp.sum(score),
        )

        w = jnp.sum(w)
        return key, (retdiff, w, unfold_tr, discard)

    @dispatch
    def _update_specialized(
        self,
        key: PRNGKey,
        prev: VectorTrace,
        chm: IndexChoiceMap,
        length: Diff,
        state: Any,
        *static_args: Any,
    ):
        new_upper = tree_diff_primal(length)
        start_lower = jnp.min(chm.indices)

        # TODO: `UnknownChange` is used here
        # to preserve the Pytree structure across the loop.
        state_diff = jtu.tree_map(
            lambda v: diff(v[start_lower], UnknownChange),
            prev.get_retval(),
        )
        prev_inner_trace = prev.inner

        def _inner(index, state):
            (key, w, state_diff, prev) = state
            sub_chm = chm.get_subtree(index)
            prev_slice = jtu.tree_map(lambda v: v[index], prev)
            key, (state_diff, idx_w, new_tr, _) = self.kernel.update(
                key, prev_slice, sub_chm, (state_diff, *static_args)
            )

            def _mutate(prev, new):
                new = prev.at[index].set(new)
                return new

            # TODO: also handle discard.
            prev = jtu.tree_map(_mutate, prev, new_tr)
            w = w + idx_w

            # TODO: c.f. message above on `UnknownChange`.
            # Preserve the diff type across the loop
            # iterations.
            primal_state = tree_diff_primal(state_diff)
            state_diff = jtu.tree_map(
                lambda v: diff(v, UnknownChange),
                primal_state,
            )

            return (key, w, state_diff, prev)

        # TODO: add discard.
        (key, w, state_diff, new_inner_trace) = jax.lax.fori_loop(
            start_lower,
            new_upper,
            _inner,
            (key, 0.0, state_diff, prev_inner_trace),
        )
        retval = new_inner_trace.get_retval()
        scores = new_inner_trace.get_score()
        args = tree_diff_primal((length, state, *static_args))

        new_tr = VectorTrace(self, new_inner_trace, args, retval, jnp.sum(scores))
        return key, (state_diff, w, new_tr, EmptyChoiceMap())

    @dispatch
    def _update_specialized(
        self,
        key: PRNGKey,
        prev: VectorTrace,
        chm: VectorChoiceMap,
        length: Diff,
        state: Any,
        *static_args: Any,
    ):
        raise NotImplementedError

    @dispatch
    def update(
        self,
        key: PRNGKey,
        prev: VectorTrace,
        chm: ChoiceMap,
        argdiffs: Tuple,
    ):
        length = argdiffs[0]
        state = argdiffs[1]
        static_args = argdiffs[2:]
        args = tree_diff_primal(argdiffs)
        self._runtime_check_bounds(args)
        check_state_static_no_change = static_check_no_change((state, static_args))
        if check_state_static_no_change:
            return self._update_specialized(
                key,
                prev,
                chm,
                length,
                state,
                *static_args,
            )
        else:
            return self._update_fallback(
                key,
                prev,
                chm,
                length,
                state,
                *static_args,
            )

    @dispatch
    def assess(
        self,
        key: PRNGKey,
        chm: VectorChoiceMap,
        args: Tuple,
        **kwargs,
    ) -> Tuple[PRNGKey, Tuple[Any, FloatArray]]:
        self._runtime_check_bounds(args)
        length = args[0]
        state = args[1]
        static_args = args[2:]

        def _inner(carry, slice):
            count, key, state = carry
            chm = slice

            check = count == chm.get_index()

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
