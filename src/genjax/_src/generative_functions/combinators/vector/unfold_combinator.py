# Copyright 2023 MIT Probabilistic Computing Project
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
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.experimental import checkify

from genjax._src.core.datatypes.generative import Choice
from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import EmptyChoice
from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.datatypes.generative import HierarchicalSelection
from genjax._src.core.datatypes.generative import JAXGenerativeFunction
from genjax._src.core.datatypes.generative import LanguageConstructor
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.datatypes.generative import mask
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.interpreters.incremental import static_check_no_change
from genjax._src.core.interpreters.incremental import tree_diff_no_change
from genjax._src.core.interpreters.incremental import tree_diff_primal
from genjax._src.core.interpreters.incremental import tree_diff_unknown_change
from genjax._src.core.typing import Any
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import Int
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.combinators.staging_utils import make_zero_trace
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    IndexedChoiceMap,
)
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    IndexedSelection,
)
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    VectorChoiceMap,
)
from genjax._src.generative_functions.static.static_gen_fn import SupportsCalleeSugar
from genjax._src.global_options import global_options


@dataclass
class UnfoldTrace(Trace):
    unfold: GenerativeFunction
    inner: Trace
    dynamic_length: IntArray
    args: Tuple
    retval: Any
    score: FloatArray

    def flatten(self):
        return (
            self.unfold,
            self.inner,
            self.dynamic_length,
            self.args,
            self.retval,
            self.score,
        ), ()

    def get_args(self):
        return self.args

    def get_choices(self):
        mask_flags = (
            (
                jnp.expand_dims(jnp.arange(self.unfold.max_length), -1)
                <= self.dynamic_length
            ).T
            if jnp.array(self.dynamic_length, copy=False).shape
            else jnp.arange(self.unfold.max_length) <= self.dynamic_length
        )
        return VectorChoiceMap.new(mask(mask_flags, self.inner.strip()))

    def get_gen_fn(self):
        return self.unfold

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    @dispatch
    def project(
        self,
        selection: IndexedSelection,
    ) -> FloatArray:
        inner_project = self.inner.project(selection.inner)
        return jnp.sum(
            jnp.where(
                selection.indices < self.dynamic_length + 1,
                jnp.take(inner_project, selection.indices, mode="fill", fill_value=0.0),
                0.0,
            )
        )

    @dispatch
    def project(
        self,
        selection: HierarchicalSelection,
    ) -> FloatArray:
        return jnp.sum(
            jnp.where(
                jnp.arange(0, len(self.inner.get_score())) < self.dynamic_length + 1,
                self.inner.project(selection),
                0.0,
            )
        )


#####
# Unfold combinator
#####


@dataclass
class UnfoldCombinator(JAXGenerativeFunction, SupportsCalleeSugar):
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
        tr = jax.jit(genjax.simulate(unfold))(key, (999, init))

        print(console.render(tr))
        ```
    """

    max_length: IntArray
    kernel: JAXGenerativeFunction

    def flatten(self):
        return (self.kernel,), (self.max_length,)

    @typecheck
    @classmethod
    def new(cls, kernel: JAXGenerativeFunction, max_length: Int) -> "UnfoldCombinator":
        """The preferred constructor for `UnfoldCombinator` generative function
        instances. The shorthand symbol is `Unfold = UnfoldCombinator.new`.

        Arguments:
            kernel: A kernel `JAXGenerativeFunction` instance.
            max_length: A static maximum possible unroll length.

        Returns:
            instance: An `UnfoldCombinator` instance.
        """
        return UnfoldCombinator(max_length, kernel)

    # To get the type of return value, just invoke
    # the scanned over source (with abstract tracer arguments).
    def __abstract_call__(self, *args) -> Any:
        state = args[1]
        static_args = args[2:]

        def _inner(carry, xs):
            state = carry
            v = self.kernel.__abstract_call__(state, *static_args)
            return v, v

        _, stacked = jax.lax.scan(_inner, state, None, length=self.max_length)

        return stacked

    def _optional_out_of_bounds_check(self, count: IntArray):
        def _check():
            check_flag = jnp.less(self.max_length, count + 1)
            checkify.check(
                check_flag,
                f"\nUnfoldCombinator received a length argument ({count}) longer than specified max length ({self.max_length})",
            )

        global_options.optional_check(_check)

    @typecheck
    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> UnfoldTrace:
        length = args[0]
        self._optional_out_of_bounds_check(length)
        state = args[1]
        static_args = args[2:]

        zero_trace = make_zero_trace(
            self.kernel,
            key,
            (state, *static_args),
        )

        def _inner_simulate(key, state, static_args, count):
            key, sub_key = jax.random.split(key)
            tr = self.kernel.simulate(sub_key, (state, *static_args))
            state = tr.get_retval()
            score = tr.get_score()
            return (tr, state, count, count + 1, score)

        def _inner_zero_fallback(key, state, _, count):
            state = state
            score = 0.0
            return (zero_trace, state, -1, count, score)

        def _inner(carry, _):
            count, key, state = carry
            check = jnp.less(count, length + 1)
            key, sub_key = jax.random.split(key)
            tr, state, index, count, score = jax.lax.cond(
                check,
                _inner_simulate,
                _inner_zero_fallback,
                sub_key,
                state,
                static_args,
                count,
            )

            return (count, key, state), (tr, index, state, score)

        (_, _, state), (tr, _, retval, scores) = jax.lax.scan(
            _inner,
            (0, key, state),
            None,
            length=self.max_length,
        )

        unfold_tr = UnfoldTrace(
            self,
            tr,
            length,
            args,
            retval,
            jnp.sum(scores),
        )

        return unfold_tr

    @dispatch
    def importance(
        self,
        key: PRNGKey,
        chm: IndexedChoiceMap,
        args: Tuple,
    ) -> Tuple[UnfoldTrace, FloatArray]:
        length = args[0]
        self._optional_out_of_bounds_check(length)
        state = args[1]
        static_args = args[2:]

        def _inner(carry, _):
            count, key, state = carry

            def _with_choice(key, count, state):
                sub_choice_map = chm.get_submap(count)
                key, sub_key = jax.random.split(key)
                (tr, w) = self.kernel.importance(
                    sub_key, sub_choice_map, (state, *static_args)
                )
                return key, count + 1, tr, tr.get_retval(), tr.get_score(), w

            def _with_empty_choice(key, count, state):
                sub_choice_map = EmptyChoice()
                key, sub_key = jax.random.split(key)
                (tr, w) = self.kernel.importance(
                    sub_key, sub_choice_map, (state, *static_args)
                )
                return key, count, tr, state, 0.0, 0.0

            check = jnp.less(count, length + 1)
            key, count, tr, state, score, w = jax.lax.cond(
                check,
                _with_choice,
                _with_empty_choice,
                key,
                count,
                state,
            )

            return (count, key, state), (w, score, tr, state)

        (_, _, state), (w, score, tr, retval) = jax.lax.scan(
            _inner,
            (0, key, state),
            None,
            length=self.max_length,
        )

        unfold_tr = UnfoldTrace(
            self,
            tr,
            length,
            args,
            retval,
            jnp.sum(score),
        )

        w = jnp.sum(w)
        return (unfold_tr, w)

    @dispatch
    def importance(
        self,
        key: PRNGKey,
        chm: VectorChoiceMap,
        args: Tuple,
    ) -> Tuple[UnfoldTrace, FloatArray]:
        length = args[0]
        self._optional_out_of_bounds_check(length)
        state = args[1]
        static_args = args[2:]

        def _inner(carry, slice):
            count, key, state = carry
            chm = slice
            key, sub_key = jax.random.split(key)

            def _importance(key, chm, state):
                return self.kernel.importance(key, chm, (state, *static_args))

            def _simulate(key, chm, state):
                tr = self.kernel.simulate(key, (state, *static_args))
                return (tr, 0.0)

            check_count = jnp.less(count, length + 1)
            (tr, w) = jax.lax.cond(
                check_count,
                _importance,
                _simulate,
                sub_key,
                chm.inner,
                state,
            )

            count, state, score, w = jax.lax.cond(
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

        (_, _, state), (w, score, tr, retval) = jax.lax.scan(
            _inner,
            (0, key, state),
            chm,
            length=self.max_length,
        )

        unfold_tr = UnfoldTrace(
            self,
            tr,
            length,
            args,
            retval,
            jnp.sum(score),
        )

        w = jnp.sum(w)
        return (unfold_tr, w)

    @dispatch
    def importance(
        self,
        key: PRNGKey,
        _: EmptyChoice,
        args: Tuple,
    ) -> Tuple[UnfoldTrace, FloatArray]:
        length = args[0]
        self._optional_out_of_bounds_check(length)
        unfold_tr = self.simulate(key, args)
        w = 0.0
        return (unfold_tr, w)

    @dispatch
    def importance(
        self,
        key: PRNGKey,
        chm: ChoiceMap,
        args: Tuple,
    ):
        raise NotImplementedError

    @dispatch
    def _update_fallback(
        self,
        key: PRNGKey,
        prev: UnfoldTrace,
        chm: ChoiceMap,
        length: Diff,
        state: Diff,
        *static_args: Diff,
    ):
        maybe_idx_chm = IndexedChoiceMap.convert(chm)
        return self._update_fallback(
            key, prev, maybe_idx_chm, length, state, *static_args
        )

    @dispatch
    def _update_fallback(
        self,
        key: PRNGKey,
        prev: UnfoldTrace,
        chm: VectorChoiceMap,
        length: Diff,
        state: Diff,
        *static_args: Diff,
    ):
        length, state, static_args = tree_diff_primal((length, state, static_args))

        def _inner(carry, slice):
            count, key, state = carry
            (prev, chm) = slice
            key, sub_key = jax.random.split(key)

            (tr, w, retval_diff, discard) = self.kernel.update(
                sub_key, prev, chm, (state, *static_args)
            )

            check = jnp.less(count, length + 1)
            count, state, score, weight = jax.lax.cond(
                check,
                lambda *args: (count + 1, retval_diff, tr.get_score(), w),
                lambda *args: (count, state, 0.0, 0.0),
            )
            return (count, key, state), (state, score, weight, tr, discard)

        (_, _, state), (retval_diff, score, w, tr, discard) = jax.lax.scan(
            _inner,
            (0, key, state),
            (prev, chm),
            length=self.max_length,
        )

        unfold_tr = UnfoldTrace(
            self,
            tr,
            length,
            (length, state, *static_args),
            tree_diff_primal(retval_diff),
            jnp.sum(score),
        )

        w = jnp.sum(w)
        return (unfold_tr, w, retval_diff, discard)

    @dispatch
    def _update_specialized(
        self,
        key: PRNGKey,
        prev: UnfoldTrace,
        chm: EmptyChoice,
        length: Diff,
        state: Any,
        *static_args: Any,
    ):
        length, state, static_args = tree_diff_primal((length, state, static_args))

        def _inner(carry, slice):
            count, key, state = carry
            (prev,) = slice
            key, sub_key = jax.random.split(key)

            (tr, w, retval_diff, discard) = self.kernel.update(
                sub_key,
                prev,
                chm,
                tree_diff_no_change((state, *static_args)),
            )

            check = jnp.less(count, length + 1)
            count, state, score, weight = jax.lax.cond(
                check,
                lambda *args: (count + 1, retval_diff, tr.get_score(), w),
                lambda *args: (count, state, 0.0, 0.0),
            )
            return (count, key, state), (state, score, weight, tr, discard)

        prev_inner_trace = prev.inner
        (_, _, state), (retval_diff, score, w, tr, discard) = jax.lax.scan(
            _inner,
            (0, key, state),
            (prev_inner_trace,),
            length=self.max_length,
        )

        unfold_tr = UnfoldTrace(
            self,
            tr,
            length,
            (length, state, *static_args),
            tree_diff_primal(retval_diff),
            jnp.sum(score),
        )

        w = jnp.sum(w)
        return (unfold_tr, w, retval_diff, discard)

    # TODO: this does not handle when the new length
    # is less than the previous!
    @dispatch
    def _update_specialized(
        self,
        key: PRNGKey,
        prev: UnfoldTrace,
        chm: IndexedChoiceMap,
        length: Diff,
        state: Any,
        *static_args: Any,
    ):
        start_lower = jnp.min(chm.indices)
        prev_length = prev.get_args()[0]

        # TODO: `UnknownChange` is used here
        # to preserve the Pytree structure across the loop.
        state_diff = tree_diff_unknown_change(
            jax.lax.cond(
                start_lower
                == 0,  # if the starting index is 0, we need to grab the state argument.
                lambda *args: state,
                # Else, we use the retval from the previous iteration in the trace.
                lambda *args: jtu.tree_map(
                    lambda v: v[start_lower - 1],
                    prev.get_retval(),
                ),
            )
        )
        prev_inner_trace = prev.inner

        def _inner(index, state):
            (key, w, state_diff, prev) = state
            sub_chm = chm.get_submap(index)
            prev_slice = jtu.tree_map(lambda v: v[index], prev)
            key, sub_key = jax.random.split(key)

            # Extending to an index greater than the previous length.
            def _importance(key):
                state_primal = tree_diff_primal(state_diff)
                (new_tr, w) = self.kernel.importance(
                    key, sub_chm, (state_primal, *static_args)
                )
                primal_state = new_tr.get_retval()
                retval_diff = tree_diff_unknown_change(primal_state)

                return (retval_diff, w, new_tr)

            # Updating an existing index.
            def _update(key):
                static_argdiffs = tree_diff_no_change(static_args)
                (new_tr, w, retval_diff, _) = self.kernel.update(
                    key, prev_slice, sub_chm, (state_diff, *static_argdiffs)
                )

                # TODO: c.f. message above on `UnknownChange`.
                # Preserve the diff type across the loop
                # iterations.
                primal_state = tree_diff_primal(retval_diff)
                retval_diff = tree_diff_unknown_change(primal_state)
                return (retval_diff, w, new_tr)

            check = prev_length < index
            (state_diff, idx_w, new_tr) = jax.lax.cond(
                check, _importance, _update, sub_key
            )

            def _mutate(prev, new):
                new = prev.at[index].set(new)
                return new

            # TODO: also handle discard.
            prev = jtu.tree_map(_mutate, prev, new_tr)
            w += idx_w

            return (key, w, state_diff, prev)

        # TODO: add discard.
        new_upper = tree_diff_primal(length)
        new_upper = jnp.where(
            new_upper >= self.max_length,
            self.max_length - 1,
            new_upper,
        )
        (_, w, _, new_inner_trace) = jax.lax.fori_loop(
            start_lower,
            new_upper + 1,  # the bound semantics follow Python range semantics.
            _inner,
            (key, 0.0, state_diff, prev_inner_trace),
        )

        # Select the new return values.
        checks = jnp.arange(0, self.max_length) < new_upper + 1

        def _where(v1, v2):
            extension = len(v1.shape) - 1
            new_checks = checks.reshape(checks.shape + (1,) * extension)
            return jnp.where(new_checks, v1, v2)

        retval = jtu.tree_map(
            _where,
            new_inner_trace.get_retval(),
            prev.get_retval(),
        )
        retval_diff = tree_diff_unknown_change(retval)
        args = tree_diff_primal((length, state, *static_args))

        # TODO: is there a faster way to do this with the information I already have?
        new_score = jnp.sum(
            jnp.where(
                jnp.arange(0, len(new_inner_trace.get_score())) < new_upper + 1,
                new_inner_trace.get_score(),
                0.0,
            )
        )

        new_tr = UnfoldTrace(
            self,
            new_inner_trace,
            new_upper,
            args,
            retval,
            new_score,
        )
        return (new_tr, w, retval_diff, EmptyChoice())

    @dispatch
    def _update_specialized(
        self,
        key: PRNGKey,
        prev: UnfoldTrace,
        chm: VectorChoiceMap,
        length: Diff,
        state: Any,
        *static_args: Any,
    ):
        raise NotImplementedError

    @dispatch
    def _update_specialized(
        self,
        key: PRNGKey,
        prev: UnfoldTrace,
        chm: ChoiceMap,
        length: Diff,
        state: Any,
        *static_args: Any,
    ):
        raise NotImplementedError

    @typecheck
    def update(
        self,
        key: PRNGKey,
        prev: UnfoldTrace,
        chm: Choice,
        argdiffs: Tuple,
    ) -> Tuple[UnfoldTrace, FloatArray, Any, Choice]:
        length = argdiffs[0]
        state = argdiffs[1]
        static_args = argdiffs[2:]
        args = tree_diff_primal(argdiffs)
        self._optional_out_of_bounds_check(args[0])  # length
        check_state_static_no_change = static_check_no_change((state, static_args))
        if check_state_static_no_change:
            state = tree_diff_primal(state)
            static_args = tree_diff_primal(static_args)
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
        chm: VectorChoiceMap,
        args: Tuple,
    ) -> Tuple[FloatArray, Any]:
        length = args[0]
        self._optional_out_of_bounds_check(length)
        state = args[1]
        static_args = args[2:]

        def _inner(carry, slice):
            count, state = carry
            chm = slice

            check = count == chm.get_index()

            (score, retval) = self.kernel.assess(chm, (state, *static_args))

            check = jnp.less(count, length + 1)
            index = jax.lax.cond(
                check,
                lambda *args: count,
                lambda *args: -1,
            )
            count, state, score = jax.lax.cond(
                check,
                lambda *args: (count + 1, retval, score),
                lambda *args: (count, state, 0.0),
            )
            return (count, state), (state, score, index)

        (_, state), (retval, score, _) = jax.lax.scan(
            _inner,
            (0, state),
            chm,
            length=self.max_length,
        )

        score = jnp.sum(score)
        return (score, retval)


#########################
# Language constructors #
#########################


@typecheck
def unfold_combinator(gen_fn: JAXGenerativeFunction, max_length: Int):
    return UnfoldCombinator.new(gen_fn, max_length)


Unfold = LanguageConstructor(
    unfold_combinator,
)
