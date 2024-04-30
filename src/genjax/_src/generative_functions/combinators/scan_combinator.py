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
"""This module implements a generative function combinator which allows statically
unrolled control flow for generative functions which can act as kernels (a kernel
generative function can accept their previous output as input)."""

import jax
import jax.numpy as jnp

from genjax._src.core.generative import (
    ChoiceMap,
    GenerativeFunction,
    Selection,
    Trace,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    FloatArray,
    Int,
    PRNGKey,
    Tuple,
    typecheck,
)


@Pytree.dataclass
class ScanTrace(Trace):
    scan_gen_fn: "ScanCombinator"
    inner: Trace
    args: Tuple
    retval: Any
    score: FloatArray

    def get_args(self):
        return self.args

    def get_choices(self):
        return jax.vmap(
            lambda idx, submap: ChoiceMap.a(idx, submap),
        )(jnp.arange(self.scan_gen_fn.max_length), self.inner.get_choices())

    def get_gen_fn(self):
        return self.scan_gen_fn

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def project(
        self,
        key: PRNGKey,
        selection: Selection,
    ) -> FloatArray:
        length_checks = (
            jnp.arange(0, len(self.inner.get_score())) < self.dynamic_length + 1
        )

        def idx_check(idx, length_check, inner_slice):
            remaining = selection.step(idx)
            sub_key = jax.random.fold_in(key, idx)
            inner_weight = inner_slice.project(sub_key, remaining)
            return length_check * inner_weight

        idxs = jnp.arange(0, len(self.inner.get_score()))
        ws = jax.vmap(idx_check)(idxs, length_checks, self.inner)
        return jnp.sum(ws, axis=0)


###################
# Scan combinator #
###################


@Pytree.dataclass
class ScanCombinator(GenerativeFunction):
    """> `ScanCombinator` accepts a kernel generative function, as well as a static
    maximum unroll length, and provides a scan-like pattern of generative computation.

    !!! info "Kernel generative functions"
        A kernel generative function is one which accepts and returns the same signature of arguments. Under the hood, `ScanCombinator` is implemented using `jax.lax.scan` - which has the same requirements.

    Examples:
        ```python exec="yes" source="tabbed-left"
        import jax
        import genjax
        console = genjax.console()

        # A kernel generative function.
        @genjax.static_gen_fn
        def random_walk(prev):
            x = genjax.normal(prev, 1.0) @ "x"
            return x

        # You can apply the Scan combinator direclty like this:

        scan_gen_fned_random_walk = genjax.Scan(max_length=1000)(random_walk)

        # But the recommended way to do this is to use `Scan` as a decorator
        # when declaring the function:

        @genjax.scan_combinator(max_length=1000)
        @genjax.static_gen_fn
        def random_walk(prev):
            x = genjax.normal(prev, 1.0) @ "x"
            return x

        init = 0.5
        key = jax.random.PRNGKey(314159)
        tr = jax.jit(genjax.simulate(ramdom_walk)(key, (999, init))

        print(console.render(tr))
        ```
    """

    kernel: GenerativeFunction
    max_length: Int = Pytree.static()

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

    @typecheck
    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> ScanTrace:
        initial_value = args[0]
        scanned_over = args[1]

        def _inner_simulate(key, carried_value, scanned_over):
            tr = self.kernel.simulate(key, (carried_value, scanned_over))
            (carry, scanned_out) = tr.get_retval()
            score = tr.get_score()
            return (carry, score), (tr, scanned_out)

        def _inner(carry, scanned_over):
            key, count, carried_value = carry
            key = jax.random.fold_in(key, count)
            (carry, score), (tr, scanned_out) = _inner_simulate(
                key, carried_value, scanned_over
            )

            return (key, count + 1, carry), (tr, scanned_out, score)

        (_, _, carried_out), (tr, scanned_out, scores) = jax.lax.scan(
            _inner,
            (key, 0, initial_value),
            scanned_over,
            length=self.max_length,
        )

        return ScanTrace(self, tr, args, (carried_out, scanned_out), jnp.sum(scores))

    def importance(
        self,
        key: PRNGKey,
        choice: ChoiceMap,
        args: Tuple,
    ) -> Tuple[ScanTrace, FloatArray]:
        length = args[0]
        state = args[1]
        static_args = args[2:]

        def _inner(carry, _):
            count, key, state = carry

            def _with_choice(key, count, state):
                sub_choice_map = choice.get_submap(count)
                key, sub_key = jax.random.split(key)
                (tr, w) = self.kernel.importance(
                    sub_key, sub_choice_map, (state, *static_args)
                )
                return key, count + 1, tr, tr.get_retval(), tr.get_score(), w

            def _with_empty_choice(key, count, state):
                sub_choice_map = ChoiceMap.n
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

        scan_gen_fn_tr = ScanTrace(
            self,
            tr,
            length,
            args,
            retval,
            jnp.sum(score),
        )

        w = jnp.sum(w)
        return (scan_gen_fn_tr, w)

    @typecheck
    def update(
        self,
        key: PRNGKey,
        prev: ScanTrace,
        choice: ChoiceMap,
        argdiffs: Tuple,
    ) -> Tuple[ScanTrace, FloatArray, Any, ChoiceMap]:
        raise NotImplementedError

    def assess(
        self,
        choice: ChoiceMap,
        args: Tuple,
    ) -> Tuple[FloatArray, Any]:
        length = args[0]
        state = args[1]
        static_args = args[2:]

        def _inner(carry, slice):
            count, state = carry
            choice = slice

            check = count == choice.get_index()

            (score, retval) = self.kernel.assess(choice, (state, *static_args))

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
            choice,
            length=self.max_length,
        )

        score = jnp.sum(score)
        return (score, retval)

    @property
    def __wrapped__(self):
        return self.kernel


#############
# Decorator #
#############


def scan_combinator(*, max_length):
    def _decorator(f):
        return ScanCombinator(f, max_length)

    return _decorator
