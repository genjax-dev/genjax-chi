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

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from genjax._src.core.generative import (
    Argdiffs,
    ChoiceMap,
    Constraint,
    EditRequest,
    ExtendedAddressComponent,
    GenerativeFunction,
    Projection,
    Retdiff,
    Score,
    Trace,
    Weight,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    Generic,
    IntArray,
    PRNGKey,
    TypeVar,
)

Carry = TypeVar("Carry", bound=ChoiceMap)
Y = TypeVar("Y")


@Pytree.dataclass
class MarkovTrace(Generic[Y], Trace[tuple[tuple[Any, ...], Y]]):
    gen_fn: "MarkovCombinator[Y]"
    inner: Trace[Y]
    args: tuple[Any, ...]
    retval: tuple[tuple[Any, ...], Y]
    score: Score
    chm: ChoiceMap
    scan_length: int = Pytree.static()

    @staticmethod
    def build(
        state_space_gen_fn: "MarkovCombinator[Y]",
        inner: Trace[Y],
        args: tuple[Any, ...],
        retval: tuple[tuple[Any, ...], Y],
        score: Score,
        scan_length: int,
    ) -> "MarkovTrace[Y]":
        chm = jax.vmap(lambda subtrace: subtrace.get_choices())(
            inner,
        )

        return MarkovTrace(
            state_space_gen_fn,
            inner,
            args,
            retval,
            score,
            chm,
            scan_length,
        )

    def get_args(self) -> tuple[Any, ...]:
        return self.args

    def get_retval(self) -> tuple[tuple[Any, ...], Y]:
        return self.retval

    def get_choices(self) -> ChoiceMap:
        return self.chm

    def get_sample(self) -> ChoiceMap:
        return self.get_choices()

    def get_gen_fn(self):
        return self.gen_fn

    def get_score(self):
        return self.score


###################
# Scan combinator #
###################


@Pytree.dataclass
class AddressFunction(Pytree):
    addresses: list[ExtendedAddressComponent]

    def __call__(self, sample: ChoiceMap) -> tuple[Any, ...]:
        return tuple([sample[addr] for addr in self.addresses])  # pyright: ignore


@Pytree.dataclass
class MarkovCombinator(
    Generic[Y],
    GenerativeFunction[tuple[tuple[Any, ...], Y]],
):
    kernel_gen_fn: GenerativeFunction[Y]
    projection: AddressFunction

    # Only required for `None` scanned inputs
    length: int | None = Pytree.static()

    def __abstract_call__(self, *args) -> tuple[Any, Y]:
        (carry, scanned_in) = args

        def _inner(carry: tuple[Any, ...], scanned_in: Any):
            key = jax.random.PRNGKey(0)
            tr = self.kernel_gen_fn.simulate(key, (carry, scanned_in))
            scanned_out = tr.get_retval()
            projected_carry = self.projection(tr.get_choices())
            return projected_carry, scanned_out

        v, scanned_out = jax.lax.scan(
            _inner,
            carry,
            scanned_in,
            length=self.length,
        )

        return v, scanned_out

    @staticmethod
    def _static_scan_length(xs: Any, length: int | None) -> int:
        # We start by triggering a scan to force all JAX validations to run.
        jax.lax.scan(lambda c, x: (c, None), None, xs, length=length)
        return length or jtu.tree_leaves(xs)[0].shape[0]

    def simulate(
        self,
        key: PRNGKey,
        args: tuple[Any, ...],
    ) -> MarkovTrace[Y]:
        carry, scanned_in = args

        def _inner_simulate(
            key: PRNGKey,
            carry: tuple[Any, ...],
            scanned_in: Any,
        ) -> tuple[
            tuple[tuple[Any, ...], Score],
            tuple[Trace[Y], Y],
        ]:
            tr = self.kernel_gen_fn.simulate(key, (carry, scanned_in))
            scanned_out = tr.get_retval()
            projected_carry = self.projection(tr.get_choices())
            score = tr.get_score()
            return (projected_carry, score), (tr, scanned_out)

        def _inner(
            carry: tuple[PRNGKey, IntArray, tuple[Any, ...]],
            scanned_over: Any,
        ) -> tuple[
            tuple[PRNGKey, IntArray, tuple[Any, ...]],
            tuple[Trace[Y], Y, Score],
        ]:
            key, count, carried_value = carry
            key = jax.random.fold_in(key, count)
            (carried_out, score), (tr, scanned_out) = _inner_simulate(
                key, carried_value, scanned_over
            )

            return (
                (key, count + 1, carried_out),
                (tr, scanned_out, score),
            )

        (_, _, carried_out), (tr, scanned_out, scores) = jax.lax.scan(
            _inner,
            (key, jnp.asarray(0), carry),
            scanned_in,
            length=self.length,
        )

        return MarkovTrace.build(
            self,
            tr,
            args,
            (carried_out, scanned_out),
            jnp.sum(scores),
            self._static_scan_length(scanned_in, self.length),
        )

    def assess(
        self,
        sample: ChoiceMap,
        args: tuple[Any, ...],
    ) -> tuple[Score, Any]:
        (carry, scanned_in) = args

        def compute_retvals(sample_slice):
            return self.projection(sample_slice)

        retvals = jax.vmap(compute_retvals)(
            sample,
        )

        def _inflate(v):
            arr = jnp.array(v, copy=False)
            return jnp.expand_dims(arr, axis=0) if not arr.shape else arr

        carried_args = jtu.tree_map(
            lambda v1, v2: jnp.concat([_inflate(v1), v2[0:-1]]),
            carry,
            retvals,
        )

        def _assess(
            idx: IntArray,
            subsample: ChoiceMap,
            carry: tuple[Any, ...],
            scanned_in: Y,
        ):
            score, retval = self.kernel_gen_fn.assess(subsample, (carry, scanned_in))
            scanned_out = retval
            return score, scanned_out

        scores, scanned_out = jax.vmap(_assess)(
            jnp.arange(self._static_scan_length(scanned_in, self.length)),
            sample,
            carried_args,
            scanned_in,
        )

        return (
            jnp.sum(scores),
            (jtu.tree_map(lambda v: v[-1], retvals), scanned_out),
        )

    ##########################################################
    # Currently, this one just supports simulate and assess. #
    ##########################################################

    def generate(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: tuple[Any, ...],
    ) -> tuple[MarkovTrace[Y], Weight]:
        raise NotImplementedError

    def project(
        self,
        key: PRNGKey,
        trace: Trace[tuple[tuple[Any, ...], Y]],
        projection: Projection[Any],
    ) -> Weight:
        raise NotImplementedError

    def edit(
        self,
        key: PRNGKey,
        trace: Trace[tuple[tuple[Any, ...], Y]],
        edit_request: EditRequest,
        argdiffs: Argdiffs,
    ) -> tuple[
        Trace[tuple[tuple[Any, ...], Y]],
        Weight,
        Retdiff[tuple[tuple[Any, ...], Y]],
        EditRequest,
    ]:
        raise NotImplementedError


##############
# Decorators #
##############


def markov(
    n: int,
    addr_fn: AddressFunction,
) -> Callable[
    [GenerativeFunction[Y]],
    GenerativeFunction[tuple[tuple[Any, ...], Y]],
]:
    def decorator(f: GenerativeFunction[Y]):
        return MarkovCombinator[Y](f, addr_fn, length=n)

    return decorator
