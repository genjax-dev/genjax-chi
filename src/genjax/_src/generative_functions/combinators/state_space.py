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
    IndexTangent,
    Projection,
    Retdiff,
    Sample,
    Score,
    Trace,
    Tracediff,
    TraceTangent,
    UnitTangent,
    Weight,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    FloatArray,
    Generic,
    Int,
    IntArray,
    PRNGKey,
    TypeVar,
)

Carry = TypeVar("Carry", bound=Sample)
Y = TypeVar("Y")


@Pytree.dataclass(match_args=True)
class StateSpaceTraceTangent(Generic[Carry, Y], TraceTangent):
    subtangents: TraceTangent
    new_args: tuple[Any, ...]
    new_retval: tuple[Carry, Y]
    scan_length: int = Pytree.static()

    def __mul__(self, other: TraceTangent) -> TraceTangent:
        raise NotImplementedError

    def get_delta_score(self) -> Score:
        return jnp.sum(
            jax.vmap(lambda tangent: tangent.get_delta_score())(self.subtangents)
        )


@Pytree.dataclass
class StateSpaceTrace(Generic[Y], Trace[tuple[tuple[Any, ...], Y]]):
    scan_gen_fn: "StateSpaceCombinator[Y]"
    inner: Trace[Y]
    args: tuple[Any, ...]
    retval: tuple[tuple[Any, ...], Y]
    score: Score
    chm: ChoiceMap
    scan_length: int = Pytree.static()

    @staticmethod
    def build(
        state_space_gen_fn: "StateSpaceCombinator[Y]",
        inner: Trace[Y],
        args: tuple[Any, ...],
        retval: tuple[tuple[Any, ...], Y],
        score: FloatArray,
        scan_length: int,
    ) -> "StateSpaceTrace[Y]":
        chm = jax.vmap(lambda subtrace: subtrace.get_choices())(
            inner,
        )
        chm = chm.vec()

        return StateSpaceTrace(
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
        return self.scan_gen_fn

    def get_score(self):
        return self.score

    def pull(self, pull_request: TraceTangent) -> "StateSpaceTrace[Y]":
        match pull_request:
            case StateSpaceTraceTangent(subtrace_diffs, args, retval, scan_length):
                new_subtrace: Trace[Y] = jax.vmap(lambda tr, diff: tr.pull(diff))(
                    self.inner, subtrace_diffs
                )
                scores = jax.vmap(lambda tr: tr.get_score())(new_subtrace)
                return StateSpaceTrace.build(
                    self.get_gen_fn(),
                    new_subtrace,
                    args,
                    retval,
                    jnp.sum(scores),
                    scan_length,
                )
            case IndexTangent(idxs, subtrace_tangents):
                new_subtrace = jax.vmap(
                    lambda idx, diff: jtu.tree_map(lambda v: v[idx], self.inner).pull(
                        diff
                    )
                )(idxs, subtrace_tangents)
                new_inner = jtu.tree_map(
                    lambda v1, v2: v1.at[idxs].set(v2), self.inner, new_subtrace
                )
                return StateSpaceTrace.build(
                    self.scan_gen_fn,
                    new_inner,
                    self.args,
                    new_inner.get_retval(),
                    jnp.sum(new_inner.get_score()),
                    self.scan_length,
                )
            case _:
                raise NotImplementedError


###################
# Scan combinator #
###################


class AddressFunction(Pytree):
    addresses: list[ExtendedAddressComponent]

    def __call__(self, sample: ChoiceMap) -> tuple[Any, ...]:
        return tuple([sample[addr] for addr in self.addresses])


@Pytree.dataclass
class StateSpaceCombinator(
    Generic[Y],
    GenerativeFunction[tuple[tuple[Any, ...], Y]],
):
    kernel_gen_fn: GenerativeFunction[Y]
    projection: AddressFunction

    # Only required for `None` carry inputs
    length: Int | None = Pytree.static()

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
    ) -> StateSpaceTrace[Y]:
        carry, scanned_in = args

        def _inner_simulate(
            key: PRNGKey,
            carry: tuple[Any, ...],
            scanned_in: Any,
        ) -> tuple[tuple[tuple[Any, ...], Score], tuple[Trace[Y], Y]]:
            tr = self.kernel_gen_fn.simulate(key, (carry, scanned_in))
            scanned_out = tr.get_retval()
            projected_carry = self.projection(tr.get_choices())
            score = tr.get_score()
            return (projected_carry, score), (tr, scanned_out)

        def _inner(
            carry: tuple[PRNGKey, IntArray, tuple[Any, ...]], scanned_over: Any
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

        return StateSpaceTrace.build(
            self,
            tr,
            args,
            (carried_out, scanned_out),
            jnp.sum(scores),
            self._static_scan_length(scanned_in, self.length),
        )

    def generate(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: tuple[Any, ...],
    ) -> tuple[StateSpaceTrace[Y], Weight]:
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
        tracediff: Tracediff[StateSpaceTrace[Y], UnitTangent],
        edit_request: EditRequest,
        argdiffs: Argdiffs,
    ) -> tuple[TraceTangent, Weight, Retdiff[tuple[tuple[Any, ...], Y]], EditRequest]:
        raise NotImplementedError

    def assess(
        self,
        sample: ChoiceMap,
        args: tuple[Any, ...],
    ) -> tuple[Score, Any]:
        (carry, scanned_in) = args

        def compute_retvals(idx, sample_slice):
            subsample = sample_slice(idx)
            return self.projection(subsample)

        retvals = jax.vmap(compute_retvals)(
            jnp.arange(self._static_scan_length(args, None)),
            sample,
        )

        carried_args = jtu.tree_map(
            lambda v1, v2: jnp.concatenate([v1, v2[0:-1]]),
            carry,
            retvals,
        )

        def _assess(
            idx: IntArray,
            sample: ChoiceMap,
            carry: tuple[Any, ...],
            scanned_in: Y,
        ):
            subsample = sample(idx)
            score, retval = self.kernel_gen_fn.assess(subsample, (carry, scanned_in))
            scanned_out = retval
            return score, scanned_out

        scores, scanned_out = jax.vmap(_assess)(
            jnp.arange(self._static_scan_length(args, None)),
            sample,
            carried_args,
            scanned_in,
        )

        return (
            jnp.sum(scores),
            (jtu.tree_map(lambda v: v[-1], retvals), scanned_out),
        )


##############
# Decorators #
##############


def state_space(
    addr_fn: AddressFunction,
    *,
    n: Int | None = None,
) -> Callable[
    [GenerativeFunction[Y]],
    GenerativeFunction[tuple[tuple[Any, ...], Y]],
]:
    def decorator(f: GenerativeFunction[Y]):
        return StateSpaceCombinator[Y](f, addr_fn, length=n)

    return decorator
