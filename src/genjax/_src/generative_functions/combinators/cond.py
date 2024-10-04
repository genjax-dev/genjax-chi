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
    ChoiceMapConstraint,
    Constraint,
    EditRequest,
    GenerativeFunction,
    IdentityTangent,
    Projection,
    Retdiff,
    Sample,
    Score,
    Trace,
    Tracediff,
    TraceTangent,
    Update,
    Weight,
)
from genjax._src.core.generative.functional_types import staged_choose
from genjax._src.core.generative.generative_function import (
    TraceTangentMonoidActionException,
)
from genjax._src.core.interpreters.incremental import Diff, NoChange, UnknownChange
from genjax._src.core.interpreters.staging import get_data_shape
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    ArrayLike,
    FloatArray,
    Generic,
    Int,
    IntArray,
    PRNGKey,
    ScalarFlag,
    Sequence,
    TypeVar,
)

R = TypeVar("R")


################
# Switch trace #
################


@Pytree.dataclass(match_args=True)
class CondTraceTangent(Generic[R], TraceTangent):
    branch: ScalarFlag
    args: tuple[Any, ...]
    subtangent: TraceTangent
    retval: R
    delta_score: Score

    def get_delta_score(self) -> Score:
        return self.delta_score


@Pytree.dataclass
class CondTrace(Generic[R], Trace[R]):
    gen_fn: "CondCombinator[R]"
    args: tuple[Any, ...]
    branch: ScalarFlag
    subtrace: Trace[R]
    retval: R
    score: FloatArray

    def get_args(self) -> tuple[Any, ...]:
        return self.args

    def get_choices(self) -> ChoiceMap:
        return self.subtrace.get_choices()

    def get_sample(self) -> ChoiceMap:
        return self.get_choices()

    def get_gen_fn(self):
        return self.gen_fn

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def pull(self, pull_request: TraceTangent) -> "CondTrace[R]":
        match pull_request:
            case CondTraceTangent(branch, args, subtangent, retval, delta_score):
                return CondTrace(
                    self.gen_fn,
                    args,
                    branch,
                    self.subtrace.pull(subtangent),
                    retval,
                    self.score + delta_score,
                )
            case _:
                raise TraceTangentMonoidActionException(pull_request)


#####################
# Switch combinator #
#####################


@Pytree.dataclass
class CondCombinator(Generic[R], GenerativeFunction[R]):
    fst_gen_fn: GenerativeFunction[R]
    snd_gen_fn: GenerativeFunction[R]

    def __abstract_call__(self, *args) -> R:
        flag, args = args[0], args[1:]
        return jax.lax.cond(
            flag,
            self.fst_gen_fn.__abstract_call__,
            self.snd_gen_fn.__abstract_call__,
            *args,
        )

    def simulate(
        self,
        key: PRNGKey,
        args: tuple[Any, ...],
    ) -> CondTrace[R]:
        flag: ScalarFlag = args[0]
        branch_args = args[1:]
        subtrace = jax.lax.cond(
            flag, self.fst_gen_fn.simulate, self.snd_gen_fn.simulate, key, branch_args
        )
        return CondTrace(
            self,
            branch_args,
            flag,
            subtrace,
            subtrace.get_retval(),
            subtrace.get_score(),
        )

    def assess(
        self,
        sample: ChoiceMap,
        args: tuple[Any, ...],
    ) -> tuple[Score, R]:
        flag, branch_args = args[0], args[1:]
        score, retval = jax.lax.cond(
            flag,
            self.fst_gen_fn.assess,
            self.snd_gen_fn.assess,
            branch_args,
        )
        return score, retval

    def generate(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: tuple[Any, ...],
    ) -> tuple[CondTrace[R], Weight]:
        flag, branch_args = args[0], args[1:]
        subtrace, weight = jax.lax.cond(
            flag,
            self.fst_gen_fn.generate,
            self.snd_gen_fn.generate,
            key,
            constraint,
            branch_args,
        )
        return CondTrace(
            self,
            branch_args,
            flag,
            subtrace,
            subtrace.get_retval(),
            subtrace.get_score(),
        ), weight

    def project(
        self,
        key: PRNGKey,
        trace: Trace[R],
        projection: Projection[Any],
    ) -> Weight:
        raise NotImplementedError

    def edit(
        self,
        key: PRNGKey,
        tracediff: Tracediff[R, IdentityTangent],
        edit_request: EditRequest,
        argdiffs: Argdiffs,
    ) -> tuple[CondTrace[R], Weight, Retdiff[R], EditRequest]:
        flag, branch_args = args[0], args[1:]
        subtangent, weight, retdiff, bwd_request = jax.lax.cond(
            flag,
            self.fst_gen_fn.edit,
            self.snd_gen_fn.edit,
            key,
            tracediff,
            edit_request,
            branch_args,
        )
        retval = Diff.tree_primal(retdiff)
        return (
            CondTraceTangent(
                flag,
                branch_args,
                subtangent,
                retval,
                subtangent.get_delta_score(),
            ),
            weight,
            retdiff,
            bwd_request,
        )


#############
# Decorator #
#############


def cond(
    fst_gen_fn: GenerativeFunction[R],
    snd_gen_fn: GenerativeFunction[R],
) -> CondCombinator[R]:
    return CondCombinator[R](fst_gen_fn, snd_gen_fn)
