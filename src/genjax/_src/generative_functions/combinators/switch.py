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


import jax.tree_util as jtu

from genjax._src.core.generative import (
    Argdiffs,
    ChoiceMap,
    Constraint,
    EditRequest,
    GenerativeFunction,
    Projection,
    Retdiff,
    Score,
    Trace,
    Tracediff,
    TraceTangent,
    UnitTangent,
    UnitTracediff,
    Weight,
)
from genjax._src.core.generative.generative_function import (
    TraceTangentMonoidActionException,
    TraceTangentMonoidOperationException,
)
from genjax._src.core.interpreters.incremental import Diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    FloatArray,
    Generic,
    IntArray,
    PRNGKey,
    TypeVar,
)

R = TypeVar("R")


################
# Switch trace #
################


@Pytree.dataclass(match_args=True)
class SwitchTraceTangent(Generic[R], TraceTangent):
    branch: IntArray
    args: tuple[Any, ...]
    subtangent: TraceTangent
    retval: R
    delta_score: Score

    def __mul__(self, other: TraceTangent) -> TraceTangent:
        match other:
            case SwitchTraceTangent(branch, args, subtangent, retval, delta_score):
                return SwitchTraceTangent(
                    branch,
                    args,
                    self.subtangent * subtangent,
                    retval,
                    self.delta_score + delta_score,
                )
            case _:
                raise TraceTangentMonoidOperationException(other)

    def get_delta_score(self) -> Score:
        return self.delta_score


@Pytree.dataclass
class SwitchTrace(Generic[R], Trace[R]):
    gen_fn: "SwitchCombinator[R]"
    branch: IntArray
    args: tuple[Any, ...]
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

    def pull(self, pull_request: TraceTangent) -> "SwitchTrace[R]":
        match pull_request:
            case SwitchTraceTangent(branch, args, subtangent, retval, delta_score):
                return SwitchTrace(
                    self.gen_fn,
                    branch,
                    args,
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
class SwitchCombinator(Generic[R], GenerativeFunction[R]):
    gen_fn: GenerativeFunction[R]

    def __abstract_call__(self, *args) -> R:
        idx, branch_args = args[0], args[1:]
        selected_branch_args = jtu.tree_map(lambda v: v[idx], *branch_args)
        return self.gen_fn.__abstract_call__(*selected_branch_args)

    def simulate(
        self,
        key: PRNGKey,
        args: tuple[Any, ...],
    ) -> SwitchTrace[R]:
        idx: IntArray = args[0]
        branch_args: tuple[tuple[Any, ...], ...] = args[1:]
        selected_branch_args = jtu.tree_map(lambda v: v[idx], *branch_args)
        subtrace = self.gen_fn.simulate(key, selected_branch_args)
        return SwitchTrace(
            self,
            idx,
            branch_args,
            subtrace,
            subtrace.get_retval(),
            subtrace.get_score(),
        )

    def assess(
        self,
        sample: ChoiceMap,
        args: tuple[Any, ...],
    ) -> tuple[Score, R]:
        idx, branch_args = args[0], args[1:]
        selected_branch_args = jtu.tree_map(lambda v: v[idx], *branch_args)
        score, retval = self.gen_fn.assess(sample, selected_branch_args)
        return score, retval

    def generate(
        self,
        key: PRNGKey,
        constraint: Constraint,
        args: tuple[Any, ...],
    ) -> tuple[SwitchTrace[R], Weight]:
        idx, branch_args = args[0], args[1:]
        selected_branch_args = jtu.tree_map(lambda v: v[idx], *branch_args)
        subtrace, weight = self.gen_fn.generate(key, constraint, selected_branch_args)
        return SwitchTrace(
            self,
            idx,
            branch_args,
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
        tracediff: Tracediff[R, UnitTangent],
        edit_request: EditRequest,
        argdiffs: Argdiffs,
    ) -> tuple[SwitchTraceTangent[R], Weight, Retdiff[R], EditRequest]:
        idx, branch_argdiffs = argdiffs[0], argdiffs[1:]
        primal: SwitchTrace[R] = tracediff.get_primal()  # pyright: ignore
        selected_branch_argdiffs = jtu.tree_map(lambda v: v[idx], *branch_argdiffs)
        subtangent, weight, retdiff, bwd_request = edit_request.edit(
            key, UnitTracediff(primal), selected_branch_argdiffs
        )
        retval = Diff.tree_primal(retdiff)
        return (
            SwitchTraceTangent(
                idx,
                Diff.tree_primal(selected_branch_argdiffs),
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


def switch(
    gen_fn: GenerativeFunction[R],
) -> SwitchCombinator[R]:
    return SwitchCombinator[R](gen_fn)
