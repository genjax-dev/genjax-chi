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


from genjax._src.core.generative import (
    ChoiceMap,
    EmptyTrace,
    GenerativeFunction,
    ImportanceProblem,
    Retdiff,
    Sample,
    Score,
    Trace,
    UpdateProblem,
    Weight,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.traceback_util import register_exclusion
from genjax._src.core.typing import (
    Any,
    Callable,
    PRNGKey,
    Tuple,
    typecheck,
)

register_exclusion(__file__)


@Pytree.dataclass
class AddressBijectionTrace(Trace):
    gen_fn: "AddressBijectionCombinator"
    inner: Trace

    def get_args(self) -> Tuple:
        return self.inner.get_args()

    def get_retval(self) -> Any:
        return self.inner.get_retval()

    def get_gen_fn(self) -> GenerativeFunction:
        return self.gen_fn

    def get_sample(self) -> Sample:
        sample: ChoiceMap = self.inner.get_sample()
        return sample.addr_fn(self.gen_fn.address_bijection)

    def get_score(self):
        return self.inner.get_score()


@Pytree.dataclass
class AddressBijectionCombinator(GenerativeFunction):
    gen_fn: GenerativeFunction
    address_bijection: dict = Pytree.static(default_factory=dict)

    def get_inverse(self) -> dict:
        inverse_map = {v: k for (k, v) in self.address_bijection.items()}
        return inverse_map

    def static_check_bijection(self):
        inverse_map = self.get_inverse()
        for k, v in self.address_bijection.items():
            assert inverse_map[v] == k

    def __post_init__(self):
        self.static_check_bijection()

    ##################################
    # Generative function interfaces #
    ##################################

    @GenerativeFunction.gfi_boundary
    @typecheck
    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> Trace:
        tr = self.gen_fn.simulate(key, args)
        return AddressBijectionTrace(self, tr)

    @typecheck
    def update_importance(
        self,
        key: PRNGKey,
        chm: ChoiceMap,
        argdiffs: Tuple,
    ):
        inner_problem = chm.addr_fn(self.get_inverse())
        tr, w, retdiff, inner_bwd_problem = self.gen_fn.update(
            key,
            EmptyTrace(self.gen_fn),
            ImportanceProblem(inner_problem),
            argdiffs,
        )
        assert isinstance(inner_bwd_problem, ChoiceMap)
        bwd_problem = inner_bwd_problem.addr_fn(self.address_bijection)
        return AddressBijectionTrace(self, tr), w, retdiff, bwd_problem

    def update_choice_map(
        self,
        key: PRNGKey,
        trace: AddressBijectionTrace,
        chm: ChoiceMap,
        argdiffs: Tuple,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        inner_problem = chm.addr_fn(self.get_inverse())
        tr, w, retdiff, inner_bwd_problem = self.gen_fn.update(
            key, trace.inner, inner_problem, argdiffs
        )
        assert isinstance(inner_bwd_problem, ChoiceMap)
        bwd_problem = inner_bwd_problem.addr_fn(self.address_bijection)
        return AddressBijectionTrace(self, tr), w, retdiff, bwd_problem

    @GenerativeFunction.gfi_boundary
    @typecheck
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        update_problem: UpdateProblem,
        argdiffs: Tuple,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        match update_problem:
            case ChoiceMap():
                return self.update_choice_map(key, trace, update_problem, argdiffs)

            case ImportanceProblem(constraint):
                return self.update_importance(key, constraint, argdiffs)

            case _:
                raise ValueError(f"Unrecognized update problem: {update_problem}")

    @GenerativeFunction.gfi_boundary
    @typecheck
    def assess(
        self,
        sample: Sample,
        args: Tuple,
    ) -> Tuple[Score, Any]:
        match sample:
            case ChoiceMap():
                inner_sample = sample.addr_fn(self.get_inverse())
                return self.gen_fn.assess(inner_sample, args)
            case _:
                raise ValueError(f"Not handled sample: {sample}")


@typecheck
def address_bijection_combinator(
    gen_fn: GenerativeFunction | None = None,
    /,
    *,
    address_bijection: dict,
) -> Callable | AddressBijectionCombinator:
    def decorator(f):
        return AddressBijectionCombinator(f, address_bijection)

    if gen_fn:
        return decorator(gen_fn)
    else:
        return decorator
