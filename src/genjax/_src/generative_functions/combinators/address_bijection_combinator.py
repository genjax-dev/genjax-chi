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
    ChangeTargetUpdateSpec,
    ChoiceMap,
    Constraint,
    GenerativeFunction,
    GenerativeFunctionClosure,
    Retdiff,
    Sample,
    Trace,
    UpdateSpec,
    Weight,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    Optional,
    PRNGKey,
    Tuple,
    typecheck,
)


@Pytree.dataclass
class AddressBijectionTrace(Trace):
    gen_fn: "AddressBijectionCombinator"
    inner: Trace

    def get_gen_fn(self) -> GenerativeFunction:
        return self.gen_fn

    def get_retval(self) -> Any:
        return self.inner.get_retval()

    def get_sample(self) -> Sample:
        sample: ChoiceMap = self.inner.get_sample()
        return sample.addr_fn(self.gen_fn.address_bijection)

    def get_score(self):
        return self.inner.get_score()


@Pytree.dataclass
class AddressBijectionCombinator(GenerativeFunction):
    args: Tuple
    gen_fn: GenerativeFunctionClosure
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

    def simulate(
        self,
        key: PRNGKey,
    ) -> Trace:
        inner = self.gen_fn(*self.args)
        tr = inner.simulate(key)
        return AddressBijectionTrace(self, tr)

    def importance(
        self,
        key: PRNGKey,
        constraint: Constraint,
    ) -> Tuple[Trace, Weight, UpdateSpec]:
        match constraint:
            case ChoiceMap():
                inner = self.gen_fn(*self.args)
                inner_constraint = constraint.addr_fn(self.get_inverse())
                tr, w, inner_bwd_spec = inner.importance(key, inner_constraint)
                assert isinstance(inner_bwd_spec, ChoiceMap)
                bwd_spec = inner_bwd_spec.addr_fn(self.address_bijection)
                return AddressBijectionTrace(self, tr), w, bwd_spec
            case _:
                raise ValueError(f"Not handled constraint: {constraint}")

    def update_choice_map(
        self,
        key: PRNGKey,
        trace: AddressBijectionTrace,
        chm: ChoiceMap,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateSpec]:
        inner = self.gen_fn(*self.args)
        inner_spec = chm.addr_fn(self.get_inverse())
        tr, w, retdiff, inner_bwd_spec = inner.update(key, trace.inner, inner_spec)
        assert isinstance(inner_bwd_spec, ChoiceMap)
        bwd_spec = inner_bwd_spec.addr_fn(self.address_bijection)
        return tr, w, retdiff, bwd_spec

    def update_change_target(
        self,
        key: PRNGKey,
        trace: Trace,
        argdiffs: Tuple,
        subspec: UpdateSpec,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateSpec]:
        old_gen_fn = trace.get_gen_fn()
        inner_spec = self.inverse(subspec)
        tr, w, retdiff, spec = old_gen_fn.update(
            key, trace.inner, ChangeTargetUpdateSpec(argdiffs, inner_spec)
        )
        return tr, w, retdiff, self.forward(spec)

    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        update_spec: UpdateSpec,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateSpec]:
        match update_spec:
            case ChangeTargetUpdateSpec(argdiffs, spec):
                return self.update_change_target(key, trace, argdiffs, spec)

            case ChoiceMap():
                return self.update_choice_map(key, trace, update_spec)

            case _:
                raise ValueError(f"Unrecognized update spec: {update_spec}")


@typecheck
def address_bijection_combinator(
    gen_fn_closure: Optional[GenerativeFunctionClosure] = None,
    /,
    *,
    address_bijection: dict,
) -> Callable | GenerativeFunctionClosure:
    def decorator(f):
        @GenerativeFunction.closure
        def inner(*args):
            return AddressBijectionCombinator(args, f, address_bijection)

        return inner

    if gen_fn_closure:
        return decorator(gen_fn_closure)
    else:
        return decorator
