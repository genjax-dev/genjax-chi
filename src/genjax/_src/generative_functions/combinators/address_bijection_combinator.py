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
    Address,
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
        forward = self.gen_fn.forward
        return ChoiceMap.address_fn(forward, sample)

    def get_score(self):
        return self.inner.get_score()


@Pytree.dataclass
class AddressBijectionCombinator(GenerativeFunction):
    args: Tuple
    gen_fn: GenerativeFunctionClosure
    forward: Callable[[Address], Address] = Pytree.static()
    inverse: Callable[[Address], Address] = Pytree.static()

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
    ) -> Tuple[Trace, Weight]:
        match constraint:
            case ChoiceMap():
                inner = self.gen_fn(*self.args)
                inner_constraint = ChoiceMap.address_fn(self.inverse, constraint)
                tr, w = inner.importance(key, inner_constraint)
                return AddressBijectionTrace(self, tr), w
            case _:
                raise ValueError(f"Not handled constraint: {constraint}")

    def update_choice_map(
        self,
        key: PRNGKey,
        trace: Trace,
        chm: ChoiceMap,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateSpec]:
        inner = self.gen_fn(*self.args)
        inner_spec = self.inverse(chm)
        tr, w, retdiff, spec = inner.update(key, trace.inner, inner_spec)
        return tr, w, retdiff, self.forward(spec)

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


def address_bijection_combinator(
    gen_fn_closure: Optional[GenerativeFunctionClosure] = None,
    /,
    *,
    forward: Callable[[Address], Address],
    inverse: Callable[[Address], Address],
) -> Callable | GenerativeFunctionClosure:
    def decorator(f):
        @GenerativeFunction.closure
        def inner(*args):
            return AddressBijectionCombinator(args, f, forward, inverse)

        return inner

    if gen_fn_closure:
        return decorator(gen_fn_closure)
    else:
        return decorator
