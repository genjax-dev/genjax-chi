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

from genjax._src.core.generative.core import (
    Constraint,
    GenerativeFunction,
    Retdiff,
    Sample,
    Trace,
    UpdateSpec,
    Weight,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import Callable, FloatArray, PRNGKey, Tuple, typecheck
from genjax._src.generative_functions.static.static_gen_fn import (
    StaticGenerativeFunction,
    static_gen_fn,
)
from genjax._src.inference.core.sp import Target


@Pytree.dataclass
class KernelTrace(Trace):
    pass


@Pytree.dataclass
class KernelGenerativeFunction(GenerativeFunction):
    args: Tuple[Sample, Callable]
    source: Callable[
        [Sample, Target],
        StaticGenerativeFunction,
    ]

    def simulate(
        self,
        key: PRNGKey,
    ) -> Trace:
        gen_fn = self.source(self.args[0], self.args[1])
        return gen_fn.simulate(key)

    def importance(
        self,
        key: PRNGKey,
        constraint: Constraint,
    ) -> Tuple[Trace, Weight]:
        pass

    def update(
        self,
        key: PRNGKey,
        trace: KernelTrace,
        update_spec: UpdateSpec,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateSpec]:
        pass

    def smcp3_weight(
        self,
        L: "KernelGenerativeFunction",
        K_aux_score: FloatArray,
        old_latents: Sample,
        new_latents: Sample,
        aux: Sample,
    ) -> FloatArray:
        pass


@typecheck
def kernel_gen_fn(
    source: Callable[
        [Sample, Target],
        Tuple[UpdateSpec, Sample],
    ],
) -> Callable[[Sample, Target], KernelGenerativeFunction]:
    def inner(sample, target):
        static_gen_fn_callable = static_gen_fn(source)
        return KernelGenerativeFunction((sample, target), static_gen_fn_callable)

    return inner
