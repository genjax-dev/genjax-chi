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


from typing import Callable

from genjax._src.core.generative import GenerativeFunction
from genjax._src.core.traceback_util import register_exclusion
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.combinators.compose_combinator import (
    ComposeCombinator,
)
from genjax._src.generative_functions.combinators.switch_combinator import (
    SwitchCombinator,
)
from genjax._src.generative_functions.distributions.tensorflow_probability import (
    categorical,
)
from genjax._src.generative_functions.static import gen

register_exclusion(__file__)


def MixtureCombinator(*gen_fns) -> ComposeCombinator:
    def argument_mapping(mixture_logits, *args):
        return (mixture_logits, *args)

    inner_combinator_closure = SwitchCombinator(gen_fns)

    @gen
    def mixture_model(mixture_logits, *args):
        mix_idx = categorical(logits=mixture_logits) @ "mixture_component"
        v = inner_combinator_closure(mix_idx, *args) @ "component_sample"
        return v

    return ComposeCombinator(
        mixture_model,
        argument_mapping=argument_mapping,
        info="Derived combinator (Mixture)",
    )


@typecheck
def mixture_combinator(
    *gen_fns: GenerativeFunction,
) -> GenerativeFunction:
    def argument_mapping(mixture_logits, *args):
        return (mixture_logits, *args)

    inner_combinator_closure = SwitchCombinator(gen_fns)

    @gen
    def mixture_model(mixture_logits, *args):
        mix_idx = categorical(logits=mixture_logits) @ "mixture_component"
        v = inner_combinator_closure(mix_idx, *args) @ "component_sample"
        return v

    return ComposeCombinator(
        mixture_model,
        argument_mapping=argument_mapping,
        info="Derived combinator (Mixture)",
    )
