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


from genjax._src.core.generative import GenerativeFunction
from genjax._src.core.traceback_util import register_exclusion
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.combinators.compose_combinator import (
    compose_combinator,
)
from genjax._src.generative_functions.combinators.switch_combinator import (
    switch_combinator,
)
from genjax._src.generative_functions.distributions.tensorflow_probability import (
    categorical,
)
from genjax._src.generative_functions.static.static_gen_fn import static_gen_fn

register_exclusion(__file__)


@typecheck
def mixture_combinator(
    *gen_fns: GenerativeFunction,
) -> GenerativeFunction:
    def argument_pushforward(mixture_logits, *args):
        return (mixture_logits, *args)

    inner_combinator_closure = switch_combinator(*gen_fns)

    @static_gen_fn
    def mixture_model(mixture_logits, *args):
        mix_idx = categorical(logits=mixture_logits) @ "mixture_component"
        v = inner_combinator_closure(mix_idx, *args) @ "component_sample"
        return v

    return compose_combinator(
        mixture_model,
        pre=argument_pushforward,
        info="Derived combinator (Mixture)",
    )
