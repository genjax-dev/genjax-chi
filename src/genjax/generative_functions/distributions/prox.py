# Copyright 2022 MIT Probabilistic Computing Project
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

"""This module supports a GenJAX implementation of Alexander K.

Lew's framework for programming with composable approximate densities
(RAVI/Prox) https://arxiv.org/abs/2203.02836 and his Gen implementation
GenProx.
"""

from genjax._src.generative_functions.distributions.prox import unnorm
from genjax._src.generative_functions.distributions.prox.approx_norm_distribution import (
    ApproximateNormalizedDistribution,
)
from genjax._src.generative_functions.distributions.prox.approx_norm_distribution import (
    approx_norm_dist,
)
from genjax._src.generative_functions.distributions.prox.choice_map_distribution import (
    ChoiceMapDistribution,
)
from genjax._src.generative_functions.distributions.prox.choice_map_distribution import (
    chm_dist,
)
from genjax._src.generative_functions.distributions.prox.inference.importance import (
    Importance,
)
from genjax._src.generative_functions.distributions.prox.inference.importance import (
    importance,
)
from genjax._src.generative_functions.distributions.prox.inference.sequential_monte_carlo import (
    ChangeTarget,
)
from genjax._src.generative_functions.distributions.prox.inference.sequential_monte_carlo import (
    Compose,
)
from genjax._src.generative_functions.distributions.prox.inference.sequential_monte_carlo import (
    Extend,
)
from genjax._src.generative_functions.distributions.prox.inference.sequential_monte_carlo import (
    Init,
)
from genjax._src.generative_functions.distributions.prox.inference.sequential_monte_carlo import (
    ParticleCollection,
)
from genjax._src.generative_functions.distributions.prox.inference.sequential_monte_carlo import (
    Sequence,
)
from genjax._src.generative_functions.distributions.prox.inference.sequential_monte_carlo import (
    SMCChangeTargetPropagator,
)
from genjax._src.generative_functions.distributions.prox.inference.sequential_monte_carlo import (
    SMCCompose,
)
from genjax._src.generative_functions.distributions.prox.inference.sequential_monte_carlo import (
    SMCExtendPropagator,
)
from genjax._src.generative_functions.distributions.prox.inference.sequential_monte_carlo import (
    SMCInit,
)
from genjax._src.generative_functions.distributions.prox.inference.sequential_monte_carlo import (
    SMCPropagator,
)
from genjax._src.generative_functions.distributions.prox.inference.sequential_monte_carlo import (
    SMCSequencePropagator,
)
from genjax._src.generative_functions.distributions.prox.marginal import Marginal
from genjax._src.generative_functions.distributions.prox.marginal import marginal
from genjax._src.generative_functions.distributions.prox.prox_distribution import (
    ProxDistribution,
)
from genjax._src.generative_functions.distributions.prox.target import Target
from genjax._src.generative_functions.distributions.prox.target import target
from genjax._src.generative_functions.distributions.prox.unnorm import score
from genjax._src.generative_functions.distributions.prox.utils import (
    static_check_supports,
)


__all__ = [
    "Target",
    "target",
    "ProxDistribution",
    "Marginal",
    "marginal",
    "ChoiceMapDistribution",
    "chm_dist",
    "Importance",
    "importance",
    "ParticleCollection",
    "SMCPropagator",
    "SMCExtendPropagator",
    "SMCChangeTargetPropagator",
    "SMCSequencePropagator",
    "SMCInit",
    "SMCCompose",
    "Init",
    "Extend",
    "ChangeTarget",
    "Compose",
    "Sequence",
    "static_check_supports",
    "unnorm",
    "score",
    "approx_norm_dist",
    "ApproximateNormalizedDistribution",
]
