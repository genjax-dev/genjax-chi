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

Lew's framework for programming with composable estimatedimate densities
(RAVI/GenSP)
https://arxiv.org/abs/2203.02836
and his Gen implementation
GenGenSP.
"""

from genjax._src.gensp.choice_map_distribution import ChoiceMapDistribution
from genjax._src.gensp.choice_map_distribution import choice_map_distribution
from genjax._src.gensp.inference.importance_sampling import CustomImportance
from genjax._src.gensp.inference.importance_sampling import DefaultImportance
from genjax._src.gensp.inference.importance_sampling import importance_sampler
from genjax._src.gensp.inference.sequential_monte_carlo import ChangeTarget
from genjax._src.gensp.inference.sequential_monte_carlo import Compose
from genjax._src.gensp.inference.sequential_monte_carlo import Extend
from genjax._src.gensp.inference.sequential_monte_carlo import Init
from genjax._src.gensp.inference.sequential_monte_carlo import ParticleCollection
from genjax._src.gensp.inference.sequential_monte_carlo import Sequence
from genjax._src.gensp.inference.sequential_monte_carlo import SMCChangeTargetPropagator
from genjax._src.gensp.inference.sequential_monte_carlo import SMCCompose
from genjax._src.gensp.inference.sequential_monte_carlo import SMCExtendPropagator
from genjax._src.gensp.inference.sequential_monte_carlo import SMCInit
from genjax._src.gensp.inference.sequential_monte_carlo import SMCPropagator
from genjax._src.gensp.inference.sequential_monte_carlo import SMCSequencePropagator
from genjax._src.gensp.marginal import Marginal
from genjax._src.gensp.marginal import marginal
from genjax._src.gensp.nablasp import ADEVDistribution
from genjax._src.gensp.nablasp import do_lower
from genjax._src.gensp.nablasp import do_upper
from genjax._src.gensp.nablasp import flip_enum
from genjax._src.gensp.nablasp import geometric_reinforce
from genjax._src.gensp.nablasp import iwae_importance
from genjax._src.gensp.nablasp import lower
from genjax._src.gensp.nablasp import mv_normal_diag_reparam
from genjax._src.gensp.nablasp import mv_normal_reparam
from genjax._src.gensp.nablasp import normal_reinforce
from genjax._src.gensp.nablasp import normal_reparam
from genjax._src.gensp.nablasp import upper
from genjax._src.gensp.target import Target
from genjax._src.gensp.target import accum_score
from genjax._src.gensp.target import target
from genjax._src.gensp.utils import merge
from genjax._src.gensp.utils import static_check_supports


__all__ = [
    "Target",
    "target",
    "Marginal",
    "marginal",
    "ChoiceMapDistribution",
    "choice_map_distribution",
    "CustomImportance",
    "DefaultImportance",
    "importance_sampler",
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
    "merge",
    "accum_score",
    "flip_enum",
    "normal_reinforce",
    "normal_reparam",
    "mv_normal_reparam",
    "mv_normal_diag_reparam",
    "geometric_reinforce",
    "upper",
    "do_upper",
    "lower",
    "do_lower",
    "ADEVDistribution",
    "iwae_importance",
]
