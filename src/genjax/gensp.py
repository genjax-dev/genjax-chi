# Copyright 2023 MIT Probabilistic Computing Project
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
"""
This module supports a GenJAX implementation of [Probabilistic programming with stochastic probabilities](https://dl.acm.org/doi/abs/10.1145/3591290).
"""

from genjax._src.gensp.core import Marginal
from genjax._src.gensp.core import marginal
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
from genjax._src.gensp.core import Target
from genjax._src.gensp.core import target


__all__ = [
    "Target",
    "target",
    "Marginal",
    "marginal",
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
]
