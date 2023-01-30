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

from genjax._src.prox import ChangeTarget
from genjax._src.prox import ChoiceMapDistribution
from genjax._src.prox import Compose
from genjax._src.prox import Extend
from genjax._src.prox import Importance
from genjax._src.prox import Init
from genjax._src.prox import Marginal
from genjax._src.prox import ParticleCollection
from genjax._src.prox import ProxDistribution
from genjax._src.prox import Sequence
from genjax._src.prox import SMCChangeTargetPropagator
from genjax._src.prox import SMCCompose
from genjax._src.prox import SMCExtendPropagator
from genjax._src.prox import SMCInit
from genjax._src.prox import SMCPropagator
from genjax._src.prox import SMCSequencePropagator
from genjax._src.prox import Target
from genjax._src.prox import chm_dist
from genjax._src.prox import importance
from genjax._src.prox import marginal
from genjax._src.prox import target


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
]
