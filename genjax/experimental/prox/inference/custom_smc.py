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

from dataclasses import dataclass
from genjax.experimental.prox.target import Target
from genjax.experimental.prox.prox_distribution import ProxDistribution
from typing import Any, Callable


@dataclass
class CustomSMC(ProxDistribution):
    initial_state: Callable[[Target], Any]
    step_model: Callable[[Any, Target], Target]
    step_proposal: ProxDistribution
