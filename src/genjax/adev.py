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

from genjax._src.core.transforms.adev import ADEVPrimitive
from genjax._src.core.transforms.adev import ADEVProgram
from genjax._src.core.transforms.adev import ADEVTerm
from genjax._src.core.transforms.adev import GradStratEnum
from genjax._src.core.transforms.adev import GradStratMVD
from genjax._src.core.transforms.adev import GradStratREINFORCE
from genjax._src.core.transforms.adev import lang
from genjax._src.core.transforms.adev import sample
from genjax._src.core.transforms.adev import strat
from genjax._src.generative_functions.distributions.adev_registry import (
    ADEVPrimBernoulli,
)
from genjax._src.generative_functions.distributions.adev_registry import ADEVPrimNormal
from genjax._src.generative_functions.distributions.adev_registry import ADEVPrimPoisson


poisson = ADEVPrimPoisson()
normal = ADEVPrimNormal()
bernoulli = ADEVPrimBernoulli()


__all__ = [
    "lang",
    "sample",
    "strat",
    "GradStratREINFORCE",
    "GradStratMVD",
    "GradStratEnum",
    "normal",
    "bernoulli",
    "poisson",
    "ADEVTerm",
    "ADEVPrimitive",
    "ADEVProgram",
]
