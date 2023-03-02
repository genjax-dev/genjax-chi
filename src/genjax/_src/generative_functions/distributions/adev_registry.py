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

"""Defines and registers ADEV primitives for several `Distribution` generative
functions."""

import dataclasses

from genjax._src.core.transforms.adev import ADEVPrimitive
from genjax._src.core.transforms.adev import SupportsMVD
from genjax._src.core.transforms.adev import SupportsReinforce
from genjax._src.core.transforms.adev import register
from genjax._src.generative_functions.distributions.scipy.normal import Normal
from genjax._src.generative_functions.distributions.scipy.normal import _Normal


#####
# Normal
#####


@dataclasses.dataclass
class ADEVPrimNormal(ADEVPrimitive, SupportsMVD, SupportsReinforce):
    def simulate(self, key, args):
        key, tr = Normal.simulate(key, args)
        v = tr.get_retval()
        return key, v

    def reinforce_estimate(self, key, duals, kont):
        pass

    def mvd_estimate(self, key, duals, kont):
        pass


register(_Normal, ADEVPrimNormal)
