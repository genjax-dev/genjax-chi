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

from genjax._src.core.datatypes import ValueChoiceMap
from genjax._src.generative_functions.distributions.distribution import Distribution
from genjax._src.generative_functions.distributions.distribution import (
    DistributionTrace,
)


@dataclass
class ProxDistribution(Distribution):
    def simulate(self, key, args):
        key, (weight, val) = self.random_weighted(key, *args)
        val = val.strip()
        chm = ValueChoiceMap.new(val)
        return key, DistributionTrace(self, args, chm, weight)

    def importance(self, key, chm, args):
        assert isinstance(chm, ValueChoiceMap)
        val = chm.get_leaf_value()
        val = val.strip()
        key, (w, new) = self.estimate_logpdf(key, val, *args)
        tr = DistributionTrace(self, args, new, w)
        return key, (w, tr)
