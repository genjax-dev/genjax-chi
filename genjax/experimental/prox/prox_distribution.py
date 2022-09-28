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

import abc
from dataclasses import dataclass
from genjax.distributions.distribution import (
    Distribution,
    DistributionTrace,
    ValueChoiceMap,
)


@dataclass
class ProxDistribution(Distribution):
    @abc.abstractmethod
    def random_weighted(self, key, target):
        pass

    @abc.abstractmethod
    def estimate_logpdf(self, key, v, target):
        pass

    def __call__(self, key, target):
        key, (v, w) = self.random_weighted(key, target)
        return (key, v)

    def sample(self, key, target):
        _, (v, _) = self.random_weighted(key, target)
        return v

    def logpdf(self, key, v, target):
        _, w, _ = self.estimate_logpdf(key, v, target)
        return w

    def simulate(self, key, args):
        key, (val, weight) = self.random_weighted(key, *args)
        val = val.strip_metadata()
        chm = ValueChoiceMap.new(val)
        return key, DistributionTrace(self, args, chm, weight)

    def importance(self, key, chm, args):
        assert isinstance(chm, ValueChoiceMap)
        val = chm.get_leaf_value()
        val = val.strip_metadata()
        key, w, new = self.estimate_logpdf(key, val, *args)
        tr = DistributionTrace(self, args, new, w)
        return key, (w, tr)
