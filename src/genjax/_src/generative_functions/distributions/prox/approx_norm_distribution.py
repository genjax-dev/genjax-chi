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

from genjax._src.core.datatypes.generative import emp_chm
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.distributions.prox.prox_distribution import (
    ProxDistribution,
)
from genjax._src.generative_functions.distributions.prox.target import Target
from genjax._src.generative_functions.distributions.prox.unnorm import (
    UnnormalizedMeasureFunction,
)


#######################################
# Approximate normalized distribution #
#######################################


@dataclass
class ApproximateNormalizedDistribution(ProxDistribution):
    unnorm_fn: UnnormalizedMeasureFunction
    custom_q: ProxDistribution

    def flatten(self):
        return (self.unnorm_fn, self.custom_q), ()

    @typecheck
    @classmethod
    def new(cls, custom_q: ProxDistribution, unnorm_fn: UnnormalizedMeasureFunction):
        return ApproximateNormalizedDistribution(unnorm_fn, custom_q)

    def random_weighted(self, key, *args):
        target = Target.new(self.unnorm_fn, args, emp_chm())
        key, (weight, val_chm) = self.custom_q.random_weighted(key, target)
        return key, (weight, val_chm)

    def estimate_logpdf(self, key, val_chm, *args):
        target = Target.new(self.unnorm_fn, args, emp_chm())
        key, w = self.custom_q.estimate_logpdf(key, val_chm, target)
        return key, w


##############
# Shorthands #
##############

approx_norm_dist = ApproximateNormalizedDistribution.new
