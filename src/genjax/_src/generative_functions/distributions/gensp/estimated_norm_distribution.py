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
from genjax._src.generative_functions.distributions.gensp.gensp_distribution import (
    GenSPDistribution,
)
from genjax._src.generative_functions.distributions.gensp.target import Target
from genjax._src.generative_functions.distributions.gensp.unnorm import (
    UnnormalizedMeasureFunction,
)


#######################################
# Estimated normalized distribution #
#######################################


@dataclass
class EstimatedNormalizedDistribution(GenSPDistribution):
    unnorm_fn: UnnormalizedMeasureFunction
    custom_q: GenSPDistribution

    def flatten(self):
        return (self.unnorm_fn, self.custom_q), ()

    @typecheck
    @classmethod
    def new(cls, custom_q: GenSPDistribution, unnorm_fn: UnnormalizedMeasureFunction):
        return EstimatedNormalizedDistribution(unnorm_fn, custom_q)

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

estimated_norm_dist = EstimatedNormalizedDistribution.new
