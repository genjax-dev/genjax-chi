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
from typing import Any

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from genjax.core.tracetypes import TraceType
from genjax.generative_functions.builtin.builtin_tracetype import lift
from genjax.generative_functions.distributions.distribution import ExactDensity


tfd = tfp.distributions


@dataclass
class TFPDistribution(ExactDensity):
    distribution: Any

    def flatten(self):
        return (), (self.distribution,)

    def sample(self, key, *args, **kwargs):
        dist = self.distribution(*args, **kwargs)
        return dist.sample(seed=key)

    def logpdf(self, v, *args, **kwargs):
        dist = self.distribution(*args, **kwargs)
        return jnp.sum(dist.log_prob(v))


Bates = TFPDistribution(tfd.Bates)
Chi = TFPDistribution(tfd.Chi)
Chi2 = TFPDistribution(tfd.Chi2)
Geometric = TFPDistribution(tfd.Geometric)
Gumbel = TFPDistribution(tfd.Gumbel)
HalfCauchy = TFPDistribution(tfd.HalfCauchy)
HalfNormal = TFPDistribution(tfd.HalfNormal)
HalfStudentT = TFPDistribution(tfd.HalfStudentT)
InverseGamma = TFPDistribution(tfd.InverseGamma)
Kumaraswamy = TFPDistribution(tfd.Kumaraswamy)
LogitNormal = TFPDistribution(tfd.LogitNormal)
Moyal = TFPDistribution(tfd.Moyal)
Multinomial = TFPDistribution(tfd.Multinomial)
NegativeBinomial = TFPDistribution(tfd.NegativeBinomial)
PlackettLuce = TFPDistribution(tfd.PlackettLuce)
PowerSpherical = TFPDistribution(tfd.PowerSpherical)
Skellam = TFPDistribution(tfd.Skellam)
StudentT = TFPDistribution(tfd.StudentT)
TFNormal = TFPDistribution(tfd.Normal)
TruncatedCauchy = TFPDistribution(tfd.TruncatedCauchy)
TruncatedNormal = TFPDistribution(tfd.TruncatedNormal)
Uniform = TFPDistribution(tfd.Uniform)
VonMises = TFPDistribution(tfd.VonMises)
VonMisesFisher = TFPDistribution(tfd.VonMisesFisher)
Weibull = TFPDistribution(tfd.Weibull)
Zipf = TFPDistribution(tfd.Zipf)
