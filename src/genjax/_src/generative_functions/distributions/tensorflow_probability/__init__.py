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
from typing import Sequence

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from genjax._src.core.datatypes import TraceType
from genjax._src.core.datatypes.tracetypes import tt_lift
from genjax._src.generative_functions.distributions.distribution import ExactDensity


tfd = tfp.distributions


@dataclass
class TFPDistribution(ExactDensity):
    distribution: Any

    def flatten(self):
        return (), (self.distribution,)

    def make_tfp_distribution(self, *args, **kwargs):
        return self.distribution(*args, **kwargs)

    def sample(self, key, *args, **kwargs):
        dist = self.make_tfp_distribution(*args, **kwargs)
        return dist.sample(seed=key)

    def logpdf(self, v, *args, **kwargs):
        dist = self.make_tfp_distribution(*args, **kwargs)
        return jnp.sum(dist.log_prob(v))


TFPBates = TFPDistribution(tfd.Bates)
TFPChi = TFPDistribution(tfd.Chi)
TFPChi2 = TFPDistribution(tfd.Chi2)
TFPGeometric = TFPDistribution(tfd.Geometric)
TFPGumbel = TFPDistribution(tfd.Gumbel)
TFPHalfCauchy = TFPDistribution(tfd.HalfCauchy)
TFPHalfNormal = TFPDistribution(tfd.HalfNormal)
TFPHalfStudentT = TFPDistribution(tfd.HalfStudentT)
TFPInverseGamma = TFPDistribution(tfd.InverseGamma)
TFPKumaraswamy = TFPDistribution(tfd.Kumaraswamy)
TFPLogitNormal = TFPDistribution(tfd.LogitNormal)
TFPMoyal = TFPDistribution(tfd.Moyal)
TFPMultinomial = TFPDistribution(tfd.Multinomial)
TFPNegativeBinomial = TFPDistribution(tfd.NegativeBinomial)
TFPPlackettLuce = TFPDistribution(tfd.PlackettLuce)
TFPPowerSpherical = TFPDistribution(tfd.PowerSpherical)
TFPSkellam = TFPDistribution(tfd.Skellam)
TFPStudentT = TFPDistribution(tfd.StudentT)
TFPNormal = TFPDistribution(tfd.Normal)
TFPCategorical = TFPDistribution(tfd.Categorical)
TFPTruncatedCauchy = TFPDistribution(tfd.TruncatedCauchy)
TFPTruncatedNormal = TFPDistribution(tfd.TruncatedNormal)
TFPUniform = TFPDistribution(tfd.Uniform)
TFPVonMises = TFPDistribution(tfd.VonMises)
TFPVonMisesFisher = TFPDistribution(tfd.VonMisesFisher)
TFPWeibull = TFPDistribution(tfd.Weibull)
TFPZipf = TFPDistribution(tfd.Zipf)


@dataclass
class TFPMixture(ExactDensity):
    cat: TFPDistribution
    components: Sequence[TFPDistribution]

    def flatten(self):
        return (), (self.cat, self.components)

    def make_tfp_distribution(self, cat_args, component_args):
        cat = self.cat.make_tfp_distribution(cat_args)
        components = list(
            map(
                lambda v: v[0].make_tfp_distribution(*v[1]),
                zip(self.components, component_args),
            )
        )
        return tfd.Mixture(cat=cat, components=components)

    def sample(self, key, cat_args, component_args, **kwargs):
        mix = self.make_tfp_distribution(cat_args, component_args)
        return mix.sample(seed=key)

    def logpdf(self, v, cat_args, component_args, **kwargs):
        mix = self.make_tfp_distribution(cat_args, component_args)
        return jnp.sum(mix.log_prob(v))
