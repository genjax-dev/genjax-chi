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

from genjax._src.core.datatypes.generative import TraceType
from genjax._src.core.datatypes.generative import tt_lift
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


tfp_bates = TFPDistribution(tfd.Bates)
tfp_chi = TFPDistribution(tfd.Chi)
tfp_chi2 = TFPDistribution(tfd.Chi2)
tfp_geometric = TFPDistribution(tfd.Geometric)
tfp_gumbel = TFPDistribution(tfd.Gumbel)
tfp_half_cauchy = TFPDistribution(tfd.HalfCauchy)
tfp_half_normal = TFPDistribution(tfd.HalfNormal)
tfp_half_student_t = TFPDistribution(tfd.HalfStudentT)
tfp_inverse_gamma = TFPDistribution(tfd.InverseGamma)
tfp_kumaraswamy = TFPDistribution(tfd.Kumaraswamy)
tfp_logit_normal = TFPDistribution(tfd.LogitNormal)
tfp_moyal = TFPDistribution(tfd.Moyal)
tfp_multinomial = TFPDistribution(tfd.Multinomial)
tfp_negative_binomial = TFPDistribution(tfd.NegativeBinomial)
tfp_plackett_luce = TFPDistribution(tfd.PlackettLuce)
tfp_power_spherical = TFPDistribution(tfd.PowerSpherical)
tfp_skellam = TFPDistribution(tfd.Skellam)
tfp_student_t = TFPDistribution(tfd.StudentT)
tfp_normal = TFPDistribution(tfd.Normal)
tfp_mv_normal_diag = TFPDistribution(tfd.MultivariateNormalDiag)
tfp_mv_normal = TFPDistribution(tfd.MultivariateNormalFullCovariance)
tfp_categorical = TFPDistribution(tfd.Categorical)
tfp_truncated_cauchy = TFPDistribution(tfd.TruncatedCauchy)
tfp_truncated_normal = TFPDistribution(tfd.TruncatedNormal)
tfp_uniform = TFPDistribution(tfd.Uniform)
tfp_von_mises = TFPDistribution(tfd.VonMises)
tfp_von_mises_fisher = TFPDistribution(tfd.VonMisesFisher)
tfp_weibull = TFPDistribution(tfd.Weibull)
tfp_zipf = TFPDistribution(tfd.Zipf)


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


tfp_mixture = TFPMixture
