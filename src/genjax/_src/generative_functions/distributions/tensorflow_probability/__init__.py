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

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from genjax._src.core.datatypes.generative import TraceType
from genjax._src.core.datatypes.generative import tt_lift
from genjax._src.core.typing import Any
from genjax._src.core.typing import Sequence
from genjax._src.generative_functions.distributions.distribution import ExactDensity


tfd = tfp.distributions


@dataclass
class TFPDistribution(ExactDensity):
    distribution: Any

    def flatten(self):
        return (), (self.distribution,)

    @classmethod
    def new(cls, tfp_d):
        new = TFPDistribution(tfp_d)
        functools.update_wrapper(new, tfp_d)
        return new

    def make_tfp_distribution(self, *args, **kwargs):
        return self.distribution(*args, **kwargs)

    def sample(self, key, *args, **kwargs):
        dist = self.make_tfp_distribution(*args, **kwargs)
        return dist.sample(seed=key)

    def logpdf(self, v, *args, **kwargs):
        dist = self.make_tfp_distribution(*args, **kwargs)
        return jnp.sum(dist.log_prob(v))


tfp_bates = TFPDistribution.new(tfd.Bates)
tfp_chi = TFPDistribution.new(tfd.Chi)
tfp_chi2 = TFPDistribution.new(tfd.Chi2)
tfp_geometric = TFPDistribution.new(tfd.Geometric)
tfp_gumbel = TFPDistribution.new(tfd.Gumbel)
tfp_half_cauchy = TFPDistribution.new(tfd.HalfCauchy)
tfp_half_normal = TFPDistribution.new(tfd.HalfNormal)
tfp_half_student_t = TFPDistribution.new(tfd.HalfStudentT)
tfp_inverse_gamma = TFPDistribution.new(tfd.InverseGamma)
tfp_kumaraswamy = TFPDistribution.new(tfd.Kumaraswamy)
tfp_logit_normal = TFPDistribution.new(tfd.LogitNormal)
tfp_moyal = TFPDistribution.new(tfd.Moyal)
tfp_multinomial = TFPDistribution.new(tfd.Multinomial)
tfp_negative_binomial = TFPDistribution.new(tfd.NegativeBinomial)
tfp_plackett_luce = TFPDistribution.new(tfd.PlackettLuce)
tfp_power_spherical = TFPDistribution.new(tfd.PowerSpherical)
tfp_skellam = TFPDistribution.new(tfd.Skellam)
tfp_student_t = TFPDistribution.new(tfd.StudentT)
tfp_normal = TFPDistribution.new(tfd.Normal)
tfp_mv_normal_diag = TFPDistribution.new(tfd.MultivariateNormalDiag)
tfp_mv_normal = TFPDistribution.new(tfd.MultivariateNormalFullCovariance)
tfp_categorical = TFPDistribution.new(tfd.Categorical)
tfp_truncated_cauchy = TFPDistribution.new(tfd.TruncatedCauchy)
tfp_truncated_normal = TFPDistribution.new(tfd.TruncatedNormal)
tfp_uniform = TFPDistribution.new(tfd.Uniform)
tfp_von_mises = TFPDistribution.new(tfd.VonMises)
tfp_von_mises_fisher = TFPDistribution.new(tfd.VonMisesFisher)
tfp_weibull = TFPDistribution.new(tfd.Weibull)
tfp_zipf = TFPDistribution.new(tfd.Zipf)


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
