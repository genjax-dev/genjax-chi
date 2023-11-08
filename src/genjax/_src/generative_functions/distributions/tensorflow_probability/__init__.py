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
from genjax._src.core.typing import Callable
from genjax._src.core.typing import Sequence
from genjax._src.generative_functions.distributions.distribution import ExactDensity


tfd = tfp.distributions


@dataclass
class TFPDistribution(ExactDensity):
    """
    A `GenerativeFunction` wrapper around [TensorFlow Probability distributions](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions).

    Implements the `ExactDensity` subclass of `genjax.Distribution` automatically using the interfaces defined for `tfp.distributions` objects.
    """

    make_tfp_distribution: Callable

    def flatten(self):
        return (), (self.make_tfp_distribution,)

    @classmethod
    def new(cls, tfp_d):
        new = TFPDistribution(tfp_d)
        functools.update_wrapper(new, tfp_d)
        return new

    def sample(self, key, *args, **kwargs):
        dist = self.make_tfp_distribution(*args, **kwargs)
        return dist.sample(seed=key)

    def logpdf(self, v, *args, **kwargs):
        dist = self.make_tfp_distribution(*args, **kwargs)
        lp = dist.log_prob(v)
        if lp.shape:
            return jnp.sum(dist.log_prob(v))
        else:
            return lp


#####################
# Wrapper instances #
#####################

tfp_bates = TFPDistribution.new(tfd.Bates)
"""
A `TFPDistribution` generative function which wraps the [`tfd.Bates`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Bates) distribution from TensorFlow Probability distributions.
"""

tfp_bernoulli = TFPDistribution.new(lambda logits: tfd.Bernoulli(logits=logits))
"""
A `TFPDistribution` generative function which wraps the [`tfd.Bernoulli`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Bernoulli) distribution from TensorFlow Probability distributions.
"""

tfp_chi = TFPDistribution.new(tfd.Chi)
"""
A `TFPDistribution` generative function which wraps the [`tfd.Chi`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Chi) distribution from TensorFlow Probability distributions.
"""

tfp_chi2 = TFPDistribution.new(tfd.Chi2)
"""
A `TFPDistribution` generative function which wraps the [`tfd.Chi2`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Chi2) distribution from TensorFlow Probability distributions.
"""

tfp_geometric = TFPDistribution.new(tfd.Geometric)
"""
A `TFPDistribution` generative function which wraps the [`tfd.Geometric`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Geometric) distribution from TensorFlow Probability distributions.
"""

tfp_gumbel = TFPDistribution.new(tfd.Gumbel)
"""
A `TFPDistribution` generative function which wraps the [`tfd.Gumbel`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Gumbel) distribution from TensorFlow Probability distributions.
"""

tfp_half_cauchy = TFPDistribution.new(tfd.HalfCauchy)
"""
A `TFPDistribution` generative function which wraps the [`tfd.HalfCauchy`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/HalfCauchy) distribution from TensorFlow Probability distributions.
"""

tfp_half_normal = TFPDistribution.new(tfd.HalfNormal)
"""
A `TFPDistribution` generative function which wraps the [`tfd.HalfNormal`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/HalfNormal) distribution from TensorFlow Probability distributions.
"""

tfp_half_student_t = TFPDistribution.new(tfd.HalfStudentT)
"""
A `TFPDistribution` generative function which wraps the [`tfd.HalfStudentT`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/HalfStudentT) distribution from TensorFlow Probability distributions.
"""

tfp_inverse_gamma = TFPDistribution.new(tfd.InverseGamma)
"""
A `TFPDistribution` generative function which wraps the [`tfd.InverseGamma`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/InverseGamma) distribution from TensorFlow Probability distributions.
"""

tfp_kumaraswamy = TFPDistribution.new(tfd.Kumaraswamy)
"""
A `TFPDistribution` generative function which wraps the [`tfd.Kumaraswamy`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Kumaraswamy) distribution from TensorFlow Probability distributions.
"""

tfp_logit_normal = TFPDistribution.new(tfd.LogitNormal)
"""
A `TFPDistribution` generative function which wraps the [`tfd.LogitNormal`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/LogitNormal) distribution from TensorFlow Probability distributions.
"""

tfp_moyal = TFPDistribution.new(tfd.Moyal)
"""
A `TFPDistribution` generative function which wraps the [`tfd.Moyal`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Moyal) distribution from TensorFlow Probability distributions.
"""

tfp_multinomial = TFPDistribution.new(tfd.Multinomial)
"""
A `TFPDistribution` generative function which wraps the [`tfd.Multinomial`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Multinomial) distribution from TensorFlow Probability distributions.
"""

tfp_negative_binomial = TFPDistribution.new(tfd.NegativeBinomial)
"""
A `TFPDistribution` generative function which wraps the [`tfd.NegativeBinomial`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/NegativeBinomial) distribution from TensorFlow Probability distributions.
"""

tfp_plackett_luce = TFPDistribution.new(tfd.PlackettLuce)
"""
A `TFPDistribution` generative function which wraps the [`tfd.PlackettLuce`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/PlackettLuce) distribution from TensorFlow Probability distributions.
"""

tfp_power_spherical = TFPDistribution.new(tfd.PowerSpherical)
"""
A `TFPDistribution` generative function which wraps the [`tfd.PowerSpherical`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/PowerSpherical) distribution from TensorFlow Probability distributions.
"""

tfp_skellam = TFPDistribution.new(tfd.Skellam)
"""
A `TFPDistribution` generative function which wraps the [`tfd.Skellam`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Skellam) distribution from TensorFlow Probability distributions.
"""

tfp_student_t = TFPDistribution.new(tfd.StudentT)
"""
A `TFPDistribution` generative function which wraps the [`tfd.StudentT`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/StudentT) distribution from TensorFlow Probability distributions.
"""

tfp_normal = TFPDistribution.new(tfd.Normal)
"""
A `TFPDistribution` generative function which wraps the [`tfd.Normal`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Normal) distribution from TensorFlow Probability distributions.
"""

tfp_mv_normal_diag = TFPDistribution.new(
    lambda μ, Σ_diag: tfd.MultivariateNormalDiag(loc=μ, scale_diag=Σ_diag)
)
"""
A `TFPDistribution` generative function which wraps the [`tfd.MultivariateNormalDiag`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MultivariateNormalDiag) distribution from TensorFlow Probability distributions.
"""

tfp_mv_normal = TFPDistribution.new(tfd.MultivariateNormalFullCovariance)
"""
A `TFPDistribution` generative function which wraps the [`tfd.MultivariateNormalFullCovariance`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MultivariateNormalFullCovariance) distribution from TensorFlow Probability distributions.
"""

tfp_categorical = TFPDistribution.new(lambda logits: tfd.Categorical(logits=logits))
"""
A `TFPDistribution` generative function which wraps the [`tfd.Categorical`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Categorical) distribution from TensorFlow Probability distributions.
"""

tfp_truncated_cauchy = TFPDistribution.new(tfd.TruncatedCauchy)
"""
A `TFPDistribution` generative function which wraps the [`tfd.TruncatedCauchy`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/TruncatedCauchy) distribution from TensorFlow Probability distributions.
"""

tfp_truncated_normal = TFPDistribution.new(tfd.TruncatedNormal)
"""
A `TFPDistribution` generative function which wraps the [`tfd.TruncatedNormal`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/TruncatedNormal) distribution from TensorFlow Probability distributions.
"""

tfp_uniform = TFPDistribution.new(tfd.Uniform)
"""
A `TFPDistribution` generative function which wraps the [`tfd.Uniform`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Uniform) distribution from TensorFlow Probability distributions.
"""

tfp_von_mises = TFPDistribution.new(tfd.VonMises)
"""
A `TFPDistribution` generative function which wraps the [`tfd.VonMises`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/VonMises) distribution from TensorFlow Probability distributions.
"""

tfp_von_mises_fisher = TFPDistribution.new(tfd.VonMisesFisher)
"""
A `TFPDistribution` generative function which wraps the [`tfd.VonMisesFisher`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/VonMisesFisher) distribution from TensorFlow Probability distributions.
"""

tfp_weibull = TFPDistribution.new(tfd.Weibull)
"""
A `TFPDistribution` generative function which wraps the [`tfd.Weibull`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Weibull) distribution from TensorFlow Probability distributions.
"""

tfp_zipf = TFPDistribution.new(tfd.Zipf)
"""
A `TFPDistribution` generative function which wraps the [`tfd.Zipf`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Zipf) distribution from TensorFlow Probability distributions.
"""

######################
# Other constructors #
######################


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
