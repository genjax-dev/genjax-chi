# Copyright 2024 MIT Probabilistic Computing Project
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


from tensorflow_probability.substrates import jax as tfp

from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import Callable
from genjax._src.generative_functions.distributions.distribution import ExactDensity

tfd = tfp.distributions


def tfp_distribution(dist: Callable):
    @Pytree.partial()
    def sampler(key, *args, **kwargs):
        d = dist(*args, **kwargs)
        return d.sample(seed=key)

    @Pytree.partial()
    def logpdf(v, *args, **kwargs):
        d = dist(*args, **kwargs)
        return d.log_prob(v)

    return ExactDensity(sampler, logpdf)


#####################
# Wrapper instances #
#####################

beta = tfp_distribution(tfd.Beta)
"""
A `tfp_distribution` generative function which wraps the [`tfd.Beta`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Beta) distribution from TensorFlow Probability distributions.
"""

bates = tfp_distribution(tfd.Bates)
"""
A `tfp_distribution` generative function which wraps the [`tfd.Bates`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Bates) distribution from TensorFlow Probability distributions.
"""

bernoulli = tfp_distribution(lambda logits: tfd.Bernoulli(logits=logits))
"""
A `tfp_distribution` generative function which wraps the [`tfd.Bernoulli`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Bernoulli) distribution from TensorFlow Probability distributions.
"""

flip = tfp_distribution(lambda p: tfd.Bernoulli(probs=p))
"""
A `tfp_distribution` generative function which wraps the [`tfd.Bernoulli`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Bernoulli) distribution from TensorFlow Probability distributions, but is constructed using a probability value and not a logit.
"""

chi = tfp_distribution(tfd.Chi)
"""
A `tfp_distribution` generative function which wraps the [`tfd.Chi`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Chi) distribution from TensorFlow Probability distributions.
"""

chi2 = tfp_distribution(tfd.Chi2)
"""
A `tfp_distribution` generative function which wraps the [`tfd.Chi2`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Chi2) distribution from TensorFlow Probability distributions.
"""

geometric = tfp_distribution(tfd.Geometric)
"""
A `tfp_distribution` generative function which wraps the [`tfd.Geometric`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Geometric) distribution from TensorFlow Probability distributions.
"""

gumbel = tfp_distribution(tfd.Gumbel)
"""
A `tfp_distribution` generative function which wraps the [`tfd.Gumbel`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Gumbel) distribution from TensorFlow Probability distributions.
"""

half_cauchy = tfp_distribution(tfd.HalfCauchy)
"""
A `tfp_distribution` generative function which wraps the [`tfd.HalfCauchy`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/HalfCauchy) distribution from TensorFlow Probability distributions.
"""

half_normal = tfp_distribution(tfd.HalfNormal)
"""
A `tfp_distribution` generative function which wraps the [`tfd.HalfNormal`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/HalfNormal) distribution from TensorFlow Probability distributions.
"""

half_student_t = tfp_distribution(tfd.HalfStudentT)
"""
A `tfp_distribution` generative function which wraps the [`tfd.HalfStudentT`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/HalfStudentT) distribution from TensorFlow Probability distributions.
"""

inverse_gamma = tfp_distribution(tfd.InverseGamma)
"""
A `tfp_distribution` generative function which wraps the [`tfd.InverseGamma`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/InverseGamma) distribution from TensorFlow Probability distributions.
"""

kumaraswamy = tfp_distribution(tfd.Kumaraswamy)
"""
A `tfp_distribution` generative function which wraps the [`tfd.Kumaraswamy`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Kumaraswamy) distribution from TensorFlow Probability distributions.
"""

logit_normal = tfp_distribution(tfd.LogitNormal)
"""
A `tfp_distribution` generative function which wraps the [`tfd.LogitNormal`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/LogitNormal) distribution from TensorFlow Probability distributions.
"""

moyal = tfp_distribution(tfd.Moyal)
"""
A `tfp_distribution` generative function which wraps the [`tfd.Moyal`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Moyal) distribution from TensorFlow Probability distributions.
"""

multinomial = tfp_distribution(tfd.Multinomial)
"""
A `tfp_distribution` generative function which wraps the [`tfd.Multinomial`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Multinomial) distribution from TensorFlow Probability distributions.
"""

negative_binomial = tfp_distribution(tfd.NegativeBinomial)
"""
A `tfp_distribution` generative function which wraps the [`tfd.NegativeBinomial`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/NegativeBinomial) distribution from TensorFlow Probability distributions.
"""

plackett_luce = tfp_distribution(tfd.PlackettLuce)
"""
A `tfp_distribution` generative function which wraps the [`tfd.PlackettLuce`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/PlackettLuce) distribution from TensorFlow Probability distributions.
"""

power_spherical = tfp_distribution(tfd.PowerSpherical)
"""
A `tfp_distribution` generative function which wraps the [`tfd.PowerSpherical`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/PowerSpherical) distribution from TensorFlow Probability distributions.
"""

skellam = tfp_distribution(tfd.Skellam)
"""
A `tfp_distribution` generative function which wraps the [`tfd.Skellam`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Skellam) distribution from TensorFlow Probability distributions.
"""

student_t = tfp_distribution(tfd.StudentT)
"""
A `tfp_distribution` generative function which wraps the [`tfd.StudentT`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/StudentT) distribution from TensorFlow Probability distributions.
"""

normal = tfp_distribution(tfd.Normal)
"""
A `tfp_distribution` generative function which wraps the [`tfd.Normal`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Normal) distribution from TensorFlow Probability distributions.
"""

mv_normal_diag = tfp_distribution(
    lambda μ, Σ_diag: tfd.MultivariateNormalDiag(loc=μ, scale_diag=Σ_diag)
)
"""
A `tfp_distribution` generative function which wraps the [`tfd.MultivariateNormalDiag`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MultivariateNormalDiag) distribution from TensorFlow Probability distributions.
"""

mv_normal = tfp_distribution(tfd.MultivariateNormalFullCovariance)
"""
A `tfp_distribution` generative function which wraps the [`tfd.MultivariateNormalFullCovariance`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MultivariateNormalFullCovariance) distribution from TensorFlow Probability distributions.
"""

categorical = tfp_distribution(lambda logits: tfd.Categorical(logits=logits))
"""
A `tfp_distribution` generative function which wraps the [`tfd.Categorical`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Categorical) distribution from TensorFlow Probability distributions.
"""

truncated_cauchy = tfp_distribution(tfd.TruncatedCauchy)
"""
A `tfp_distribution` generative function which wraps the [`tfd.TruncatedCauchy`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/TruncatedCauchy) distribution from TensorFlow Probability distributions.
"""

truncated_normal = tfp_distribution(tfd.TruncatedNormal)
"""
A `tfp_distribution` generative function which wraps the [`tfd.TruncatedNormal`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/TruncatedNormal) distribution from TensorFlow Probability distributions.
"""

uniform = tfp_distribution(tfd.Uniform)
"""
A `tfp_distribution` generative function which wraps the [`tfd.Uniform`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Uniform) distribution from TensorFlow Probability distributions.
"""

von_mises = tfp_distribution(tfd.VonMises)
"""
A `tfp_distribution` generative function which wraps the [`tfd.VonMises`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/VonMises) distribution from TensorFlow Probability distributions.
"""

von_mises_fisher = tfp_distribution(tfd.VonMisesFisher)
"""
A `tfp_distribution` generative function which wraps the [`tfd.VonMisesFisher`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/VonMisesFisher) distribution from TensorFlow Probability distributions.
"""

weibull = tfp_distribution(tfd.Weibull)
"""
A `tfp_distribution` generative function which wraps the [`tfd.Weibull`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Weibull) distribution from TensorFlow Probability distributions.
"""

zipf = tfp_distribution(tfd.Zipf)
"""
A `tfp_distribution` generative function which wraps the [`tfd.Zipf`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Zipf) distribution from TensorFlow Probability distributions.
"""
