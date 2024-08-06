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

from genjax._src.core.typing import Any, Callable
from genjax._src.generative_functions.distributions.distribution import exact_density

tfd = tfp.distributions


def tfp_distribution(dist: Callable[..., Any]):
    def sampler(key, *args, **kwargs):
        d = dist(*args, **kwargs)
        return d.sample(seed=key)

    def logpdf(v, *args, **kwargs):
        d = dist(*args, **kwargs)
        return d.log_prob(v)

    return exact_density(sampler, logpdf)


#####################
# Wrapper instances #
#####################

beta = tfp_distribution(tfd.Beta)
"""
A `tfp_distribution` generative function which wraps the [`tfd.Beta`](https:
//www.tensorflow.org/probability/api_docs/python/tfp/distributions/Beta)
distribution from TensorFlow Probability distributions.
"""

bates = tfp_distribution(tfd.Bates)
"""A `tfp_distribution` generative function which wraps the [`tfd.Bates`](https
://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Bates)
distribution from TensorFlow Probability distributions."""

bernoulli = tfp_distribution(lambda logits: tfd.Bernoulli(logits=logits))
"""A `tfp_distribution` generative function which wraps the [`tfd.Bernoulli`](h
ttps://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Bernoul
li) distribution from TensorFlow Probability distributions."""

flip = tfp_distribution(lambda p: tfd.Bernoulli(probs=p))
"""A `tfp_distribution` generative function which wraps the [`tfd.Bernoulli`](h
ttps://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Bernoul
li) distribution from TensorFlow Probability distributions, but is constructed
using a probability value and not a logit."""

chi = tfp_distribution(tfd.Chi)
"""
A `tfp_distribution` generative function which wraps the [`tfd.Chi`](https:/
/www.tensorflow.org/probability/api_docs/python/tfp/distributions/Chi)
distribution from TensorFlow Probability distributions.
"""

chi2 = tfp_distribution(tfd.Chi2)
"""
A `tfp_distribution` generative function which wraps the [`tfd.Chi2`](https:
//www.tensorflow.org/probability/api_docs/python/tfp/distributions/Chi2)
distribution from TensorFlow Probability distributions.
"""

geometric = tfp_distribution(tfd.Geometric)
"""A `tfp_distribution` generative function which wraps the [`tfd.Geometric`](h
ttps://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Geometr
ic) distribution from TensorFlow Probability distributions."""

gumbel = tfp_distribution(tfd.Gumbel)
"""A `tfp_distribution` generative function which wraps the [`tfd.Gumbel`](http
s://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Gumbel)
distribution from TensorFlow Probability distributions."""

half_cauchy = tfp_distribution(tfd.HalfCauchy)
"""
A `tfp_distribution` generative function which wraps the [`tfd.HalfCauchy`](
https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/HalfCa
uchy) distribution from TensorFlow Probability distributions.
"""

half_normal = tfp_distribution(tfd.HalfNormal)
"""
A `tfp_distribution` generative function which wraps the [`tfd.HalfNormal`](
https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/HalfNo
rmal) distribution from TensorFlow Probability distributions.
"""

half_student_t = tfp_distribution(tfd.HalfStudentT)
"""
A `tfp_distribution` generative function which wraps the [`tfd.HalfStudentT`
](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Half
StudentT) distribution from TensorFlow Probability distributions.
"""

inverse_gamma = tfp_distribution(tfd.InverseGamma)
"""
A `tfp_distribution` generative function which wraps the [`tfd.InverseGamma`
](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Inve
rseGamma) distribution from TensorFlow Probability distributions.
"""

kumaraswamy = tfp_distribution(tfd.Kumaraswamy)
"""
A `tfp_distribution` generative function which wraps the [`tfd.Kumaraswamy`]
(https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Kumar
aswamy) distribution from TensorFlow Probability distributions.
"""

laplace = tfp_distribution(tfd.Laplace)
"""A `tfp_distribution` generative function which wraps the [`tfd.Laplace`](htt
ps://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Laplace)
distribution from TensorFlow Probability distributions."""

logit_normal = tfp_distribution(tfd.LogitNormal)
"""
A `tfp_distribution` generative function which wraps the [`tfd.LogitNormal`]
(https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Logit
Normal) distribution from TensorFlow Probability distributions.
"""

moyal = tfp_distribution(tfd.Moyal)
"""A `tfp_distribution` generative function which wraps the [`tfd.Moyal`](https
://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Moyal)
distribution from TensorFlow Probability distributions."""

multinomial = tfp_distribution(tfd.Multinomial)
"""
A `tfp_distribution` generative function which wraps the [`tfd.Multinomial`]
(https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Multi
nomial) distribution from TensorFlow Probability distributions.
"""

negative_binomial = tfp_distribution(tfd.NegativeBinomial)
"""
A `tfp_distribution` generative function which wraps the [`tfd.NegativeBinom
ial`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/
NegativeBinomial) distribution from TensorFlow Probability distributions.
"""

plackett_luce = tfp_distribution(tfd.PlackettLuce)
"""
A `tfp_distribution` generative function which wraps the [`tfd.PlackettLuce`
](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Plac
kettLuce) distribution from TensorFlow Probability distributions.
"""

power_spherical = tfp_distribution(tfd.PowerSpherical)
"""
A `tfp_distribution` generative function which wraps the [`tfd.PowerSpherica
l`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Po
werSpherical) distribution from TensorFlow Probability distributions.
"""

skellam = tfp_distribution(tfd.Skellam)
"""A `tfp_distribution` generative function which wraps the [`tfd.Skellam`](htt
ps://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Skellam)
distribution from TensorFlow Probability distributions."""

student_t = tfp_distribution(tfd.StudentT)
"""A `tfp_distribution` generative function which wraps the
[`tfd.StudentT`](ht.

tps://www.tensorflow.org/probability/api_docs/python/tfp/distributions/StudentT
) distribution from TensorFlow Probability distributions.

"""

normal = tfp_distribution(tfd.Normal)
"""A `tfp_distribution` generative function which wraps the [`tfd.Normal`](http
s://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Normal)
distribution from TensorFlow Probability distributions."""

mv_normal_diag = tfp_distribution(
    lambda μ, Σ_diag: tfd.MultivariateNormalDiag(loc=μ, scale_diag=Σ_diag)
)
"""
A `tfp_distribution` generative function which wraps the [`tfd.MultivariateN
ormalDiag`](https://www.tensorflow.org/probability/api_docs/python/tfp/distribu
tions/MultivariateNormalDiag) distribution from TensorFlow Probability
distributions.
"""

mv_normal = tfp_distribution(tfd.MultivariateNormalFullCovariance)
"""
A `tfp_distribution` generative function which wraps the [`tfd.MultivariateN
ormalFullCovariance`](https://www.tensorflow.org/probability/api_docs/python/tf
p/distributions/MultivariateNormalFullCovariance) distribution from TensorFlow
Probability distributions.
"""

categorical = tfp_distribution(lambda logits: tfd.Categorical(logits=logits))
"""
A `tfp_distribution` generative function which wraps the [`tfd.Categorical`]
(https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Categ
orical) distribution from TensorFlow Probability distributions.
"""

truncated_cauchy = tfp_distribution(tfd.TruncatedCauchy)
"""
A `tfp_distribution` generative function which wraps the [`tfd.TruncatedCauc
hy`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/T
runcatedCauchy) distribution from TensorFlow Probability distributions.
"""

truncated_normal = tfp_distribution(tfd.TruncatedNormal)
"""
A `tfp_distribution` generative function which wraps the [`tfd.TruncatedNorm
al`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/T
runcatedNormal) distribution from TensorFlow Probability distributions.
"""

uniform = tfp_distribution(tfd.Uniform)
"""A `tfp_distribution` generative function which wraps the [`tfd.Uniform`](htt
ps://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Uniform)
distribution from TensorFlow Probability distributions."""

von_mises = tfp_distribution(tfd.VonMises)
"""A `tfp_distribution` generative function which wraps the
[`tfd.VonMises`](ht.

tps://www.tensorflow.org/probability/api_docs/python/tfp/distributions/VonMises
) distribution from TensorFlow Probability distributions.

"""

von_mises_fisher = tfp_distribution(tfd.VonMisesFisher)
"""
A `tfp_distribution` generative function which wraps the [`tfd.VonMisesFishe
r`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Vo
nMisesFisher) distribution from TensorFlow Probability distributions.
"""

weibull = tfp_distribution(tfd.Weibull)
"""A `tfp_distribution` generative function which wraps the [`tfd.Weibull`](htt
ps://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Weibull)
distribution from TensorFlow Probability distributions."""

zipf = tfp_distribution(tfd.Zipf)
"""
A `tfp_distribution` generative function which wraps the [`tfd.Zipf`](https:
//www.tensorflow.org/probability/api_docs/python/tfp/distributions/Zipf)
distribution from TensorFlow Probability distributions.
"""
