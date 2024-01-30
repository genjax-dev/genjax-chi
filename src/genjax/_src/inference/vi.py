# Copyright 2023 MIT Probabilistic Computing Project
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

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from genjax._src.adev.core import (
    ADEVPrimitive,
    sample_with_key,
)
from genjax._src.adev.primitives import (
    categorical_enum_parallel,
    flip_enum,
    flip_mvd,
    geometric_reinforce,
    mv_normal_diag_reparam,
    normal_reinforce,
    normal_reparam,
)
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    FloatArray,
    Int,
    PRNGKey,
    Tuple,
)
from genjax._src.generative_functions.distributions.distribution import (
    ExactDensity,
)
from genjax._src.generative_functions.distributions.tensorflow_probability import (
    geometric,
    normal,
    uniform,
)
from genjax._src.inference.core import Target

tfd = tfp.distributions


##########################################
# Differentiable distribution primitives #
##########################################


@dataclass
class ADEVDistribution(ExactDensity):
    """
    The class `ADEVDistribution` is a wrapper class which exposes the `sample` and `logpdf` interfaces, where `sample` utilizes an ADEV differentiable sampling primitive, and `logpdf` is a differentiable logpdf function.
    """

    adev_primitive: ADEVPrimitive
    differentiable_logpdf: Callable = Pytree.static()

    def sample(
        self,
        key: PRNGKey,
        *args: Any,
    ) -> Any:
        return sample_with_key(self.adev_primitive, key, *args)

    def logpdf(
        self,
        v: Any,
        *args: Any,
    ) -> FloatArray:
        lp = self.differentiable_logpdf(v, *args)
        # Branching here is statically resolved.
        if lp.shape:
            return jnp.sum(lp)
        else:
            return lp


# We import ADEV specific sampling primitives, but then wrap them in
# ADEVDistribution, for usage inside of generative functions.
flip_enum = ADEVDistribution(
    flip_enum,
    lambda v, p: tfd.Bernoulli(probs=p).log_prob(v),
)

flip_mvd = ADEVDistribution(
    flip_mvd,
    lambda v, p: tfd.Bernoulli(probs=p).log_prob(v),
)


categorical_enum = ADEVDistribution(
    categorical_enum_parallel,
    lambda v, probs: tfd.Categorical(probs=probs).log_prob(v),
)

normal_reinforce = ADEVDistribution(
    normal_reinforce,
    lambda v, μ, σ: normal.logpdf(v, μ, σ),
)

normal_reparam = ADEVDistribution(
    normal_reparam,
    lambda v, μ, σ: normal.logpdf(v, μ, σ),
)

mv_normal_diag_reparam = ADEVDistribution(
    mv_normal_diag_reparam,
    lambda v, loc, scale_diag: tfd.MultivariateNormalDiag(
        loc=loc, scale_diag=scale_diag
    ).log_prob(v),
)

geometric_reinforce = ADEVDistribution(
    geometric_reinforce,
    lambda v, *args: geometric.logpdf(v, *args),
)

uniform = ADEVDistribution(
    uniform,
    lambda v: uniform.logpdf(v, 0.0, 1.0),
)

##############
# Loss terms #
##############


class IWELBO(Pytree):
    p: ChoiceDistribution
    make_algorithm: Callable
    data: Choice
    N: Int = Pytree.static()

    def gradient_estimate(self, key, p_args: Tuple, q_args: Tuple):
        # In the source language of ADEV.
        @expectation
        def _loss(p_args, q_args):
            target = Target(p, p_args, data)

        return _loss.gradient_estimate(p_args, q_args)


class QWake(Pytree):
    target: Target
    approx: InferenceAlgorithm
    q: ChoiceDistribution
    data: Choice


class PWake(Pytree):
    target: Target
    approx: InferenceAlgorithm
    p: ChoiceDistribution
    data: Choice
