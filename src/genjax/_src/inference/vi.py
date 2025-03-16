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


import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from genjax._src.adev.core import (
    ADEVPrimitive,
    expectation,
    sample_primitive,
)
from genjax._src.adev.primitives import (
    categorical_enum_parallel,
    flip_enum,
    flip_mvd,
    geometric_reinforce,
    normal_reinforce,
    normal_reparam,
)
from genjax._src.core.generative import Arguments, ChoiceMap
from genjax._src.core.typing import (
    Any,
    Callable,
    FloatArray,
)
from genjax._src.generative_functions.distributions.distribution import (
    ExactDensity,
    exact_density,
)
from genjax._src.generative_functions.distributions.tensorflow_probability import (
    flip,
    geometric,
    normal,
)
from genjax._src.inference.smc import Importance, ImportanceK
from genjax._src.inference.sp import SampleDistribution, Target

tfd = tfp.distributions


##########################################
# Differentiable distribution primitives #
##########################################


def adev_distribution(
    adev_primitive: ADEVPrimitive, differentiable_logpdf: Callable[..., Any], name: str
) -> ExactDensity[Any]:
    """
    Return an [`ExactDensity`][genjax.ExactDensity] distribution whose sampler invokes an ADEV sampling primitive, with a provided differentiable log density function.

    Exact densities created using this function can be used as distributions in variational guide programs.
    """

    def sampler(*args: Any) -> Any:
        return sample_primitive(adev_primitive, *args)

    def logpdf(v: Any, *args: Any) -> FloatArray:
        lp = differentiable_logpdf(v, *args)
        # Branching here is statically resolved.
        if lp.shape:
            return jnp.sum(lp)
        else:
            return lp

    return exact_density(sampler, logpdf, name)


def logpdf(gen_fn):
    return lambda v, *args: gen_fn.assess(ChoiceMap.choice(v), args)[0]


# We import ADEV specific sampling primitives, but then wrap them in
# adev_distribution, for usage inside of generative functions.
flip_enum = adev_distribution(flip_enum, logpdf(flip), "flip_enum")

flip_mvd = adev_distribution(flip_mvd, logpdf(flip), "flip_mvd")

categorical_enum = adev_distribution(
    categorical_enum_parallel,
    lambda v, probs: tfd.Categorical(probs=probs).log_prob(v),
    "categorical_enum",
)

normal_reinforce = adev_distribution(
    normal_reinforce, logpdf(normal), "normal_reinforce"
)

normal_reparam = adev_distribution(normal_reparam, logpdf(normal), "normal_reparam")


geometric_reinforce = adev_distribution(
    geometric_reinforce, logpdf(geometric), "geometric_reinforce"
)


##############
# Loss terms #
##############

GradientEstimate = Any
"""
The type of gradient estimates returned by sampling from gradient estimators for loss terms.
"""


def ELBO(
    guide: SampleDistribution,
    make_target: Callable[..., Target[Any]],
) -> Callable[[Arguments], GradientEstimate]:
    """
    Return a function that computes the gradient estimate of the ELBO loss term.
    """

    def grad_estimate(*args) -> GradientEstimate:
        # In the source language of ADEV.
        @expectation
        def _loss(*args):
            target = make_target(*args)
            guide_alg = Importance(target, guide)
            w = guide_alg.estimate_normalizing_constant(target)
            return -w

        return _loss.grad_estimate(*args)

    return grad_estimate


def IWELBO(
    proposal: SampleDistribution,
    make_target: Callable[[Any], Target[Any]],
    N: int,
) -> Callable[[Arguments], GradientEstimate]:
    """
    Return a function that computes the gradient estimate of the IWELBO loss term.
    """

    def grad_estimate(*args) -> GradientEstimate:
        # In the source language of ADEV.
        @expectation
        def _loss(*args):
            target = make_target(*args)
            guide = ImportanceK(target, proposal, N)
            w = guide.estimate_normalizing_constant(target)
            return -w

        return _loss.grad_estimate(*args)

    return grad_estimate


def PWake(
    posterior_approx: SampleDistribution,
    make_target: Callable[[Any], Target[Any]],
) -> Callable[[Arguments], GradientEstimate]:
    """
    Return a function that computes the gradient estimate of the PWake loss term.
    """

    def grad_estimate(*args) -> GradientEstimate:
        # In the source language of ADEV.
        @expectation
        def _loss(*target_args):
            target = make_target(*target_args)
            _, sample = posterior_approx.random_weighted(target)
            tr, _ = target.importance(sample)
            return -tr.get_score()

        return _loss.grad_estimate(*args)

    return grad_estimate


def QWake(
    proposal: SampleDistribution,
    posterior_approx: SampleDistribution,
    make_target: Callable[[Any], Target[Any]],
) -> Callable[[Arguments], GradientEstimate]:
    """
    Return a function that computes the gradient estimate of the QWake loss term.
    """

    def grad_estimate(*args) -> GradientEstimate:
        # In the source language of ADEV.
        @expectation
        def _loss(*target_args):
            target = make_target(*target_args)
            _, sample = posterior_approx.random_weighted(target)
            w = proposal.estimate_logpdf(sample, target)
            return -w

        return _loss.grad_estimate(*args)

    return grad_estimate
