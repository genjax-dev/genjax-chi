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


import jax
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
    mv_normal_diag_reparam,
    normal_reinforce,
    normal_reparam,
)
from genjax._src.core.generative import Arguments, ChoiceMap
from genjax._src.core.typing import (
    Any,
    Callable,
    FloatArray,
    Int,
    PRNGKey,
    Tuple,
    typecheck,
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


@typecheck
def adev_distribution(
    adev_primitive: ADEVPrimitive,
    differentiable_logpdf: Callable[..., Any],
) -> ExactDensity:
    """
    Return an [`ExactDensity`][genjax.ExactDensity] distribution whose sampler invokes an ADEV sampling primitive, with a provided differentiable log density function.

    Exact densities created using this function can be used as distributions in variational guide programs.
    """

    def sampler(key: PRNGKey, *args: Any) -> Any:
        return sample_primitive(adev_primitive, *args, key=key)

    def logpdf(v: Any, *args: Any) -> FloatArray:
        lp = differentiable_logpdf(v, *args)
        # Branching here is statically resolved.
        if lp.shape:
            return jnp.sum(lp)
        else:
            return lp

    return exact_density(sampler, logpdf)


def logpdf(gen_fn):
    return lambda v, *args: gen_fn.assess(ChoiceMap.value(v), args)[0]


# We import ADEV specific sampling primitives, but then wrap them in
# adev_distribution, for usage inside of generative functions.
flip_enum = adev_distribution(
    flip_enum,
    logpdf(flip),
)

flip_mvd = adev_distribution(
    flip_mvd,
    logpdf(flip),
)

categorical_enum = adev_distribution(
    categorical_enum_parallel,
    lambda v, probs: tfd.Categorical(probs=probs).log_prob(v),
)

normal_reinforce = adev_distribution(
    normal_reinforce,
    logpdf(normal),
)

normal_reparam = adev_distribution(
    normal_reparam,
    logpdf(normal),
)

mv_normal_diag_reparam = adev_distribution(
    mv_normal_diag_reparam,
    lambda v, loc, scale_diag: tfd.MultivariateNormalDiag(
        loc=loc, scale_diag=scale_diag
    ).log_prob(v),
)

geometric_reinforce = adev_distribution(
    geometric_reinforce,
    logpdf(geometric),
)


##############
# Loss terms #
##############

GradientEstimate = Any
"""
The type of gradient estimates returned by sampling from gradient estimators for loss terms.
"""


@typecheck
def ELBO(
    guide: SampleDistribution,
    make_target: Callable[..., Target],
) -> Callable[[PRNGKey, Arguments], GradientEstimate]:
    """
    Return a function that computes the gradient estimate of the ELBO loss term.
    """

    def grad_estimate(
        key: PRNGKey,
        args: Tuple,
    ) -> Tuple:
        # In the source language of ADEV.
        @expectation
        def _loss(*args):
            target = make_target(*args)
            guide_alg = Importance(target, guide)
            w = guide_alg.estimate_normalizing_constant(key, target)
            return -w

        return _loss.grad_estimate(key, args)

    return grad_estimate


@typecheck
def IWELBO(
    proposal: SampleDistribution,
    make_target: Callable[[Any], Target],
    N: Int,
) -> Callable[[PRNGKey, Arguments], GradientEstimate]:
    """
    Return a function that computes the gradient estimate of the IWELBO loss term.
    """

    def grad_estimate(
        key: PRNGKey,
        args: Arguments,
    ) -> GradientEstimate:
        # In the source language of ADEV.
        @expectation
        def _loss(*args):
            target = make_target(*args)
            guide = ImportanceK(target, proposal, N)
            w = guide.estimate_normalizing_constant(key, target)
            return -w

        return _loss.grad_estimate(key, args)

    return grad_estimate


@typecheck
def PWake(
    posterior_approx: SampleDistribution,
    make_target: Callable[[Any], Target],
) -> Callable[[PRNGKey, Arguments], GradientEstimate]:
    """
    Return a function that computes the gradient estimate of the PWake loss term.
    """

    def grad_estimate(
        key: PRNGKey,
        args: Tuple,
    ) -> Tuple:
        key, sub_key1, sub_key2 = jax.random.split(key, 3)

        # In the source language of ADEV.
        @expectation
        def _loss(*target_args):
            target = make_target(*target_args)
            _, sample = posterior_approx.random_weighted(sub_key1, target)
            tr, _ = target.importance(sub_key2, sample)
            return -tr.get_score()

        return _loss.grad_estimate(key, args)

    return grad_estimate


@typecheck
def QWake(
    proposal: SampleDistribution,
    posterior_approx: SampleDistribution,
    make_target: Callable[[Any], Target],
) -> Callable[[PRNGKey, Arguments], GradientEstimate]:
    """
    Return a function that computes the gradient estimate of the QWake loss term.
    """

    def grad_estimate(
        key: PRNGKey,
        args: Tuple,
    ) -> Tuple:
        key, sub_key1, sub_key2 = jax.random.split(key, 3)

        # In the source language of ADEV.
        @expectation
        def _loss(*target_args):
            target = make_target(*target_args)
            _, sample = posterior_approx.random_weighted(sub_key1, target)
            w = proposal.estimate_logpdf(sub_key2, sample, target)
            return -w

        return _loss.grad_estimate(key, args)

    return grad_estimate
