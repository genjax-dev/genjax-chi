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
)
from genjax._src.generative_functions.distributions.tensorflow_probability import (
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
    differentiable_logpdf: Callable,
):
    def sampler(key: PRNGKey, *args: Any) -> Any:
        return sample_primitive(adev_primitive, key, *args)

    def logpdf(v: Any, *args: Any) -> FloatArray:
        lp = differentiable_logpdf(v, *args)
        # Branching here is statically resolved.
        if lp.shape:
            return jnp.sum(lp)
        else:
            return lp

    return ExactDensity(sampler, logpdf)


# We import ADEV specific sampling primitives, but then wrap them in
# adev_distribution, for usage inside of generative functions.
flip_enum = adev_distribution(
    flip_enum,
    lambda v, p: tfd.Bernoulli(probs=p).log_prob(v),
)

flip_mvd = adev_distribution(
    flip_mvd,
    lambda v, p: tfd.Bernoulli(probs=p).log_prob(v),
)

categorical_enum = adev_distribution(
    categorical_enum_parallel,
    lambda v, probs: tfd.Categorical(probs=probs).log_prob(v),
)

normal_reinforce = adev_distribution(
    normal_reinforce,
    lambda *args: normal(*args).logpdf,
)

normal_reparam = adev_distribution(
    normal_reparam,
    lambda *args: normal(*args).logpdf,
)

mv_normal_diag_reparam = adev_distribution(
    mv_normal_diag_reparam,
    lambda v, loc, scale_diag: tfd.MultivariateNormalDiag(
        loc=loc, scale_diag=scale_diag
    ).log_prob(v),
)

geometric_reinforce = adev_distribution(
    geometric_reinforce,
    lambda v, *args: geometric.logpdf(v, *args),
)


##############
# Loss terms #
##############


def ELBO(
    guide: SampleDistribution,
    make_target: Callable[[Any], Target],
):
    def grad_estimate(
        key: PRNGKey,
        args: Tuple,
    ) -> Tuple:
        # In the source language of ADEV.
        @expectation
        def _loss(*target_args):
            target = make_target(*target_args)
            guide_alg = Importance(target, guide)
            w = guide_alg.estimate_normalizing_constant(key, target)
            return -w

        return _loss.grad_estimate(key, args)

    return grad_estimate


def IWELBO(
    proposal: SampleDistribution,
    make_target: Callable[[Any], Target],
    N: Int,
):
    def grad_estimate(
        key: PRNGKey,
        args: Tuple,
    ) -> Tuple:
        # In the source language of ADEV.
        @expectation
        def _loss(*target_args):
            target = make_target(*target_args)
            guide = ImportanceK(target, proposal, N)
            w = guide.estimate_normalizing_constant(key, target)
            return -w

        return _loss.grad_estimate(key, args)

    return grad_estimate


def PWake(
    posterior_approx: SampleDistribution,
    make_target: Callable[[Any], Target],
):
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


def QWake(
    proposal: SampleDistribution,
    posterior_approx: SampleDistribution,
    make_target: Callable[[Any], Target],
):
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
