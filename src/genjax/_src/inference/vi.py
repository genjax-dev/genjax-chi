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

from abc import abstractmethod

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from genjax._src.adev.core import (
    ADEVPrimitive,
    expectation,
    reap_key,
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
from genjax._src.core.datatypes.generative import JAXGenerativeFunction
from genjax._src.core.pytree import Pytree
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
)
from genjax._src.inference.core import ChoiceDistribution, InferenceAlgorithm, Target
from genjax._src.inference.smc import Importance, ImportanceK

tfd = tfp.distributions


##########################################
# Differentiable distribution primitives #
##########################################


class ADEVDistribution(JAXGenerativeFunction, ExactDensity):
    """The class `ADEVDistribution` is a wrapper class which exposes the `sample` and
    `logpdf` interfaces, where `sample` utilizes an ADEV differentiable sampling
    primitive, and `logpdf` is a differentiable logpdf function."""

    adev_primitive: ADEVPrimitive
    differentiable_logpdf: Callable = Pytree.static()

    def sample(
        self,
        key: PRNGKey,
        *args: Any,
    ) -> Any:
        return sample_primitive(self.adev_primitive, key, *args)

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


##############
# Loss terms #
##############


class ExpectedValueLoss(Pytree):
    """Base class for expected value loss functions.

    Exposes a `grad_estimate` interface, which takes a PRNGKey and a tuple of arguments (which are allowed to be `Pytree` instances), and returns a tuple of gradient estimates (a tuple, with values which are the same shape as the primal `Pytree` instances).
    """

    @abstractmethod
    def grad_estimate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> Tuple:
        pass


class ELBO(ExpectedValueLoss):
    target: Target
    make_proposal: Callable[[Any], ChoiceDistribution] = Pytree.static()

    def grad_estimate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> Tuple:
        # In the source language of ADEV.
        @expectation
        def _loss(proposal_args):
            proposal = self.make_proposal(proposal_args)
            guide = Importance(self.target, proposal)
            key = reap_key()
            w = guide.estimate_normalizing_constant(key, self.target)
            return w

        return _loss.grad_estimate(key, args)


class IWELBO(ExpectedValueLoss):
    target: Target
    make_proposal: Callable[[Any], ChoiceDistribution] = Pytree.static()
    N: Int = Pytree.static()

    def grad_estimate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> Tuple:
        # In the source language of ADEV.
        @expectation
        def _loss(proposal_args):
            proposal = self.make_proposal(proposal_args)
            guide = ImportanceK(self.target, proposal, self.N)
            key = reap_key()
            w = guide.estimate_normalizing_constant(key, self.target)
            return w

        return _loss.grad_estimate(key, args)


class QWake(Pytree):
    target: Target
    approx: InferenceAlgorithm
    make_proposal: Callable[[Any], ChoiceDistribution] = Pytree.static()

    def grad_estimate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> Tuple:
        # In the source language of ADEV.
        @expectation
        def _loss(*q_args):
            key = reap_key()
            _, v = self.approx.random_weighted(key, self.target)
            proposal = self.make_proposal(*q_args)
            key = reap_key()
            w = proposal.estimate_logpdf(key, v)
            return -w

        return _loss.grad_estimate(key, args)
