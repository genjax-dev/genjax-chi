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
from genjax._src.inference.core import ChoiceDistribution, Target
from genjax._src.inference.smc import Importance, ImportanceK

tfd = tfp.distributions


##########################################
# Differentiable distribution primitives #
##########################################


class ADEVDistribution(JAXGenerativeFunction, ExactDensity):
    """The class `ADEVDistribution` is a distribution wrapper class which exposes `sample` and
    `logpdf` interfaces, where `sample` is expected to utilize an ADEV differentiable sampling
    primitive, and `logpdf` is a differentiable logpdf function.

    Given a `prim: ADEVPrimitive`, a user can readily construct an `ADEVDistribution`:

    ```python exec="yes" source="tabbed-left"
    import genjax
    from genjax.adev import flip_enum
    from genjax.inference.vi import ADEVDistribution

    console = genjax.console()

    flip_enum = ADEVDistribution(
        flip_enum, lambda v, p: tfd.Bernoulli(probs=p).log_prob(v)
    )
    print(console.render(flip_enum))
    ```

    These objects can then be utilized in guide programs, and support unbiased gradient estimator automation via ADEV's gradient transformations.
    ```
    """

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
    normal.logpdf,
)

normal_reparam = ADEVDistribution(
    normal_reparam,
    normal.logpdf,
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
    guide: ChoiceDistribution
    make_target: Callable[[Any], Target] = Pytree.static()

    def grad_estimate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> Tuple:
        # In the source language of ADEV.
        @expectation
        def _loss(*target_args):
            target = self.make_target(*target_args)
            guide = Importance(target, self.guide)
            key = reap_key()
            w = guide.estimate_normalizing_constant(key, target)
            return -w

        return _loss.grad_estimate(key, args)


class IWELBO(ExpectedValueLoss):
    proposal: ChoiceDistribution
    make_target: Callable[[Any], Target] = Pytree.static()
    N: Int = Pytree.static()

    def grad_estimate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> Tuple:
        # In the source language of ADEV.
        @expectation
        def _loss(*target_args):
            target = self.make_target(*target_args)
            guide = ImportanceK(target, self.proposal, self.N)
            key = reap_key()
            w = guide.estimate_normalizing_constant(key, target)
            return -w

        return _loss.grad_estimate(key, args)
