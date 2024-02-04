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

import jax

from genjax._src.core.datatypes.generative import (
    AllSelection,
    Choice,
    GenerativeFunction,
    JAXGenerativeFunction,
    Selection,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Callable,
    FloatArray,
    Optional,
    PRNGKey,
    Tuple,
    typecheck,
)
from genjax._src.generative_functions.distributions.distribution import Distribution

####################
# Posterior target #
####################


class Target(Pytree):
    p: GenerativeFunction
    args: Tuple
    constraints: Choice

    def project(self, choice: Choice):
        constraint_selection = self.constraints.get_selection()
        complement = constraint_selection.complement()
        return choice.filter(complement)

    def generate(self, key: PRNGKey, choice: Choice):
        merged = self.constraints.safe_merge(choice)
        (tr, _) = self.p.importance(key, merged, self.args)
        return (tr.get_score(), tr)

    def __getitem__(self, v):
        return self.constraints[v]


########################
# Inference algorithms #
########################


class InferenceAlgorithm(Distribution, JAXGenerativeFunction):
    """The abstract class `InferenceAlgorithm` represents the type of inference
    algorithms, programs which implement interfaces for sampling from approximate
    posterior representations, and estimating the density of the approximate posterior.

    Subclasses of type `InferenceAlgorithm` can also implement two optional methods
    designed to support effective gradient estimators for variational objectives
    (`estimate_normalizing_constant` and `estimate_reciprocal_normalizing_constant`).
    """

    #########
    # GenSP #
    #########

    @abstractmethod
    def random_weighted(
        self,
        key: PRNGKey,
        target: Target,
    ) -> Tuple[FloatArray, Choice]:
        pass

    @abstractmethod
    def estimate_logpdf(
        self,
        key: PRNGKey,
        latent_choices: Choice,
        target: Target,
    ) -> FloatArray:
        pass

    ################
    # VI via GRASP #
    ################

    @abstractmethod
    def estimate_normalizing_constant(
        self,
        key: PRNGKey,
        target: Target,
    ) -> FloatArray:
        pass

    @abstractmethod
    def estimate_reciprocal_normalizing_constant(
        self,
        key: PRNGKey,
        target: Target,
        latent_choices: Choice,
        w: FloatArray,
    ) -> FloatArray:
        pass


#######################
# Choice distribution #
#######################


class ChoiceDistribution(Distribution):
    @abstractmethod
    def random_weighted(
        self,
        key: PRNGKey,
    ) -> Tuple[FloatArray, Choice]:
        raise NotImplementedError

    @abstractmethod
    def estimate_logpdf(
        self,
        key: PRNGKey,
        latent_choices: Choice,
    ) -> FloatArray:
        raise NotImplementedError


############
# Marginal #
############


@typecheck
class Marginal(ChoiceDistribution):
    p: GenerativeFunction
    p_args: Tuple
    selection: Selection = Pytree.field(default=AllSelection())
    algorithm_builder: Optional[Callable[[Target], InferenceAlgorithm]] = Pytree.static(
        default=None
    )

    @typecheck
    def random_weighted(
        self,
        key: PRNGKey,
    ) -> Tuple[FloatArray, Choice]:
        key, sub_key = jax.random.split(key)
        tr = self.p.simulate(sub_key, self.p_args)
        weight = tr.get_score()
        choices = tr.get_choice()
        latent_choices = choices.filter(self.selection)
        other_choices = choices.filter(self.selection.complement())
        target = Target(self.p, self.p_args, latent_choices)
        if self.algorithm_builder is None:
            return weight, latent_choices
        else:
            alg = self.algorithm_builder(target)
            Z = alg.estimate_reciprocal_normalizing_constant(
                key, target, other_choices, weight
            )

            return (Z, latent_choices)

    @typecheck
    def estimate_logpdf(
        self,
        key: PRNGKey,
        latent_choices: Choice,
    ) -> FloatArray:
        if self.algorithm_builder is None:
            _, weight = self.p.importance(key, latent_choices, self.p_args)
            return weight
        else:
            target = Target(self.p, self.p_args, latent_choices)
            alg = self.algorithm_builder(target)
            Z = alg.estimate_normalizing_constant(key, target)
            return Z
