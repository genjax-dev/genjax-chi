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

from abc import abstractmethod

import jax

from genjax._src.core.datatypes.generative import (
    Choice,
    ChoiceMap,
    ChoiceValue,
    GenerativeFunction,
    Selection,
)
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    FloatArray,
    PRNGKey,
    Tuple,
    typecheck,
)
from genjax._src.generative_functions.distributions.distribution import Distribution

#######################
# Choice distribution #
#######################


class ChoiceDistribution(Distribution):
    @abstractmethod
    def random_weighted(
        self,
        key: PRNGKey,
        *args,
    ) -> Tuple[FloatArray, Choice]:
        raise NotImplementedError

    @abstractmethod
    def estimate_logpdf(
        self,
        key: PRNGKey,
        latent_choices: Choice,
        *args,
    ) -> FloatArray:
        raise NotImplementedError


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


########################
# Inference algorithms #
########################


class InferenceAlgorithm(Pytree):
    """
    The class `InferenceAlgorithm` represents the type of inference algorithms, programs which implement interfaces for sampling from approximate posterior representations, and estimating the density of the approximate posterior.

    `InferenceAlgorithm` implementors can also implement two optional methods designed to support effective gradient estimators for variational objectives (`estimate_normalizing_constant` and `estimate_recip_normalizing_constant`).
    """

    @abstractmethod
    def random_weighted(
        self,
        key: PRNGKey,
        target: Target,
    ):
        pass

    @abstractmethod
    def estimate_logpdf(
        self,
        key: PRNGKey,
        target: Target,
    ):
        pass

    @abstractmethod
    def estimate_normalizing_constant(
        self,
        key: PRNGKey,
        target: Target,
    ):
        pass

    @abstractmethod
    def estimate_recip_normalizing_constant(
        self,
        key: PRNGKey,
        target: Target,
        latent_choices: ChoiceMap,
        w: FloatArray,
    ):
        pass


############
# Marginal #
############


class Marginal(ChoiceDistribution):
    selection: Selection
    p: GenerativeFunction
    q: Callable[[Any, ...], InferenceAlgorithm] = Pytree.static()  # type: ignore

    @typecheck
    def random_weighted(
        self,
        key: PRNGKey,
        p_args: Tuple,
        q_args: Tuple,
    ) -> Any:
        key, sub_key = jax.random.split(key)
        tr = self.p.simulate(sub_key, p_args)
        weight = tr.get_score()
        choices = tr.get_choices()
        latent_choices = choices.filter(self.selection)
        other_choices = choices.filter(self.selection.complement())
        target = Target(self.p, p_args, latent_choices)
        alg = self.q(*q_args)
        Z = alg.estimate_recip_normalizing_constant(key, target, other_choices, weight)
        return (Z, ChoiceValue(latent_choices))

    @typecheck
    def estimate_logpdf(
        self,
        key: PRNGKey,
        latent_choices: Choice,
        p_args: Tuple,
        q_args: Tuple,
    ) -> FloatArray:
        target = Target(self.p, p_args, latent_choices)
        alg = self.q(*q_args)
        Z = alg.estimate_normalizing_constant(key, target)
        return Z
