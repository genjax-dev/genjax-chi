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
from equinox import module_update_wrapper

from genjax._src.core.datatypes.generative import (
    AllSelection,
    Choice,
    GenerativeFunction,
    JAXGenerativeFunction,
    Selection,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    FloatArray,
    Optional,
    PRNGKey,
    Tuple,
    Union,
    typecheck,
)
from genjax._src.generative_functions.distributions.distribution import Distribution
from genjax._src.shortcuts import choice_map, select

####################
# Posterior target #
####################


class Target(Pytree):
    """
    Instances of `Target` represent unnormalized target distributions. A `Target` is created by pairing a generative function and its arguments with a `Choice` object, which represents constraints applied to the generative function.
    """

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


#######################
# Choice distribution #
#######################


class ChoiceDistribution(Distribution):
    """
    The abstract class `ChoiceDistribution` represents the type of distributions whose return value type is a `Choice`. This is the abstract base class of `InferenceAlgorithm`, as well as `Marginal`.
    """

    @abstractmethod
    def random_weighted(
        self,
        key: PRNGKey,
        *args: Any,
    ) -> Tuple[FloatArray, Choice]:
        raise NotImplementedError

    @abstractmethod
    def estimate_logpdf(
        self,
        key: PRNGKey,
        latent_choices: Choice,
        *args: Any,
    ) -> FloatArray:
        raise NotImplementedError


########################
# Inference algorithms #
########################


class InferenceAlgorithm(ChoiceDistribution, JAXGenerativeFunction):
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
        """
        Given a `key: PRNGKey`, and a `target: Target`, returns a pair `(w, choice)` where `w` is a sample from an estimator with $\mathbb{E}[w] = 1 / Z$ and `choice: Choice` is a sample from the approximate posterior which `self: InferenceAlgorithm` represents.
        """
        pass

    @abstractmethod
    def estimate_logpdf(
        self,
        key: PRNGKey,
        latent_choices: Choice,
        target: Target,
    ) -> FloatArray:
        """
        Given a `key: PRNGKey`, `latent_choices: Choice` and a `target: Target`, returns `w` where `w` is a sample from an estimator with $\mathbb{E}[w] = Z$.
        """
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


############
# Marginal #
############


@typecheck
class Marginal(ChoiceDistribution):
    """The `Marginal` class represents the marginal distribution of a generative function over
    a selection of addresses. The `Marginal` class implements
    the stochastic probability interface by utilizing an optional `InferenceAlgorithm`, which can be specified
    by providing an `algorithm_builder: Target -> InferenceAlgorithm` function.

    When provided with a `Selection`, `Marginal` acts as a distribution
    which returns a `Choice` object, representing a structured sample of choices.
    """

    p: GenerativeFunction
    selection: Selection = Pytree.field(default=AllSelection())
    alg: Optional[InferenceAlgorithm] = Pytree.field(default=None)

    @typecheck
    def random_weighted(
        self,
        key: PRNGKey,
        *args: Any,
    ) -> Tuple[FloatArray, Choice]:
        key, sub_key = jax.random.split(key)
        tr = self.p.simulate(sub_key, args)
        weight = tr.get_score()
        choices = tr.get_choices()
        latent_choices = choices.filter(self.selection)
        other_choices = choices.filter(self.selection.complement())
        target = Target(self.p, args, latent_choices)
        if self.alg is None:
            return weight, latent_choices
        else:
            Z = self.alg.estimate_reciprocal_normalizing_constant(
                key, target, other_choices, weight
            )

            return (Z, latent_choices)

    @typecheck
    def estimate_logpdf(
        self,
        key: PRNGKey,
        latent_choices: Choice,
        *args: Any,
    ) -> FloatArray:
        if self.alg is None:
            _, weight = self.p.importance(key, latent_choices, args)
            return weight
        else:
            target = Target(self.p, args, latent_choices)
            Z = self.alg.estimate_normalizing_constant(key, target)
            return Z

    @property
    def __wrapped__(self):
        return self.p


@typecheck
class ValueMarginal(Distribution):
    """The `ValueMarginal` class represents the marginal distribution of a generative function over
    a single address. The `ValueMarginal` class implements
    the stochastic probability interface by utilizing an optional `InferenceAlgorithm`, which can be specified
    by providing an `algorithm_builder: Target -> InferenceAlgorithm` function.

    While similar to `Marginal`, `ValueMarginal` operates in "value" mode, meaning that
    the `random_weighted` method returns the value at the address. This allows `ValueMarginal` in this mode
    to be used inside of generative functions which support callees (for example, `StaticGenerativeFunction`).
    """

    p: GenerativeFunction
    addr: Any
    alg: Optional[InferenceAlgorithm] = Pytree.field(default=None)

    @typecheck
    def random_weighted(
        self,
        key: PRNGKey,
        *args: Any,
    ) -> Tuple[FloatArray, Any]:
        marginal = Marginal(self.p, select(self.addr), self.alg)
        Z, choice = marginal.random_weighted(key, *args)
        return Z, choice[self.addr]

    @typecheck
    def estimate_logpdf(
        self,
        key: PRNGKey,
        v: Any,
        *args: Any,
    ) -> FloatArray:
        marginal = Marginal(self.p, select(self.addr), self.alg)
        latent_choice = choice_map({self.addr: v})
        return marginal.estimate_logpdf(key, latent_choice, *args)

    @property
    def __wrapped__(self):
        return self.p


################################
# Inference construct language #
################################


@typecheck
def marginal(
    selection_or_addr: Union[Selection, Any] = AllSelection(),
    alg: Optional[InferenceAlgorithm] = None,
):
    def decorator(gen_fn: GenerativeFunction) -> Union[Marginal, ValueMarginal]:
        if isinstance(selection_or_addr, Selection):
            marginal = Marginal(
                gen_fn,
                selection_or_addr,
                alg,
            )
        else:
            marginal = ValueMarginal(
                gen_fn,
                selection_or_addr,
                alg,
            )
        return module_update_wrapper(marginal)

    return decorator
