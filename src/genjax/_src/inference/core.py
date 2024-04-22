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

from genjax._src.core.generative import (
    ChoiceMap,
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

####################
# Posterior target #
####################


class Target(Pytree):
    """
    Instances of `Target` represent unnormalized target distributions. A `Target` is created by pairing a generative function and its arguments with a `ChoiceMap` object.
    The target represents the unnormalized distribution on the unconstrained choices in the generative function, fixing the constraints.
    """

    p: GenerativeFunction
    args: Tuple
    constraints: ChoiceMap

    def filter_to_unconstrained(self, choice: ChoiceMap):
        constraint_selection = self.constraints.get_selection()
        complement = constraint_selection.complement()
        return choice.filter(complement)

    def importance(self, key: PRNGKey, choice: ChoiceMap):
        """
        Uses `target.p.importance` to sample an importance particle consistent with both
        `target.constraints` and `choice`.
        """
        merged = self.constraints.safe_merge(choice)
        return self.p.importance(key, merged, self.args)

    def __getitem__(self, v):
        return self.constraints[v]


#######################
# ChoiceMap distribution #
#######################


class ChoiceMapDistribution(Distribution):
    """
    The abstract class `ChoiceMapDistribution` represents the type of distributions whose return value type is a `ChoiceMap`. This is the abstract base class of `InferenceAlgorithm`, as well as `Marginal`.
    """

    @abstractmethod
    def random_weighted(
        self,
        key: PRNGKey,
        *args: Any,
    ) -> Tuple[FloatArray, ChoiceMap]:
        raise NotImplementedError

    @abstractmethod
    def estimate_logpdf(
        self,
        key: PRNGKey,
        latent_choices: ChoiceMap,
        *args: Any,
    ) -> FloatArray:
        raise NotImplementedError


########################
# Inference algorithms #
########################


class InferenceAlgorithm(ChoiceMapDistribution, JAXGenerativeFunction):
    """The abstract class `InferenceAlgorithm` represents the type of inference
    algorithms, programs which implement interfaces for sampling from approximate
    posterior representations, and estimating the density of the approximate posterior.

    An InferenceAlgorithm is a genjax `Distribution`.
    It accepts a `Target` as input, representing the unnormalized
    distribution $R$, and samples from an approximation to
    the normalized distribution $R / R(X)$,
    where $X$ is the space of all choicemaps.
    The `InferenceAlgorithm` object is semantically equivalent as a genjax `Distribution`
    to the normalized distribution $R / R(X)$, in the sense
    defined by the stochastic probability interface.  (See `Distribution`.)

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
    ) -> Tuple[FloatArray, ChoiceMap]:
        """
        Given a `key: PRNGKey`, and a `target: Target`, returns a pair `(log_w, choice)`.
        `choice : ChoiceMap` is a choicemap on the addresses sampled at in `target.gen_fn` not in `target.constraints`;
        it is sampled by running the inference algorithm represented by `self`.
        `log_w` is a random weight such that $w = \exp(\texttt{log_w})$ satisfies
        $\mathbb{E}[1 / w \mid \texttt{choice}] = 1 / P(\texttt{choice} \mid \texttt{target.constraints})`, where `P` is the
        distribution on choicemaps represented by `target.gen_fn`.
        """
        pass

    @abstractmethod
    def estimate_logpdf(
        self,
        key: PRNGKey,
        latent_choices: ChoiceMap,
        target: Target,
    ) -> FloatArray:
        """
        Given a `key: PRNGKey`, `latent_choices: ChoiceMap` and a `target: Target`, returns a random value $\log(w)$
        such that $\mathbb{E}[w] = P(\texttt{latent_choices} \mid \texttt{target.constraints})$, where $P$
        is the distribution on choicemaps represented by `target.gen_fn`.
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
        latent_choices: ChoiceMap,
        w: FloatArray,
    ) -> FloatArray:
        pass


############
# Marginal #
############


@typecheck
class Marginal(ChoiceMapDistribution):
    """The `Marginal` class represents the marginal distribution of a generative function over
    a selection of addresses. The return value type is a subtype of `ChoiceMap`.
    """

    p: GenerativeFunction
    selection: Selection = Pytree.field(default=Selection.a)
    algorithm: Optional[InferenceAlgorithm] = Pytree.field(default=None)

    @typecheck
    def random_weighted(
        self,
        key: PRNGKey,
        *args: Any,
    ) -> Tuple[FloatArray, ChoiceMap]:
        key, sub_key = jax.random.split(key)
        tr = self.p.simulate(sub_key, args)
        choices = tr.get_choices()
        latent_choices = choices.filter(self.selection)
        other_choices = choices.filter(~self.selection)
        target = Target(self.p, args, latent_choices)
        key, sub_key = jax.random.split(key)
        weight = tr.project(sub_key, self.selection)
        if self.algorithm is None:
            return weight, latent_choices
        else:
            Z = self.algorithm.estimate_reciprocal_normalizing_constant(
                key, target, other_choices, weight
            )

            return (Z, latent_choices)

    @typecheck
    def estimate_logpdf(
        self,
        key: PRNGKey,
        latent_choices: ChoiceMap,
        *args: Any,
    ) -> FloatArray:
        if self.algorithm is None:
            _, weight = self.p.importance(key, latent_choices, args)
            return weight
        else:
            target = Target(self.p, args, latent_choices)
            Z = self.algorithm.estimate_normalizing_constant(key, target)
            return Z

    @property
    def __wrapped__(self):
        return self.p


@typecheck
class ValueMarginal(Distribution):
    """The `ValueMarginal` class represents the marginal distribution of a generative function over
    a single address `addr: Any`. The return value type is the type of the value at that address.
    """

    p: GenerativeFunction
    addr: Any
    algorithm: Optional[InferenceAlgorithm] = Pytree.field(default=None)

    @typecheck
    def random_weighted(
        self,
        key: PRNGKey,
        *args: Any,
    ) -> Tuple[FloatArray, Any]:
        marginal = Marginal(self.p, select(self.addr), self.algorithm)
        Z, choice = marginal.random_weighted(key, *args)
        return Z, choice[self.addr]

    @typecheck
    def estimate_logpdf(
        self,
        key: PRNGKey,
        v: Any,
        *args: Any,
    ) -> FloatArray:
        marginal = Marginal(
            self.p,
            Selection.at[self.addr],
            self.algorithm,
        )
        latent_choice = ChoiceMap.a(self.addr, v)
        return marginal.estimate_logpdf(key, latent_choice, *args)

    @property
    def __wrapped__(self):
        return self.p


################################
# Inference construct language #
################################


@typecheck
def marginal(
    gen_fn: Optional[GenerativeFunction] = None,
    *,
    select_or_addr: Union[Selection, Any] = Selection.a,
    algorithm: Optional[InferenceAlgorithm] = None,
):
    """If `select_or_addr` is a `Selection`, this constructs a `Marginal` distribution
    which samples `ChoiceMap` objects with addresses given in the selection.
    If `select_or_addr` is an address, this constructs a `ValueMarginal` distribution
    which samples values of the type stored at the given address in `gen_fn`.
    """

    def decorator(gen_fn: GenerativeFunction) -> Union[Marginal, ValueMarginal]:
        if isinstance(select_or_addr, Selection):
            marginal = Marginal(
                gen_fn,
                select_or_addr,
                algorithm,
            )
        else:
            marginal = ValueMarginal(
                gen_fn,
                select_or_addr,
                algorithm,
            )
        return module_update_wrapper(marginal)

    if gen_fn is not None:
        return decorator(gen_fn)
    else:
        return decorator
