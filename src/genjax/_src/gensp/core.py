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

import abc

import jax

from genjax._src.core.datatypes.generative import (
    ChoiceMap,
    ChoiceValue,
    EmptyChoice,
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
    dispatch,
    typecheck,
)
from genjax._src.generative_functions.distributions.distribution import Distribution

##################
# SPDistribution #
##################

# The interface is the same as `genjax.Distribution`, but we create a separate type
# so that we can use the type in static checking, dispatch, etc.


class SPDistribution(Distribution):
    @abc.abstractmethod
    def random_weighted(key: PRNGKey, *args) -> Tuple[FloatArray, ChoiceValue]:
        pass

    @abc.abstractmethod
    def estimate_logpdf(key: PRNGKey, v, *args) -> FloatArray:
        pass


####################
# Posterior target #
####################


class Target(Pytree):
    p: GenerativeFunction
    args: Tuple
    constraints: ChoiceMap

    def latent_selection(self):
        return self.constraints.get_selection().complement()

    def get_latents(self, v):
        latent_selection = self.latent_selection()
        latents = v.strip().filter(latent_selection)
        return latents

    @dispatch
    def importance(self, key: PRNGKey, chm: ChoiceValue):
        inner = chm.get_value()
        assert isinstance(inner, ChoiceMap)
        merged = self.constraints.safe_merge(inner)
        (tr, _) = self.p.importance(key, merged, self.args)
        return (0.0, tr)

    @dispatch
    def importance(self, key: PRNGKey):
        (tr, _) = self.p.importance(key, self.constraints, self.args)
        return (0.0, tr)


@dispatch
def target(
    p: GenerativeFunction,
    args: Tuple,
):
    return Target.new(p, args, EmptyChoice())


@dispatch
def target(
    p: GenerativeFunction,
    args: Tuple,
    constraints: ChoiceMap,
):
    return Target.new(p, args, constraints)


###############
# SPAlgorithm #
###############


class SPAlgorithm(Pytree):
    @abc.abstractmethod
    def random_weighted(self, key: PRNGKey, target: Target):
        pass

    @abc.abstractmethod
    def estimate_logpdf(
        self,
        key: PRNGKey,
        target: Target,
    ):
        pass

    @abc.abstractmethod
    def estimate_normalizing_constant(
        self,
        key: PRNGKey,
        target: Target,
    ):
        pass

    @abc.abstractmethod
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


class Marginal(SPDistribution):
    q: Callable[[Any, ...], SPAlgorithm] = Pytree.static()  # type: ignore
    selection: Selection
    p: GenerativeFunction

    @typecheck
    def random_weighted(
        self,
        key: PRNGKey,
        *args,
    ) -> Any:
        key, sub_key = jax.random.split(key)
        p_args, q_args = args
        tr = self.p.simulate(sub_key, p_args)
        weight = tr.get_score()
        choices = tr.get_choices()
        latent_choices = choices.filter(self.selection)
        other_choices = choices.filter(self.selection.complement())
        tgt = target(self.p, p_args, latent_choices)
        alg = self.q(*q_args)
        Z = alg.estimate_recip_normalizing_constant(key, tgt, other_choices, weight)
        return (Z, ChoiceValue(latent_choices))

    @typecheck
    def estimate_logpdf(
        self,
        key: PRNGKey,
        latent_choices: ChoiceValue,
        *args,
    ) -> FloatArray:
        inner_choices = latent_choices.get_value()
        (p_args, q_args) = args
        tgt = target(self.p, p_args, inner_choices)
        alg = self.q(*q_args)
        Z = alg.estimate_normalizing_constant(key, tgt)
        return Z


@dispatch
def marginal(
    selection: Selection,
    p: GenerativeFunction,
    q: Callable[[Any, ...], SPAlgorithm],  # type: ignore
):
    return Marginal.new(q, selection, p)
