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

from genjax._src.core.generative import (
    ChoiceMap,
    GenerativeFunction,
    Selection,
    Weight,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    FloatArray,
    Generic,
    PRNGKey,
    TypeVar,
    typecheck,
)
from genjax._src.generative_functions.distributions.distribution import Distribution

R = TypeVar("R")

####################
# Posterior target #
####################


@Pytree.dataclass
class Target(Generic[R], Pytree):
    """
    A `Target` represents an unnormalized target distribution induced by conditioning a generative function on a [`genjax.Constraint`][].

    Targets are created by providing a generative function, arguments to the generative function, and a constraint.

    Examples:
        Creating a target from a generative function, by providing arguments and a constraint:
        ```python exec="yes" html="true" source="material-block" session="core"
        import genjax
        from genjax import ChoiceMapBuilder as C
        from genjax.inference import Target


        @genjax.gen
        def model():
            x = genjax.normal(0.0, 1.0) @ "x"
            y = genjax.normal(x, 1.0) @ "y"
            return x


        target = Target(model, (), C["y"].set(3.0))
        print(target.render_html())
        ```
    """

    p: GenerativeFunction[R]
    args: tuple[Any, ...]
    constraint: ChoiceMap

    def importance(self, key: PRNGKey, constraint: ChoiceMap):
        merged = self.constraint.merge(constraint)
        return self.p.importance(key, merged, self.args)

    def filter_to_unconstrained(self, choice_map):
        selection = ~self.constraint.get_selection()
        return choice_map.filter(selection)

    def __getitem__(self, addr):
        return self.constraint[addr]


#######################
# Sample distribution #
#######################


@Pytree.dataclass
class SampleDistribution(Generic[R], Distribution[R]):
    """
    The abstract class `SampleDistribution` represents the type of distributions whose return value type is a `Sample`. This is the abstract base class of `Algorithm`, as well as `Marginal`.
    """

    @abstractmethod
    def random_weighted(
        self,
        key: PRNGKey,
        *args: Any,
    ) -> tuple[FloatArray, ChoiceMap]:
        raise NotImplementedError

    @abstractmethod
    def estimate_logpdf(
        self,
        key: PRNGKey,
        v: ChoiceMap,
        *args: Any,
    ) -> FloatArray:
        raise NotImplementedError


########################
# Inference algorithms #
########################


class Algorithm(SampleDistribution):
    """`Algorithm` is the type of inference
    algorithms: probabilistic programs which provide interfaces for sampling from
    posterior approximations, and estimating densities.

    **The stochastic probability interface for `Algorithm`**

    Inference algorithms implement the stochastic probability interface:

    * `Algorithm.random_weighted` exposes sampling from the approximation
    which the algorithm represents: it accepts a `Target` as input, representing the
    unnormalized distribution, and returns a sample from an approximation to
    the normalized distribution, along with a density estimate of the normalized distribution.

    * `Algorithm.estimate_logpdf` exposes density estimation for the
    approximation which `Algorithm.random_weighted` samples from:
    it accepts a value on the support of the approximation, and the `Target` which
    induced the approximation as input, and returns an estimate of the density of
    the approximation.

    **Optional methods for gradient estimators**

    Subclasses of type `Algorithm` can also implement two optional methods
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
        *args: Target,
    ) -> tuple[Weight, ChoiceMap]:
        """
        Given a [`Target`][genjax.inference.Target], return a [`Sample`][genjax.core.Sample] from an approximation to the normalized distribution of the target, and a random [`Weight`][genjax.core.Weight] estimate of the normalized density of the target at the sample.

        The `sample` is a sample on the support of `target.gen_fn` which _are not in_ `target.constraints`, produced by running the inference algorithm.

        Let $T_P(a, c)$ denote the target, with $P$ the distribution on samples represented by `target.gen_fn`, and $S$ denote the sample. Let $w$ denote the weight `w`. The weight $w$ is a random weight such that $w$ satisfies:

        $$
        \\mathbb{E}\\big[\\frac{1}{w} \\mid S \\big] = \\frac{1}{P(S \\mid c; a)}
        $$

        This interface corresponds to **(Defn 3.2) Unbiased Density Sampler** in [[Lew23](https://dl.acm.org/doi/pdf/10.1145/3591290)].
        """
        pass

    @abstractmethod
    def estimate_logpdf(self, key: PRNGKey, v: ChoiceMap, *args: Any) -> Weight:
        """
        Given a [`Sample`][genjax.core.Sample] and a [`Target`][genjax.inference.Target], return a random [`Weight`][genjax.core.Weight] estimate of the normalized density of the target at the sample.

        Let $T_P(a, c)$ denote the target, with $P$ the distribution on samples represented by `target.gen_fn`, and $S$ denote the sample. Let $w$ denote the weight `w`. The weight $w$ is a random weight such that $w$ satisfies:

        $$
        \\mathbb{E}[w] = P(S \\mid c, a)
        $$

        This interface corresponds to **(Defn 3.1) Positive Unbiased Density Estimator** in [[Lew23](https://dl.acm.org/doi/pdf/10.1145/3591290)].
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
    ) -> Weight:
        pass

    @abstractmethod
    def estimate_reciprocal_normalizing_constant(
        self,
        key: PRNGKey,
        target: Target,
        latent_choices: ChoiceMap,
        w: Weight,
    ) -> Weight:
        pass


############
# Marginal #
############


@Pytree.dataclass
@typecheck
class Marginal(Generic[R], SampleDistribution[R]):
    """The `Marginal` class represents the marginal distribution of a generative function over
    a selection of addresses. The return value type is a subtype of `Sample`.
    """

    gen_fn: GenerativeFunction[R]
    selection: Selection = Pytree.field(default=Selection.all())
    algorithm: Algorithm | None = Pytree.field(default=None)

    @typecheck
    def random_weighted(
        self,
        key: PRNGKey,
        *args,
    ) -> tuple[FloatArray, ChoiceMap]:
        key, sub_key = jax.random.split(key)
        tr = self.gen_fn.simulate(sub_key, args)
        choices: ChoiceMap = tr.get_choices()
        latent_choices = choices.filter(self.selection)
        key, sub_key = jax.random.split(key)
        bwd_problem = ~self.selection
        weight = tr.project(sub_key, bwd_problem)
        if self.algorithm is None:
            return weight, latent_choices
        else:
            target = Target(self.gen_fn, args, latent_choices)
            other_choices = choices.filter(~self.selection)
            Z = self.algorithm.estimate_reciprocal_normalizing_constant(
                key, target, other_choices, weight
            )

            return (Z, latent_choices)

    @typecheck
    def estimate_logpdf(
        self,
        key: PRNGKey,
        v: ChoiceMap,
        *args,
    ) -> Weight:
        if self.algorithm is None:
            _, weight = self.gen_fn.importance(key, v, args)
            return weight
        else:
            target = Target(self.gen_fn, args, v)
            Z = self.algorithm.estimate_normalizing_constant(key, target)
            return Z


################################
# Inference construct language #
################################


@typecheck
def marginal(
    selection: Selection = Selection.all(),
    algorithm: Algorithm | None = None,
) -> Callable[[GenerativeFunction[R]], Marginal[R]]:
    def decorator(
        gen_fn: GenerativeFunction[R],
    ) -> Marginal[R]:
        return Marginal(
            gen_fn,
            selection,
            algorithm,
        )

    return decorator
