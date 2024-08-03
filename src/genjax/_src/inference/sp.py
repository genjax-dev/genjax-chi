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
    Arguments,
    ChoiceMap,
    ChoiceMapConstraint,
    ChoiceMapSample,
    Constraint,
    EditRequest,
    GenerativeFunction,
    Projection,
    Retval,
    Sample,
    Score,
    Selection,
    Weight,
)
from genjax._src.core.generative.core import Trace
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    Generic,
    Optional,
    PRNGKey,
    TypeVar,
    tuple,
)
from genjax._src.generative_functions.distributions.distribution import Distribution

Tr = TypeVar("Tr", bound=Trace)
A = TypeVar("A", bound=Arguments)
R = TypeVar("R", bound=Retval)
S = TypeVar("S", bound=Sample)
C = TypeVar("C", bound=Constraint)
P = TypeVar("P", bound=Projection)
U = TypeVar("U", bound=EditRequest)


####################
# Posterior target #
####################


@Pytree.dataclass
class Target(Pytree):
    """A `Target` represents an unnormalized target distribution induced by
    conditioning a generative function on a [`genjax.Constraint`][].

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

    p: GenerativeFunction
    arguments: tuple
    constraint: ChoiceMapConstraint

    def importance(
        self,
        key: PRNGKey,
        constraint: ChoiceMap | ChoiceMapConstraint,
    ):
        forced_constraint: ChoiceMapConstraint = (
            ChoiceMapConstraint(constraint)
            if not isinstance(constraint, ChoiceMapConstraint)
            else constraint
        )
        merged = self.constraint.merge(forced_constraint)
        return self.p.importance(key, merged, self.arguments)

    def filter_to_unconstrained(self, choice_map):
        selection = ~self.constraint.get_selection()
        return choice_map.filter(selection)

    def __getitem__(self, addr):
        return self.constraint.choice_map[addr]


#######################
# Sample distribution #
#######################


@Pytree.dataclass
class SampleDistribution(Generic[A, S], Distribution[A, S]):
    """The abstract class `SampleDistribution` represents the type of
    distributions whose return value type is a `Sample`.

    This is the abstract
    base class of `Algorithm`, as well as `Marginal`.

    """

    @abstractmethod
    def random_weighted(
        self,
        key: PRNGKey,
        *arguments: Any,
    ) -> tuple[Score, S]:
        raise NotImplementedError

    @abstractmethod
    def estimate_logpdf(
        self,
        key: PRNGKey,
        v: S,
        *arguments: Any,
    ) -> Weight:
        raise NotImplementedError


########################
# Inference algorithms #
########################


class Algorithm(SampleDistribution):
    """`Algorithm` is the type of inference
    algorithms: probabilistic programs which provide interfaces for sampling
    from
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
        target: Target,
    ) -> tuple[Weight, Sample]:
        """Given a [`Target`][genjax.inference.Target], return a
        [`Sample`][genjax.core.Sample] from an approximation to the normalized
        distribution of the target, and a random [`Weight`][genjax.core.Weight]
        estimate of the normalized density of the target at the sample.

        The `sample` is a sample on the support of `target.gen_fn` which _are not in_ `target.constraints`, produced by running the inference algorithm.

        Let $T_P(a, c)$ denote the target, with $P$ the distribution on samples represented by `target.gen_fn`, and $S$ denote the sample. Let $w$ denote the weight `w`. The weight $w$ is a random weight such that $w$ satisfies:

        $$
        \\mathbb{E}\\big[\\frac{1}{w} \\mid S \\big] = \\frac{1}{P(S \\mid c; a)}
        $$

        This interface corresponds to **(Defn 3.2) Unbiased Density Sampler** in [[Lew23](https://dl.acm.org/doi/pdf/10.1145/3591290)].

        """
        pass

    @abstractmethod
    def estimate_logpdf(
        self,
        key: PRNGKey,
        sample: Sample,
        target: Target,
    ) -> Weight:
        """Given a [`Sample`][genjax.core.Sample] and a
        [`Target`][genjax.inference.Target], return a random
        [`Weight`][genjax.core.Weight] estimate of the normalized density of
        the target at the sample.

        Let $T_P(a, c)$ denote the target, with $P$ the distribution on samples represented by `target.gen_fn`, and $S$ denote the sample. Let $w$ denote the weight `w`. The weight $w$ is a random weight such that $w$ satisfies:

        $$
        \\mathbb{E}[w] = P(S \\mid c, a)
        $$

        This interface corresponds to **(Defn 3.1) Positive Unbiased Density Estimator** in [[Lew23](https://dl.acm.org/doi/pdf/10.1145/3591290)].

        """
        pass

    #####################
    # VI specialization #
    #####################

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
        latent_choices: Sample,
        w: Weight,
    ) -> Weight:
        pass


############
# Marginal #
############


@Pytree.dataclass
class MarginalTrace(Trace):
    gen_fn: "Marginal"
    inner: Trace
    score: Score

    def get_gen_fn(self) -> "Marginal":
        return self.gen_fn

    def get_args(self):
        return self.inner.get_args()

    def get_retval(self):
        return self.inner.get_retval()

    def get_score(self):
        return self.score

    def get_sample(self):
        return self.inner.get_sample()


@Pytree.dataclass
class Marginal(SampleDistribution):
    """The `Marginal` class represents the marginal distribution of a
    generative function over a selection of addresses.

    The return value type is a subtype of `Sample`.

    """

    gen_fn: GenerativeFunction
    selection: Selection = Pytree.field(default=Selection.all())
    algorithm: Optional[Algorithm] = Pytree.field(default=None)

    def random_weighted(
        self,
        key: PRNGKey,
        *args,
    ) -> tuple[Score, Sample]:
        key, sub_key = jax.random.split(key)
        tr = self.gen_fn.simulate(sub_key, args)
        choices: ChoiceMap = tr.get_choices()
        latent_choices = choices.filter(self.selection)
        key, sub_key = jax.random.split(key)
        bwd_problem = ~self.selection
        weight = tr.project(sub_key, bwd_problem)
        if self.algorithm is None:
            return weight, ChoiceMapSample(latent_choices)
        else:
            target = Target(self.gen_fn, args, latent_choices)
            other_choices = choices.filter(~self.selection)
            Z = self.algorithm.estimate_reciprocal_normalizing_constant(
                key, target, other_choices, weight
            )

            return (Z, ChoiceMapSample(latent_choices))

    def estimate_logpdf(
        self,
        key: PRNGKey,
        v: ChoiceMap,
        *args,
    ) -> Weight:
        if self.algorithm is None:
            _, weight = self.gen_fn.importance(key, constraint, args)
            return weight
        else:
            target = Target(self.gen_fn, args, constraint)
            Z = self.algorithm.estimate_normalizing_constant(key, target)
            return Z


################################
# Inference construct language #
################################


def marginal(
    selection: Optional[Selection] = Selection.all(),
    algorithm: Optional[Algorithm] = None,
) -> Callable[[GenerativeFunction], Marginal]:
    def decorator(
        gen_fn: GenerativeFunction,
    ) -> Marginal:
        return Marginal(
            gen_fn,
            selection,
            algorithm,
        )

    return decorator
