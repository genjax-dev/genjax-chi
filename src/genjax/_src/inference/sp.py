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
from jax.lax import cond

from genjax._src.core.generative import (
    GFI,
    ChoiceMap,
    Selection,
    Weight,
)
from genjax._src.core.generative.concepts import (
    Argdiffs,
    EditRequest,
    NotSupportedEditRequest,
    Retdiff,
    Score,
    Weight,
)
from genjax._src.core.generative.functional_types import Mask
from genjax._src.core.generative.interface import Trace
from genjax._src.core.generative.requests import Regenerate, Update
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Annotated,
    Any,
    Callable,
    Is,
    TypeVar,
)

R = TypeVar("R")

####################
# Posterior target #
####################


def validate_non_marginal(x):
    if isinstance(x, Marginal):
        raise TypeError("Target does not support Marginal generative functions.")
    return True


@Pytree.dataclass
class Target(Pytree):
    """
    A `Target` represents an unnormalized target distribution induced by conditioning a generative function on a [`genjax.ChoiceMap`][].

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

    p: Annotated[GFI[Any], Is[validate_non_marginal]]
    args: tuple[Any, ...]
    constraint: ChoiceMap

    def importance(
        self,
        constraint: ChoiceMap,
    ) -> tuple[Trace[Any], Weight]:
        merged = self.constraint.merge(constraint)
        return self.p.importance(merged, self.args)

    def filter_to_unconstrained(self, choice_map):
        selection = ~self.constraint.get_selection()
        return choice_map.filter(selection)

    def __getitem__(self, addr):
        return self.constraint[addr]


#######################
# Sample distribution #
#######################


@Pytree.dataclass
class ChoiceMapDistributionTrace(Trace[ChoiceMap]):
    gen_fn: GFI[Any]
    args: tuple[Any, ...]
    value: ChoiceMap
    score: Score

    def get_args(self) -> tuple[Any, ...]:
        return self.args

    def get_retval(self) -> ChoiceMap:
        return self.value

    def get_gen_fn(self) -> GFI[Any]:
        return self.gen_fn

    def get_score(self) -> Score:
        return self.score

    def get_choices(self) -> ChoiceMap:
        return ChoiceMap.choice(self.value)


class ChoiceMapDistribution(GFI[ChoiceMap], Pytree):
    """
    The abstract class `SampleDistribution` represents the type of distributions whose return value type is a `ChoiceMap`. This is the abstract base class of `Algorithm`, as well as `Marginal`.
    """

    @abstractmethod
    def random_weighted(
        self,
        *args: Any,
    ) -> tuple[Score, ChoiceMap]:
        """
        Given a [`Target`][genjax.inference.Target], return a [`ChoiceMap`][genjax.core.ChoiceMap] from an approximation to the normalized distribution of the target, and a random [`Weight`][genjax.core.Weight] estimate of the normalized density of the target at the sample.

        The `sample` is a sample on the support of `target.gen_fn` which _are not in_ `target.constraints`, produced by running the inference algorithm.

        Let $T_P(a, c)$ denote the target, with $P$ the distribution on samples represented by `target.gen_fn`, and $S$ denote the sample. Let $w$ denote the weight `w`. The weight $w$ is a random weight such that $w$ satisfies:

        $$
        \\mathbb{E}\\big[\\frac{1}{w} \\mid S \\big] = \\frac{1}{P(S \\mid c; a)}
        $$

        This interface corresponds to **(Defn 3.2) Unbiased Density Sampler** in [[Lew23](https://dl.acm.org/doi/pdf/10.1145/3591290)].
        """
        assert isinstance(args[0], Target)

    @abstractmethod
    def estimate_logpdf(
        self,
        v: ChoiceMap,
        *args,
    ) -> Score:
        """
        Given a [`ChoiceMap`][genjax.core.ChoiceMap] and a [`Target`][genjax.inference.Target], return a random [`Weight`][genjax.core.Weight] estimate of the normalized density of the target at the sample.

        Let $T_P(a, c)$ denote the target, with $P$ the distribution on samples represented by `target.gen_fn`, and $S$ denote the sample. Let $w$ denote the weight `w`. The weight $w$ is a random weight such that $w$ satisfies:

        $$
        \\mathbb{E}[w] = P(S \\mid c, a)
        $$

        This interface corresponds to **(Defn 3.1) Positive Unbiased Density Estimator** in [[Lew23](https://dl.acm.org/doi/pdf/10.1145/3591290)].
        """

    def simulate(
        self,
        args: tuple[Any, ...],
    ) -> Trace[ChoiceMap]:
        (w, v) = self.random_weighted(*args)
        tr = ChoiceMapDistributionTrace(self, args, v, w)
        return tr

    def generate_choice_map(
        self,
        chm: ChoiceMap,
        args: tuple[Any, ...],
    ) -> tuple[Trace[ChoiceMap], Weight]:
        v = chm.get_value()
        match v:
            case None:
                tr = self.simulate(args)
                return tr, jnp.array(0.0)

            case Mask(value, flag):

                def _simulate(v):
                    score, new_v = self.random_weighted(*args)
                    w = 0.0
                    return (score, w, new_v)

                def _importance(v):
                    w = self.estimate_logpdf(v, *args)
                    return (w, w, v)

                score, w, new_v = cond(flag, _importance, _simulate, value)
                tr = ChoiceMapDistributionTrace(self, args, new_v, score)
                return tr, w

            case _:
                w = self.estimate_logpdf(v, *args)
                tr = ChoiceMapDistributionTrace(self, args, v, w)
                return tr, w

    def generate(
        self,
        constraint: ChoiceMap,
        args: tuple[Any, ...],
    ) -> tuple[Trace[ChoiceMap], Weight]:
        match constraint:
            case ChoiceMap():
                tr, w = self.generate_choice_map(constraint, args)

            case _:
                raise Exception("Unhandled type.")
        return tr, w

    def project(
        self,
        trace: Trace[ChoiceMap],
        selection: Selection,
    ) -> Weight:
        raise NotImplementedError

    def edit(
        self,
        trace: Trace[ChoiceMap],
        edit_request: EditRequest,
        argdiffs: Argdiffs,
    ) -> tuple[Trace[ChoiceMap], Weight, Retdiff[ChoiceMap], EditRequest]:
        match edit_request:
            case Update(_):
                raise NotImplementedError
            case Regenerate(_):
                raise NotImplementedError

            case _:
                raise NotSupportedEditRequest(edit_request)

    def assess(
        self,
        chm: ChoiceMap,
        args: tuple[Any, ...],
    ):
        raise NotImplementedError


########################
# Inference algorithms #
########################


class Algorithm(ChoiceMapDistribution):
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
        *args: Any,
    ) -> tuple[Score, ChoiceMap]:
        """
        Given a [`Target`][genjax.inference.Target], return a [`ChoiceMap`][genjax.core.ChoiceMap] from an approximation to the normalized distribution of the target, and a random [`Weight`][genjax.core.Weight] estimate of the normalized density of the target at the sample.

        The `sample` is a sample on the support of `target.gen_fn` which _are not in_ `target.constraints`, produced by running the inference algorithm.

        Let $T_P(a, c)$ denote the target, with $P$ the distribution on samples represented by `target.gen_fn`, and $S$ denote the sample. Let $w$ denote the weight `w`. The weight $w$ is a random weight such that $w$ satisfies:

        $$
        \\mathbb{E}\\big[\\frac{1}{w} \\mid S \\big] = \\frac{1}{P(S \\mid c; a)}
        $$

        This interface corresponds to **(Defn 3.2) Unbiased Density Sampler** in [[Lew23](https://dl.acm.org/doi/pdf/10.1145/3591290)].
        """
        assert isinstance(args[0], Target)

    @abstractmethod
    def estimate_logpdf(self, v: ChoiceMap, *args: tuple[Any, ...]) -> Score:
        """
        Given a [`ChoiceMap`][genjax.core.ChoiceMap] and a [`Target`][genjax.inference.Target], return a random [`Weight`][genjax.core.Weight] estimate of the normalized density of the target at the sample.

        Let $T_P(a, c)$ denote the target, with $P$ the distribution on samples represented by `target.gen_fn`, and $S$ denote the sample. Let $w$ denote the weight `w`. The weight $w$ is a random weight such that $w$ satisfies:

        $$
        \\mathbb{E}[w] = P(S \\mid c, a)
        $$

        This interface corresponds to **(Defn 3.1) Positive Unbiased Density Estimator** in [[Lew23](https://dl.acm.org/doi/pdf/10.1145/3591290)].
        """

    ################
    # VI via GRASP #
    ################

    @abstractmethod
    def estimate_normalizing_constant(
        self,
        target: Target,
    ) -> Weight:
        pass

    @abstractmethod
    def estimate_reciprocal_normalizing_constant(
        self,
        target: Target,
        latent_choices: ChoiceMap,
        w: Weight,
    ) -> Weight:
        pass


############
# Marginal #
############


@Pytree.dataclass
class Marginal(ChoiceMapDistribution):
    """The `Marginal` class represents the marginal distribution of a generative function over
    a selection of addresses.
    """

    gen_fn: GFI[Any]
    selection: Selection = Pytree.field(default=Selection.all())
    algorithm: Algorithm | None = Pytree.field(default=None)

    def random_weighted(
        self,
        *args: Any,
    ) -> tuple[Score, ChoiceMap]:
        tr = self.gen_fn.simulate(args)
        choices: ChoiceMap = tr.get_choices()
        latent_choices = choices.filter(self.selection)
        bwd_request = ~self.selection
        weight = tr.project(bwd_request)
        if self.algorithm is None:
            return weight, latent_choices
        else:
            target = Target(self.gen_fn, args, latent_choices)
            other_choices = choices.filter(~self.selection)
            Z = self.algorithm.estimate_reciprocal_normalizing_constant(
                target, other_choices, weight
            )

            return (Z, latent_choices)

    def estimate_logpdf(
        self,
        v: ChoiceMap,
        *args: tuple[Any, ...],
    ) -> Score:
        if self.algorithm is None:
            _, weight = self.gen_fn.importance(v, args)
            return weight
        else:
            target = Target(self.gen_fn, args, v)
            Z = self.algorithm.estimate_normalizing_constant(target)
            return Z


################################
# Inference construct language #
################################


def marginal(
    selection: Selection = Selection.all(),
    algorithm: Algorithm | None = None,
) -> Callable[[GFI[Any]], Marginal]:
    def decorator(
        gen_fn: GFI[R],
    ) -> Marginal:
        return Marginal(
            gen_fn,
            selection,
            algorithm,
        )

    return decorator
