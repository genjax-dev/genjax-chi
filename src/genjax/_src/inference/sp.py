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
    Address,
    ChoiceMap,
    ChoiceMapBuilder,
    Constraint,
    GenerativeFunction,
    Sample,
    Selection,
    SelectionBuilder,
    Weight,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
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


@Pytree.dataclass
class Target(Pytree):
    """
    A `Target` represents an unnormalized target distribution induced by conditioning a generative function on a [`Constraint`](core.md#genjax.core.Constraint).

    Targets are created by providing a generative function, arguments to the generative function, and a constraint.

    Examples:
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
    args: Tuple
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
class SampleDistribution(Distribution):
    """
    The abstract class `SampleDistribution` represents the type of distributions whose return value type is a `Sample`. This is the abstract base class of `InferenceAlgorithm`, as well as `Marginal`.
    """

    @abstractmethod
    def random_weighted(
        self,
        key: PRNGKey,
        *args: Any,
    ) -> Tuple[FloatArray, Sample]:
        raise NotImplementedError

    @abstractmethod
    def estimate_logpdf(
        self,
        key: PRNGKey,
        latent_choices: Sample,
        *args: Any,
    ) -> FloatArray:
        raise NotImplementedError


########################
# Inference algorithms #
########################


class InferenceAlgorithm(SampleDistribution):
    """The abstract class `InferenceAlgorithm` is the type of inference
    algorithms: programs which provide interfaces for sampling from
    posterior approximations, and estimating the density of these samplers.

    Inference algorithms expose two key interfaces:
    [`InferenceAlgorithm.random_weighted`](inference.md#genjax.inference.InferenceAlgorithm.random_weighted)
    and [`InferenceAlgorithm.estimate_logpdf`](inference.md#genjax.inference.InferenceAlgorithm.estimate_logpdf).

    `InferenceAlgorithm.random_weighted` exposes sampling from the approximation
    which the algorithm represents: it accepts a `Target` as input, representing the
    unnormalized distribution $\\pi$, and samples from an approximation to
    the normalized distribution $\\pi / Z$.

    `InferenceAlgorithm.estimate_logpdf` exposes density _estimation_ for the
    approximation which `InferenceAlgorithm.random_weighted` samples from:
    it accepts a value sampled from this approximation, and the `Target` which
    induced the approximation as input, and returns an estimate of the density of
    the approximation.

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
    ) -> Tuple[FloatArray, Sample]:
        """
        Given a `key: PRNGKey`, and a `target: Target`, returns a pair `(log(w), sample)`.

        The sample from the approximation `sample : Sample` is a sample on the support
        of `target.gen_fn` which _are not in_ `target.constraints`.
        The sample is produced by running the inference algorithm represented by `self`.

        Let `T` denote the target and `S` denote the sample. The weight `log(w)`
        is a random weight such that $w$ satisfies:

        $$
        \\mathbb{E}\\big[\\frac{1}{w} \\mid \\texttt{S}\\big] = \\frac{1}{P(\\texttt{S} \\mid \\texttt{T.constraints}; \\texttt{T.args})}
        $$

        where `P` is the distribution on samples represented by `target.gen_fn`.
        """
        pass

    @abstractmethod
    def estimate_logpdf(
        self,
        key: PRNGKey,
        sample: Sample,
        target: Target,
    ) -> FloatArray:
        """
        Given a `key: PRNGKey`, `sample: Sample` and a `target: Target`, returns a random value $\\log(w)$.

        Let `T` denote the target and `S` denote the sample. The weight $\\log(w)$
        is a random weight such that $w$ satisfies:

        $$
        \\mathbb{E}[w] = P(\\texttt{S} \\mid \\texttt{T.constraints})
        $$

        where $P$ is the distribution on samples provided by `T.gen_fn`.
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
        latent_choices: Sample,
        w: Weight,
    ) -> Weight:
        pass


############
# Marginal #
############


@Pytree.dataclass
@typecheck
class Marginal(SampleDistribution):
    """The `Marginal` class represents the marginal distribution of a generative function over
    a selection of addresses. The return value type is a subtype of `Sample`.
    """

    gen_fn: GenerativeFunction
    selection: Selection = Pytree.field(default=Selection.all())
    algorithm: Optional[InferenceAlgorithm] = Pytree.field(default=None)

    @typecheck
    def random_weighted(
        self,
        key: PRNGKey,
        *args,
    ) -> Tuple[Weight, Sample]:
        key, sub_key = jax.random.split(key)
        tr = self.gen_fn.simulate(sub_key, args)
        choices: ChoiceMap = tr.get_sample()
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
        constraint: Constraint,
        *args,
    ) -> Weight:
        if self.algorithm is None:
            _, weight = self.gen_fn.importance(key, constraint, args)
            return weight
        else:
            target = Target(self.gen_fn, args, constraint)
            Z = self.algorithm.estimate_normalizing_constant(key, target)
            return Z


@Pytree.dataclass
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
        *args,
    ) -> Tuple[Weight, Any]:
        marginal = Marginal(
            self.p,
            SelectionBuilder[self.addr],
            self.algorithm,
        )
        Z, choice = marginal.random_weighted(key, *args)
        return Z, choice[self.addr]

    @typecheck
    def estimate_logpdf(
        self,
        key: PRNGKey,
        v: Any,
        *args,
    ) -> Weight:
        marginal = Marginal(
            self.p,
            SelectionBuilder[self.addr],
            self.algorithm,
        )
        latent_choice: Sample = ChoiceMapBuilder.a(self.addr, v)
        return marginal.estimate_logpdf(key, latent_choice, *args)


################################
# Inference construct language #
################################


@typecheck
def marginal(
    gen_fn: Optional[GenerativeFunction] = None,
    /,
    *,
    select_or_addr: Optional[Selection | Address] = None,
    algorithm: Optional[InferenceAlgorithm] = None,
) -> Callable | GenerativeFunction:
    """If `select_or_addr` is a `Selection`, this constructs a `Marginal` distribution
    which samples `Sample` objects with addresses given in the selection.
    If `select_or_addr` is an address, this constructs a `ValueMarginal` distribution
    which samples values of the type stored at the given address in `gen_fn`.
    """

    @Pytree.partial(select_or_addr)
    def decorator(
        select_or_addr: Optional[Selection | Address],
        gen_fn: GenerativeFunction,
    ) -> Marginal | ValueMarginal:
        if not select_or_addr:
            select_or_addr = Selection.all()
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
        return marginal

    if gen_fn is not None:
        return decorator(gen_fn)
    else:
        return decorator
