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
    Constraint,
    GenerativeFunction,
    GenerativeFunctionClosure,
    RemoveSelectionUpdateSpec,
    Sample,
    Selection,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
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


@Pytree.dataclass
class Target(Pytree):
    """
    Instances of `Target` represent unnormalized target distributions. A `Target` is created by pairing a generative function and its arguments with a `Sample` object.
    The target represents the unnormalized distribution on the unconstrained choices in the generative function, fixing the constraints.
    """

    p: GenerativeFunctionClosure
    args: Tuple
    constraint: Constraint


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
    ) -> Tuple[FloatArray, Sample]:
        """
        Given a `key: PRNGKey`, and a `target: Target`, returns a pair `(log_w, choice)`.
        `choice : Sample` is a choicemap on the addresses sampled at in `target.gen_fn` not in `target.constraints`;
        it is sampled by running the inference algorithm represented by `self`.
        `log_w` is a random weight such that $w = \\exp(\\texttt{log_w})$ satisfies
        $\\mathbb{E}[1 / w \\mid \\texttt{choice}] = 1 / P(\\texttt{choice} \\mid \\texttt{target.constraints})`, where `P` is the
        distribution on choicemaps represented by `target.gen_fn`.
        """
        pass

    @abstractmethod
    def estimate_logpdf(
        self,
        key: PRNGKey,
        latent_choices: Sample,
        target: Target,
    ) -> FloatArray:
        """
        Given a `key: PRNGKey`, `latent_choices: Choice` and a `target: Target`, returns a random value $\\log(w)$
        such that $\\mathbb{E}[w] = P(\texttt{latent_choices} \\mid \\texttt{target.constraints})$, where $P$
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
        latent_choices: Sample,
        w: FloatArray,
    ) -> FloatArray:
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

    args: Tuple
    gen_fn: GenerativeFunctionClosure
    selection: Selection = Pytree.field(default=Selection.a)
    algorithm: Optional[InferenceAlgorithm] = Pytree.field(default=None)

    @typecheck
    def random_weighted(
        self,
        key: PRNGKey,
    ) -> Tuple[FloatArray, Sample]:
        key, sub_key = jax.random.split(key)
        gen_fn = self.gen_fn(*self.args)
        tr = gen_fn.simulate(sub_key)
        choices = tr.get_sample()
        latent_choices = choices.filter(self.selection)
        key, sub_key = jax.random.split(key)
        bwd_spec = RemoveSelectionUpdateSpec(~self.selection)
        weight = tr.project(sub_key, bwd_spec)
        if self.algorithm is None:
            return weight, latent_choices
        else:
            target = Target(self.gen_fn, self.args, latent_choices)
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
    ) -> FloatArray:
        gen_fn = self.gen_fn(*self.args)
        if self.algorithm is None:
            _, weight = gen_fn.importance(key, constraint)
            return weight
        else:
            target = Target(gen_fn, self.args, constraint)
            Z = self.algorithm.estimate_normalizing_constant(key, target)
            return Z


@Pytree.dataclass
@typecheck
class ValueMarginal(Distribution):
    """The `ValueMarginal` class represents the marginal distribution of a generative function over
    a single address `addr: Any`. The return value type is the type of the value at that address.
    """

    args: Tuple
    p: GenerativeFunctionClosure
    addr: Any
    algorithm: Optional[InferenceAlgorithm] = Pytree.field(default=None)

    @typecheck
    def random_weighted(
        self,
        key: PRNGKey,
    ) -> Tuple[FloatArray, Any]:
        marginal = Marginal(
            self.args,
            self.p,
            Selection.at[self.addr],
            self.algorithm,
        )
        Z, choice = marginal.random_weighted(key)
        return Z, choice[self.addr]

    @typecheck
    def estimate_logpdf(
        self,
        key: PRNGKey,
        v: Any,
    ) -> FloatArray:
        marginal = Marginal(
            self.args,
            self.p,
            Selection.at[self.addr],
            self.algorithm,
        )
        latent_choice = Sample.a(self.addr, v)
        return marginal.estimate_logpdf(key, latent_choice)

    @property
    def __wrapped__(self):
        return self.p


################################
# Inference construct language #
################################


@typecheck
def marginal(
    gen_fn_closure: Optional[GenerativeFunctionClosure] = None,
    /,
    *,
    select_or_addr: Union[Selection, Any] = Selection.a,
    algorithm: Optional[InferenceAlgorithm] = None,
) -> (
    Callable[[GenerativeFunctionClosure], GenerativeFunctionClosure]
    | GenerativeFunctionClosure
):
    """If `select_or_addr` is a `Selection`, this constructs a `Marginal` distribution
    which samples `Sample` objects with addresses given in the selection.
    If `select_or_addr` is an address, this constructs a `ValueMarginal` distribution
    which samples values of the type stored at the given address in `gen_fn`.
    """

    def decorator(gen_fn_closure):
        @GenerativeFunction.closure
        def marginal_closure(*args) -> Union[Marginal, ValueMarginal]:
            if isinstance(select_or_addr, Selection):
                marginal = Marginal(
                    args,
                    gen_fn_closure,
                    select_or_addr,
                    algorithm,
                )
            else:
                marginal = ValueMarginal(
                    args,
                    gen_fn_closure,
                    select_or_addr,
                    algorithm,
                )
            return marginal

        return marginal_closure

    if gen_fn_closure is not None:
        return decorator(gen_fn_closure)
    else:
        return decorator
