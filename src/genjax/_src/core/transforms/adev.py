# Copyright 2022 MIT Probabilistic Computing Project
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
import dataclasses
import functools

import jax.tree_util as jtu

from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.interpreters import context
from genjax._src.core.interpreters import primitives
from genjax._src.core.pytree import Pytree
from genjax._src.core.transforms import harvest
from genjax._src.core.typing import Callable
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import typecheck


#############################
# Probabilistic computation #
#############################


@dataclasses.dataclass
class AbstractProbabilisticComputation(Pytree):
    def simulate(self, key, args):
        pass

    def grad_estimate(self, key, args):
        pass


# Overloads for leaf generative functions, like distributions.
@dataclasses.dataclass
class ProbabilisticPrimitive(AbstractProbabilisticComputation):
    distribution: GenerativeFunction

    def simulate(self, key, args):
        return self.distribution.random_weighted(key, *args)


@dataclasses.dataclass
class ProbabilisticComputation(AbstractProbabilisticComputation):
    source: Callable

    def flatten(self):
        return (), (self.source,)


@typecheck
def prob_comp(gen_fn: GenerativeFunction):
    return ProbabilisticComputation(gen_fn.adev_convert)


#######################
# Gradient strategies #
#######################


@dataclasses.dataclass
class GradientStrategy(Pytree):
    pass


@dataclasses.dataclass
class Reinforce(GradientStrategy):
    pass


@dataclasses.dataclass
class ExactEnum(GradientStrategy):
    pass


######################
# Strategy intrinsic #
######################

NAMESPACE = "adev_strategy"
adev_tag = functools.partial(harvest.sow, tag=NAMESPACE)


@typecheck
def strat(strategy: GradientStrategy, addr):
    return adev_tag(strategy, name=addr)


####################
# Sample intrinsic #
####################

sample_p = primitives.InitialStylePrimitive("sample")


def _abstract_prob_comp_call(prob_comp, *args):
    return prob_comp.simulate(*args)


def _sample(prob_comp, strat, key, args, **kwargs):
    return primitives.initial_style_bind(sample_p)(_abstract_prob_comp_call)(
        prob_comp, key, strat, args, **kwargs
    )


@typecheck
def sample(
    prob_comp: AbstractProbabilisticComputation,
    key: PRNGKey,
    strat: GradientStrategy,
    args: Tuple,
    **kwargs
):
    return _sample(prob_comp, key, strat, args, **kwargs)


##############
# Transforms #
##############

#####
# Simulate
#####


@dataclasses.dataclass
class ADEVContext(context.Context):
    @abc.abstractmethod
    def handle_sample(self, *tracers, **params):
        pass

    def can_handle(self, primitive):
        return False

    def process_primitive(self, primitive):
        raise NotImplementedError

    def get_custom_rule(self, primitive):
        if primitive is sample_p:
            return self.handle_trace
        else:
            return None


@dataclasses.dataclass
class SimulateContext(ADEVContext):
    def flatten(self):
        return (), ()

    @classmethod
    def new(cls):
        return SimulateContext()

    def yield_state(self):
        return ()

    def handle_sample(self, _, *args, **params):
        in_tree = params.get("in_tree")
        prob_comp, key, _, args = jtu.tree_unflatten(in_tree, args)
        key, v = prob_comp.simulate(key, args)
        return key, jtu.tree_leaves(v)


def simulate_transform(source_fn, **kwargs):
    @functools.wraps(source_fn)
    def wrapper(*args):
        ctx = SimulateContext.new()
        retvals, _ = context.transform(source_fn, ctx)(*args, **kwargs)
        return retvals

    return wrapper


#####
# Grad estimate
#####


@dataclasses.dataclass
class GradEstimateContext(ADEVContext):
    def flatten(self):
        return (), ()

    @classmethod
    def new(cls):
        return GradEstimateContext()

    def yield_state(self):
        return ()

    def handle_sample(self, _, *args, **params):
        in_tree = params["in_tree"]
        cont = params["cont"]
        prob_comp, key, strat, args = jtu.tree_unflatten(in_tree, args)
        key, v = strat.apply(cont, prob_comp, key, args)
        return key, jtu.tree_leaves(v)


def grad_estimate_transform(source_fn, **kwargs):
    @functools.wraps(source_fn)
    def wrapper(*args):
        ctx = GradEstimateContext.new()
        retvals, _ = context.transform(source_fn, ctx)(*args, **kwargs)
        return retvals

    return wrapper
