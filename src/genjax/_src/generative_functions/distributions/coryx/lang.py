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
"""This module holds the language interface for the `TransformedDistribution`
DSL. It borrows syntax from the `BuiltinGenerativeFunction` DSL, and utilizes
some of the `BuiltinGenerativeFunction` transformation infrastructure.

It also relies on `coryx` - the core [`Oryx`][oryx] functionality forked from Oryx and implemented in the enclosing `coryx` module.

[oryx]: https://github.com/jax-ml/oryx
"""

import abc
import dataclasses
import functools

import jax
import jax.tree_util as jtu

from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.interpreters import context as ctx
from genjax._src.core.interpreters import primitives
from genjax._src.core.typing import Any
from genjax._src.core.typing import Callable
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import List
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.distributions.coryx import core as inverse_core
from genjax._src.generative_functions.distributions.distribution import Distribution


##############
# Intrinsics #
##############

random_variable_p = primitives.InitialStylePrimitive("random_variable")


def _random_variable(gen_fn, *args, **kwargs):
    result = primitives.initial_style_bind(random_variable_p)(_abstract_gen_fn_call)(
        gen_fn, *args, **kwargs
    )
    return result


@typecheck
def random_variable(gen_fn: GenerativeFunction, **kwargs):
    return lambda *args: _random_variable(gen_fn, *args, **kwargs)


# We defer the abstract call here so that, when we
# stage, any traced values stored in `gen_fn`
# get lifted to by `get_shaped_aval`.
def _abstract_gen_fn_call(gen_fn, *args):
    return gen_fn.__abstract_call__(*args)


##############
# Transforms #
##############

# NOTE: base context class for transforms below.
@dataclasses.dataclass
class CoryxContext(ctx.Context):
    @abc.abstractmethod
    def handle_random_variable(self, *tracers, **params):
        pass

    def can_process(self, primitive):
        return False

    def process_primitive(self, primitive):
        raise NotImplementedError

    def get_custom_rule(self, primitive):
        if primitive is random_variable_p:
            return self.handle_random_variable
        else:
            return None


#####
# Sample
#####


@dataclasses.dataclass
class SampleContext(CoryxContext):
    key: PRNGKey

    def flatten(self):
        return (self.key,), (self.handles,)

    @classmethod
    def new(cls, key: PRNGKey):
        return SampleContext(key)

    def handle_random_variable(self, _, *tracers, **params):
        in_tree = params["in_tree"]
        gen_fn, *args = jtu.tree_unflatten(in_tree, tracers)
        self.key, (_, v) = gen_fn.random_weighted(self.key, *args)
        return jtu.tree_leaves(v)


def sample_transform(source_fn, **kwargs):
    @functools.wraps(source_fn)
    def wrapper(key, *args):
        context = SampleContext.new(key)
        retvals, statefuls = ctx.transform(source_fn, context)(*args, **kwargs)
        (key,) = statefuls
        return key, retvals

    return wrapper


#####
# Sow
#####


@dataclasses.dataclass
class SowContext(CoryxContext):
    key: PRNGKey
    values: List[Any]
    score: FloatArray

    def flatten(self):
        return (self.key, self.values, self.score), ()

    @classmethod
    def new(cls, key: PRNGKey, values: List[Any]):
        values.reverse()
        score = 0.0
        return SowContext(key, values, score)

    def handle_random_variable(self, _, *tracers, **params):
        in_tree = params["in_tree"]
        gen_fn, *args = jtu.tree_unflatten(in_tree, tracers)
        v = self.values.pop()
        self.key, w = gen_fn.estimate_logpdf(self.key, v, *args)
        self.score += w
        return jtu.tree_leaves(v)


def sow_transform(source_fn, constraints, **kwargs):
    @functools.wraps(source_fn)
    def wrapper(key, *args):
        if isinstance(constraints, tuple):
            context = SowContext.new(key, [*constraints])
        else:
            context = SowContext.new(key, [constraints])
        retvals, statefuls = ctx.transform(source_fn, context)(*args, **kwargs)
        key, score = statefuls
        return key, (retvals, score)

    return wrapper


#####################
# Distribution type #
#####################


@dataclasses.dataclass
class TransformedDistribution(Distribution):
    source: Callable

    def random_weighted(self, key, *args, **kwargs):
        key, v = sample_transform(self.source)(key, *args)
        key, w = self.estimate_logpdf(key, v, *args)
        return key, (w, v)

    def estimate_logpdf(self, key, v, *args, **kwargs):
        key, sub_key = jax.random.split(key)

        def returner(constraints):
            return sow_transform(self.source, constraints)(sub_key, *args)[1][0]

        def scorer(constraints):
            return sow_transform(self.source, constraints)(sub_key, *args)[1][1]

        inverses, ildj_correction = inverse_core.inverse_and_ildj(returner)(v)
        score = scorer(inverses) + ildj_correction
        return key, score


##############
# Shorthands #
##############

lang = TransformedDistribution.new
rv = random_variable
