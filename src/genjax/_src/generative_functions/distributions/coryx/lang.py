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

import functools
from dataclasses import dataclass

import jax
import jax.core as jc
import jax.tree_util as jtu

from genjax._src.core.datatypes import GenerativeFunction
from genjax._src.core.interpreters import primitives
from genjax._src.core.interpreters.cps import cps
from genjax._src.core.interpreters.staging import stage
from genjax._src.core.typing import Any
from genjax._src.core.typing import Callable
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import List
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.builtin.builtin_transforms import Bare
from genjax._src.generative_functions.distributions.coryx import core as inverse_core
from genjax._src.generative_functions.distributions.distribution import ExactDensity


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
def rv(gen_fn: GenerativeFunction, **kwargs):
    return lambda *args: _random_variable(gen_fn, *args, **kwargs)


# We defer the abstract call here so that, when we
# stage, any traced values stored in `gen_fn`
# get lifted to by `get_shaped_aval`.
def _abstract_gen_fn_call(gen_fn, *args):
    return gen_fn.__abstract_call__(*args)


##############
# Transforms #
##############

#####
# Sample
#####


@dataclass
class Sample(cps.Handler):
    handles: List[jc.Primitive]
    key: PRNGKey

    def flatten(self):
        return (self.key,), (self.handles,)

    @classmethod
    def new(cls, key: PRNGKey):
        handles = [random_variable_p]
        return Sample(handles, key)

    def random_variable(self, cell_type, prim, args, cont, in_tree, **kwargs):
        gen_fn, *args = jtu.tree_unflatten(in_tree, cps.static_map_unwrap(args))
        self.key, sub_key = jax.random.split(self.key)
        v = gen_fn.sample(sub_key, *args)
        v = cps.flatmap_outcells(cell_type, v)
        return cont(*v)


def sample_transform(source_fn, **kwargs):
    @functools.wraps(source_fn)
    def _inner(key, *args):
        closed_jaxpr, (flat_args, _, out_tree) = stage(source_fn)(*args, **kwargs)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        handler = Sample.new(key)
        with cps.Interpreter.new(Bare, handler) as interpreter:
            flat_out = interpreter(
                jaxpr, [Bare.new(v) for v in consts], list(map(Bare.new, flat_args))
            )
            flat_out = map(lambda v: v.get_val(), flat_out)
            retvals = jtu.tree_unflatten(out_tree, flat_out)
            key = handler.key

        return key, retvals

    return _inner


#####
# Sow
#####


@dataclass
class Sow(cps.Handler):
    handles: List[jc.Primitive]
    values: List[Any]
    score: FloatArray

    def flatten(self):
        return (self.values, self.score), (self.handles,)

    @classmethod
    def new(cls, values: List[Any]):
        handles = [random_variable_p]
        values.reverse()
        score = 0.0
        return Sow(handles, values, score)

    def random_variable(self, cell_type, prim, args, cont, in_tree, **kwargs):
        gen_fn, *args = jtu.tree_unflatten(in_tree, cps.static_map_unwrap(args))
        v = self.values.pop()
        w = gen_fn.logpdf(v, *args)
        self.score += w
        v = cps.flatmap_outcells(cell_type, v)
        return cont(*v)


def sow_transform(source_fn, constraints, **kwargs):
    @functools.wraps(source_fn)
    def _inner(*args):
        closed_jaxpr, (flat_args, _, out_tree) = stage(source_fn)(*args, **kwargs)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        if isinstance(constraints, tuple):
            handler = Sow.new([*constraints])
        else:
            handler = Sow.new([constraints])
        with cps.Interpreter.new(Bare, handler) as interpreter:
            flat_out = interpreter(
                jaxpr, [Bare.new(v) for v in consts], list(map(Bare.new, flat_args))
            )
        flat_out = map(lambda v: v.get_val(), flat_out)
        retvals = jtu.tree_unflatten(out_tree, flat_out)
        score = handler.score
        return retvals, score

    return _inner


#####################
# Distribution type #
#####################


@dataclass
class TransformedDistribution(ExactDensity):
    source: Callable

    def sample(self, key, *args, **kwargs):
        _, v = sample_transform(self.source)(key, *args)
        return v

    def logpdf(self, v, *args, **kwargs):
        def returner(constraints):
            return sow_transform(self.source, constraints)(*args)[0]

        def scorer(constraints):
            return sow_transform(self.source, constraints)(*args)[1]

        inverses, ildj_correction = inverse_core.inverse_and_ildj(returner)(v)
        score = scorer(inverses) + ildj_correction
        return score


##############
# Shorthands #
##############

dist = TransformedDistribution.new
