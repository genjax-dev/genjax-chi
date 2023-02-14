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

import jax.core as jc
import jax.tree_util as jtu

import genjax._src.core.interpreters.cps as cps
from genjax._src.core.datatypes import GenerativeFunction
from genjax._src.core.staging import stage
from genjax._src.core.typing import Callable
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import List
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.builtin.intrinsics import gen_fn_p
from genjax._src.generative_functions.builtin.transforms import Bare
from genjax._src.generative_functions.distributions.distribution import Distribution


##############
# Intrinsics #
##############

random_variable_p = jc.Primitive("random_variable")


def _random_variable(gen_fn, *args, **kwargs):
    flat_args, tree_in = jtu.tree_flatten((gen_fn, args))
    retvals = random_variable_p.bind(*flat_args, tree_in=tree_in, **kwargs)
    retvals = tuple(retvals)

    # Collapse single arity returns.
    return retvals[0] if len(retvals) == 1 else retvals


@typecheck
def rv(gen_fn: GenerativeFunction, **kwargs):
    return lambda *args: _random_variable(gen_fn, *args, **kwargs)


# We defer the abstract call here so that, when we
# stage, any traced values stored in `gen_fn`
# get lifted to by `get_shaped_aval`.
def _abstract_gen_fn_call(gen_fn, *args):
    return gen_fn.__abstract_call__(*args)


def random_variable_abstract_eval(*args, tree_in, **kwargs):
    gen_fn, args = jtu.tree_unflatten(tree_in, args)

    # See note above on `_abstract_gen_fn_call`.
    closed_jaxpr, _ = stage(_abstract_gen_fn_call)(gen_fn, *args)

    retvals = closed_jaxpr.out_avals[1:]
    return retvals


random_variable_p.def_abstract_eval(random_variable_abstract_eval)
random_variable_p.multiple_results = True

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
    score: FloatArray

    def flatten(self):
        return (
            self.key,
            self.score,
        ), (self.handles,)

    @classmethod
    def new(cls, key: PRNGKey):
        score = 0.0
        handles = [gen_fn_p]
        return Sample(
            handles,
            key,
            score,
        )

    def random_variable(self, cell_type, prim, args, cont, addr, tree_in, **kwargs):
        # Unflatten the flattened `Pytree` arguments.
        gen_fn, args = jtu.tree_unflatten(tree_in, cps.static_map_unwrap(args))

        # Send the GFI call to the generative function callee.
        self.key, (w, v) = gen_fn.random_weighted(self.key, *args, **kwargs)

        # Set state in the handler.
        self.score += w

        # Get the return value, lift back to the CPS
        # interpreter value type lattice (here, `Bare`).
        v = cps.flatmap_outcells(cell_type, v)
        return cont(*v)


def sample_transform(source_fn, **kwargs):
    @functools.wraps(source_fn)
    def _inner(key, args):
        closed_jaxpr, (flat_args, _, out_tree) = stage(source_fn)(*args, **kwargs)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        handler = Sample.new(key)
        with cps.Interpreter.new(Bare, handler) as interpreter:
            flat_out = interpreter(
                jaxpr, [Bare.new(v) for v in consts], list(map(Bare.new, flat_args))
            )
            flat_out = map(lambda v: v.get_val(), flat_out)
            retvals = jtu.tree_unflatten(out_tree, flat_out)
            score = handler.score
            key = handler.key

        return key, (retvals, score)

    return _inner


#####
# Logpdf
#####


@dataclass
class Logpdf(cps.Handler):
    pass


#####################
# Distribution type #
#####################


@dataclass
class TransformedDistribution(Distribution):
    source: Callable

    def random_weighted(self, key, *args, **kwargs):
        key, v = sample_transform(self.source)(key, args)
        # key, score = logpdf_transform(self.source)(key, v, args)
        return key, (0.0, v)

    def estimate_logpdf(self, key, v, *args, **kwargs):
        # key, score = logpdf_transform(self.source)(key, v, args)
        return key, 0.0


##############
# Shorthands #
##############

trans_dist = TransformedDistribution.new
