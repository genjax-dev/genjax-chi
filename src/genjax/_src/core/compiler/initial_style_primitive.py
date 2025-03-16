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

import itertools as it

import jax.core as jc
from jax import tree_util
from jax import util as jax_util
from jax.extend.core import Primitive
from jax.interpreters import mlir
from jax.interpreters import partial_eval as pe

from genjax._src.core.compiler.staging import stage

#########################
# Custom JAX primitives #
#########################


class InitialStylePrimitive(Primitive):
    """Contains default implementations of transformations."""

    def __init__(self, name):
        super(InitialStylePrimitive, self).__init__(name)
        self.multiple_results = True

        def _abstract(*flat_avals, **params):
            abs_eval = params["abs_eval"]
            return abs_eval(*flat_avals, **params)

        self.def_abstract_eval(_abstract)

        def fun_impl(*args, **params):
            impl = params["impl"]
            return impl(*args, **params)

        self.def_impl(fun_impl)

        def _mlir(ctx: mlir.LoweringRuleContext, *mlir_args, **params):
            lowering = mlir.lower_fun(self.impl, multiple_results=True)
            return lowering(ctx, *mlir_args, **params)

        mlir.register_lowering(self, _mlir)


def initial_style_bind(prim, **params):
    """Binds a primitive to a function call."""

    def bind(f):
        """Wraps a function to be bound to a primitive, keeping track of Pytree
        information."""

        def wrapped(*args, **kwargs):
            """Runs a function and binds it to a call primitive."""
            jaxpr, (flat_args, in_tree, out_tree) = stage(f)(*args, **kwargs)
            debug_info = jaxpr.jaxpr.debug_info

            def _impl(*args, **params):
                consts, args = jax_util.split_list(args, [params["num_consts"]])
                return jc.eval_jaxpr(jaxpr.jaxpr, consts, *args)

            def _abs_eval(*flat_avals, **params):
                return pe.abstract_eval_fun(
                    _impl,
                    *flat_avals,
                    debug_info=debug_info,
                    **params,
                )

            outs = prim.bind(
                *it.chain(jaxpr.literals, flat_args),
                abs_eval=params.get("abs_eval", _abs_eval),
                impl=_impl,
                in_tree=in_tree,
                out_tree=out_tree,
                num_consts=len(jaxpr.literals),
                **params,
            )
            return tree_util.tree_unflatten(out_tree(), outs)

        return wrapped

    return bind
