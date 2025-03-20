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
import jax.extend.linear_util as lu
from jax import tree_util
from jax import util as jax_util

# TODO: stupid.
from jax._src.interpreters.batching import AxisData
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir
from jax.interpreters import partial_eval as pe

from genjax._src.core.compiler.staging import stage
from genjax._src.core.typing import Any

#########################
# Custom JAX primitives #
#########################


class InitialStylePrimitive(Primitive):
    """Contains default implementations of transformations."""

    def __init__(self, name):
        super(InitialStylePrimitive, self).__init__(name)
        self.multiple_results = True

        def impl(*flat_args, **params):
            return params["impl"](*flat_args, **params)

        def abstract(*flat_avals, **params):
            return params["abstract"](*flat_avals, **params)

        def jvp(
            flat_primals: tuple[Any, ...] | list[Any],
            flat_tangents: tuple[Any, ...] | list[Any],
            **params,
        ) -> tuple[list[Any], list[Any]]:
            return params["jvp"](flat_primals, flat_tangents, **params)

        def batch(flat_vector_args: tuple[Any, ...] | list[Any], dim, **params):
            return params["batch"](flat_vector_args, dim, **params)

        def lowering(ctx: mlir.LoweringRuleContext, *mlir_args, **params):
            lowering = mlir.lower_fun(self.impl, multiple_results=True)
            return lowering(ctx, *mlir_args, **params)

        # Store for elaboration.
        self.impl = impl
        self.jvp = jvp
        self.abstract = abstract
        self.batch = batch
        self.lowering = lowering

        self.def_impl(impl)
        ad.primitive_jvps[self] = jvp
        self.def_abstract_eval(abstract)
        batching.primitive_batchers[self] = batch
        mlir.register_lowering(self, lowering)


class ElaboratedPrimitive(Primitive):
    """
    An `ElaboratedPrimitive` is a primitive which wraps an underlying primitive, but is elaborated with additional parameters that only show up in the pretty printing of a `Jaxpr`.
    In addition, `ElaboratedPrimitive` instances hide excess metadata in
    pretty printing.
    """

    def __init__(self, prim: InitialStylePrimitive, **params):
        super(ElaboratedPrimitive, self).__init__(prim.name)
        self.prim = prim
        self.multiple_results = self.prim.multiple_results
        self.params = params

        def impl(*args, **_):
            return self.prim.impl(*args, **self.params)

        def abstract(*args, **_):
            return self.prim.abstract(*args, **self.params)

        def jvp(*args, **_):
            return self.prim.jvp(*args, **self.params)

        def batch(*args, **_):
            return self.prim.batch(*args, **self.params)

        def lowering(*args, **_):
            return self.prim.lowering(*args, **self.params)

        self.def_impl(impl)
        ad.primitive_jvps[self] = jvp
        self.def_abstract_eval(abstract)
        batching.primitive_batchers[self] = batch
        mlir.register_lowering(self, lowering)

    def get_bind_params(self, params):
        return [], params

    @classmethod
    def unwrap(cls, v):
        return (v.prim, v.params) if isinstance(v, ElaboratedPrimitive) else (v, {})

    @classmethod
    def check(cls, primitive, other):
        if isinstance(primitive, ElaboratedPrimitive):
            return primitive.prim == other
        else:
            return primitive == other

    @classmethod
    def rebind(cls, primitive, *args, **params):
        if isinstance(primitive, ElaboratedPrimitive):
            return primitive.bind(*args, **params)
        else:
            return primitive.bind(*args, **params)


def batch_fun(fun: lu.WrappedFun, axis_data, in_dims):
    tag = jc.TraceTag()
    in_dims = in_dims() if callable(in_dims) else in_dims
    batched, out_dims = batching.batch_subtrace(fun, tag, axis_data, in_dims)
    return batched, out_dims


def initial_style_bind(prim, **params):
    """Binds a primitive to a function call."""

    def bind(f, **elaboration_kwargs):
        """Wraps a function to be bound to a primitive, keeping track of Pytree
        information."""

        def wrapped(*args, **kwargs):
            """Runs a function and binds it to an `ElaboratedPrimitive`
            primitive, hiding the implementation details of the eval
            (impl) rule, abstract rule, batch rule, and jvp rule."""
            jaxpr, (flat_args, in_tree, out_tree) = stage(f)(*args, **kwargs)
            debug_info = jaxpr.jaxpr.debug_info

            def impl(*flat_args, **params) -> list[Any]:
                consts, flat_args = jax_util.split_list(
                    flat_args, [params["num_consts"]]
                )
                return jc.eval_jaxpr(jaxpr.jaxpr, consts, *flat_args)

            def abstract(*flat_avals, **params):
                return pe.abstract_eval_fun(
                    impl,
                    *flat_avals,
                    debug_info=debug_info,
                    **params,
                )

            def batch(flat_vector_args: tuple[Any, ...] | list[Any], dims, **params):
                axis_data = AxisData(None, None, None, None)
                batched, out_dims = batch_fun(
                    lu.wrap_init(impl, params, debug_info=debug_info),
                    axis_data,
                    dims,
                )
                return batched.call_wrapped(*flat_vector_args), out_dims()

            def jvp(
                flat_primals: tuple[Any, ...] | list[Any],
                flat_tangents: tuple[Any, ...] | list[Any],
                **params,
            ) -> tuple[list[Any], list[Any]]:
                primals_out, tangents_out = ad.jvp(
                    lu.wrap_init(impl, params, debug_info=debug_info)
                ).call_wrapped(flat_primals, flat_tangents)

                # We always normalize back to list.
                return list(primals_out), list(tangents_out)

            if "impl" in params:
                impl = params["impl"]
                params.pop("impl")

            if "abstract" in params:
                abstract = params["abstract"]
                params.pop("abstract")

            if "batch" in params:
                batch = params["batch"]
                params.pop("batch")

            if "jvp" in params:
                jvp = params["jvp"]
                params.pop("jvp")

            elaborated_prim = ElaboratedPrimitive(
                prim,
                impl=impl,
                abstract=abstract,
                batch=batch,
                jvp=jvp,
                in_tree=in_tree,
                out_tree=out_tree,
                num_consts=len(jaxpr.literals),
                **params,
            )
            outs = elaborated_prim.bind(
                *it.chain(jaxpr.literals, flat_args),
                **elaboration_kwargs,
            )
            return tree_util.tree_unflatten(out_tree(), outs)

        return wrapped

    return bind
