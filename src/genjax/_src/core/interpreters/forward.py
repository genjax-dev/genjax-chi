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
"""This module contains a transformation infrastructure based on custom
primitives, interpreters with stateful handlers and custom primitive handling
lookups."""

import abc
import copy
import functools
import itertools as it
from dataclasses import dataclass

import jax.core as jc
import jax.tree_util as jtu
from adevjax.hashable_dict import HashableDict
from adevjax.hashable_dict import hashable_dict
from adevjax.pytree import Pytree
from adevjax.staging import stage
from adevjax.typing import Any
from adevjax.typing import Bool
from adevjax.typing import Callable
from adevjax.typing import List
from adevjax.typing import Union
from adevjax.typing import Value
from adevjax.typing import typecheck
from jax import tree_util
from jax import util as jax_util
from jax._src import core as jax_core
from jax.extend import linear_util as lu
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters import partial_eval as pe

from genjax._src.core.datatypes.hashable_dict import HashableDict
from genjax._src.core.datatypes.hashable_dict import hashable_dict
from genjax._src.core.interpreters.staging import stage
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import Any
from genjax._src.core.typing import Bool
from genjax._src.core.typing import Callable
from genjax._src.core.typing import List
from genjax._src.core.typing import Union
from genjax._src.core.typing import Value


#########################
# Custom JAX primitives #
#########################


def batch_fun(fun: lu.WrappedFun, in_dims):
    fun, out_dims = batching.batch_subtrace(fun)
    return _batch_fun(fun, in_dims), out_dims


@lu.transformation
def _batch_fun(in_dims, *in_vals, **params):
    with jax_core.new_main(
        batching.BatchTrace, axis_name=jax_core.no_axis_name
    ) as main:
        out_vals = (
            yield (
                main,
                in_dims,
            )
            + in_vals,
            params,
        )
        del main
    yield out_vals


class FlatPrimitive(jax_core.Primitive):
    """Contains default implementations of transformations."""

    def __init__(self, name):
        super(FlatPrimitive, self).__init__(name)
        self.multiple_results = True

        def _abstract(*flat_avals, **params):
            return pe.abstract_eval_fun(self.impl, *flat_avals, **params)

        self.def_abstract_eval(_abstract)

        def _jvp(primals, tangents, **params):
            primals_out, tangents_out = ad.jvp(
                lu.wrap_init(self.impl, params)
            ).call_wrapped(primals, tangents)
            tangents_out = jax_util.safe_map(
                ad.recast_to_float0, primals_out, tangents_out
            )
            return primals_out, tangents_out

        ad.primitive_jvps[self] = _jvp

        def _batch(args, dims, **params):
            batched, out_dims = batch_fun(lu.wrap_init(self.impl, params), dims)
            return batched.call_wrapped(*args), out_dims()

        batching.primitive_batchers[self] = _batch

        def _mlir(c, *mlir_args, **params):
            lowering = mlir.lower_fun(self.impl, multiple_results=True)
            return lowering(c, *mlir_args, **params)

        mlir.register_lowering(self, _mlir)


class InitialStylePrimitive(FlatPrimitive):
    """Contains default implementations of transformations."""

    def __init__(self, name, batch_semantics=None):
        super().__init__(name)

        def fun_impl(*args, **params):
            consts, args = jax_util.split_list(args, [params["num_consts"]])
            return jax_core.eval_jaxpr(params["_jaxpr"], consts, *args)

        self.def_impl(fun_impl)

    def subcall(self, name):
        return InitialStylePrimitive(f"{self.name}/{name}")


def initial_style_bind(prim, **params):
    """Binds a primitive to a function call."""

    def bind(f):
        """Wraps a function to be bound to a primitive, keeping track of Pytree
        information."""

        def wrapped(*args, **kwargs):
            """Runs a function and binds it to a call primitive."""
            _jaxpr, (flat_args, in_tree, out_tree) = stage(f)(*args, **kwargs)
            outs = prim.bind(
                *it.chain(_jaxpr.literals, flat_args),
                _jaxpr=_jaxpr.jaxpr,
                in_tree=in_tree,
                out_tree=out_tree,
                num_consts=len(_jaxpr.literals),
                **params,
            )
            return tree_util.tree_unflatten(out_tree(), outs)

        return wrapped

    return bind


#######################
# Forward interpreter #
#######################

VarOrLiteral = Union[jc.Var, jc.Literal]


@dataclass
class Environment(Pytree):
    """Keeps track of variables and their values during propagation."""

    env: HashableDict[jc.Var, Value]

    def flatten(self):
        return (self.env,), ()

    @classmethod
    def new(cls):
        return Environment(hashable_dict())

    def read(self, var: VarOrLiteral) -> Value:
        if isinstance(var, jc.Literal):
            return var.val
        else:
            return self.env.get(var.count)

    def write(self, var: VarOrLiteral, cell: Value) -> Value:
        if isinstance(var, jc.Literal):
            return cell
        cur_cell = self.read(var)
        if isinstance(var, jc.DropVar):
            return cur_cell
        self.env[var.count] = cell
        return self.env[var.count]

    def __getitem__(self, var: VarOrLiteral) -> Value:
        return self.read(var)

    def __setitem__(self, key, val):
        raise ValueError(
            "Environments do not support __setitem__. Please use the "
            "`write` method instead."
        )

    def __contains__(self, var: VarOrLiteral):
        if isinstance(var, jc.Literal):
            return True
        return var in self.env

    def copy(self):
        return copy.copy(self)


@dataclass
class StatefulHandler(Pytree):
    @abc.abstractmethod
    def handles(self, primitive: jc.Primitive) -> Bool:
        pass

    @abc.abstractmethod
    def dispatch(
        self,
        primitive: jc.Primitive,
        *args,
        **kwargs,
    ) -> List[Any]:
        pass


@dataclass
class ForwardInterpreter(Pytree):
    def flatten(self):
        return (), ()

    def _eval_jaxpr_forward(
        self,
        stateful_handler,
        _jaxpr: jc.Jaxpr,
        consts: List[Value],
        args: List[Value],
    ):
        env = Environment.new()
        jax_util.safe_map(env.write, _jaxpr.constvars, consts)
        jax_util.safe_map(env.write, _jaxpr.invars, args)
        for eqn in _jaxpr.eqns:
            invals = jax_util.safe_map(env.read, eqn.invars)
            subfuns, params = eqn.primitive.get_bind_params(eqn.params)
            args = subfuns + invals
            if stateful_handler.handles(eqn.primitive):
                outvals = stateful_handler.dispatch(eqn.primitive, *args, **params)
            else:
                outvals = eqn.primitive.bind(*args, **params)
            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            jax_util.safe_map(env.write, eqn.outvars, outvals)

        return jax_util.safe_map(env.read, _jaxpr.outvars)

    def run_interpreter(self, stateful_handler, fn, *args, **kwargs):
        def _inner(*args):
            return fn(*args, **kwargs)

        _closed_jaxpr, (flat_args, _, out_tree) = stage(_inner)(*args)
        _jaxpr, consts = _closed_jaxpr.jaxpr, _closed_jaxpr.literals
        flat_out = self._eval_jaxpr_forward(
            stateful_handler,
            _jaxpr,
            consts,
            flat_args,
        )
        return jtu.tree_unflatten(out_tree(), flat_out)


@typecheck
def forward(f: Callable):
    @functools.wraps(f)
    @typecheck
    def wrapped(stateful_handler: StatefulHandler, *args):
        interpreter = ForwardInterpreter()
        return interpreter.run_interpreter(
            stateful_handler,
            f,
            *args,
        )

    return wrapped
