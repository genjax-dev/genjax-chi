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

import abc
import functools
import itertools as it

import jax.core as jc
import jax.tree_util as jtu
from jax import tree_util
from jax import util as jax_util
from jax.extend import linear_util as lu
from jax.interpreters import batching, mlir
from jax.interpreters import partial_eval as pe

from genjax._src.core.interpreters.staging import WrappedFunWithAux, stage
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import Any, Callable

#########################
# Custom JAX primitives #
#########################

# Wrapper to assign a correct type.
batch_subtrace: Callable[[lu.WrappedFun], WrappedFunWithAux] = batching.batch_subtrace  # pyright: ignore[reportAssignmentType]


@lu.transformation
def __batch_fun(in_dims, *in_vals, **params):
    with jc.new_main(batching.BatchTrace, axis_name=jc.no_axis_name) as main:
        out_vals = yield (
            (main, in_dims, *in_vals),
            params,
        )
        del main
    yield out_vals


_batch_fun: Callable[[lu.WrappedFun, Any], lu.WrappedFun] = __batch_fun  # pyright: ignore[reportAssignmentType]


def batch_fun(fun: lu.WrappedFun, in_dims) -> WrappedFunWithAux:
    fun, out_dims = batch_subtrace(fun)
    return _batch_fun(fun, in_dims), out_dims


class FlatPrimitive(jc.Primitive):
    """Contains default implementations of transformations."""

    def __init__(self, name):
        super(FlatPrimitive, self).__init__(name)
        self.multiple_results = True

        def _abstract(*flat_avals, **params):
            return pe.abstract_eval_fun(self.impl, *flat_avals, **params)

        self.def_abstract_eval(_abstract)

        def _batch(args, dims, **params):
            batched, out_dims = batch_fun(lu.wrap_init(self.impl, params), dims)
            return batched.call_wrapped(*args), out_dims()

        batching.primitive_batchers[self] = _batch

        def _mlir(ctx: mlir.LoweringRuleContext, *mlir_args, **params):
            lowering = mlir.lower_fun(self.impl, multiple_results=True)
            return lowering(ctx, *mlir_args, **params)

        mlir.register_lowering(self, _mlir)


class InitialStylePrimitive(FlatPrimitive):
    """Contains default implementations of transformations."""

    def __init__(self, name):
        super().__init__(name)

        def fun_impl(*args, **params):
            consts, args = jax_util.split_list(args, [params["num_consts"]])
            return jc.eval_jaxpr(params["_jaxpr"], consts, *args)

        self.def_impl(fun_impl)


def initial_style_bind(prim, **params):
    """Binds a primitive to a function call."""

    def bind(f):
        """Wraps a function to be bound to a primitive, keeping track of Pytree
        information."""

        def wrapped(*args, **kwargs):
            """Runs a function and binds it to a call primitive."""
            jaxpr, (flat_args, in_tree, out_tree) = stage(f)(*args, **kwargs)
            outs = prim.bind(
                *it.chain(jaxpr.literals, flat_args),
                _jaxpr=jaxpr.jaxpr,
                in_tree=in_tree,
                out_tree=out_tree,
                num_consts=len(jaxpr.literals),
                **params,
            )
            return tree_util.tree_unflatten(out_tree(), outs)

        return wrapped

    return bind


#######################
# Forward interpreter #
#######################

VarOrLiteral = jc.Var | jc.Literal


@Pytree.dataclass
class Environment(Pytree):
    """Keeps track of variables and their values during propagation."""

    env: dict[int, Any] = Pytree.field(default_factory=dict)

    def read(self, var: VarOrLiteral) -> Any:
        if isinstance(var, jc.Literal):
            return var.val
        else:
            v = self.env.get(var.count)
            if v is None:
                raise ValueError(
                    f"Unbound variable in interpreter environment at count {var.count}:\nEnvironment keys (count): {list(self.env.keys())}"
                )
            return v

    def get(self, var: VarOrLiteral) -> Any:
        if isinstance(var, jc.Literal):
            return var.val
        else:
            return self.env.get(var.count)

    def write(self, var: VarOrLiteral, cell: Any) -> Any:
        if isinstance(var, jc.Literal):
            return cell
        cur_cell = self.get(var)
        if isinstance(var, jc.DropVar):
            return cur_cell
        self.env[var.count] = cell
        return self.env[var.count]

    def __getitem__(self, var: VarOrLiteral) -> Any:
        return self.read(var)

    def __setitem__(self, key, val):
        raise ValueError(
            "Environments do not support __setitem__. Please use the "
            "`write` method instead."
        )

    def __contains__(self, var: VarOrLiteral):
        if isinstance(var, jc.Literal):
            return True
        return var.count in self.env

    def copy(self):
        keys = list(self.env.keys())
        return Environment({k: self.env[k] for k in keys})


class StatefulHandler:
    @abc.abstractmethod
    def handles(self, primitive: jc.Primitive) -> bool:
        pass

    @abc.abstractmethod
    def dispatch(
        self,
        primitive: jc.Primitive,
        *args,
        **kwargs,
    ) -> list[Any]:
        pass


@Pytree.dataclass
class ForwardInterpreter(Pytree):
    def _eval_jaxpr_forward(
        self,
        stateful_handler,
        _jaxpr: jc.Jaxpr,
        consts: list[Any],
        args: list[Any],
    ):
        env = Environment()
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

        closed_jaxpr, (flat_args, _, out_tree) = stage(_inner)(*args)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        flat_out = self._eval_jaxpr_forward(
            stateful_handler,
            jaxpr,
            consts,
            flat_args,
        )
        return jtu.tree_unflatten(out_tree(), flat_out)


def forward(f: Callable[..., Any]):
    @functools.wraps(f)
    def wrapped(stateful_handler: StatefulHandler, *args):
        interpreter = ForwardInterpreter()
        return interpreter.run_interpreter(
            stateful_handler,
            f,
            *args,
        )

    return wrapped
