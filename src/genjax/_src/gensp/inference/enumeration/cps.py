# Copyright 2023 MIT Probabilistic Computing Project
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

import copy
import dataclasses
import functools

import jax.core as jc
import jax.tree_util as jtu
from jax import api_util
from jax import util as jax_util
from jax.extend import linear_util as lu

from genjax._src.core.datatypes.hashable_dict import HashableDict, hashable_dict
from genjax._src.core.interpreters.forward import initial_style_bind
from genjax._src.core.interpreters.staging import stage
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import List, Union, Value

###################
# CPS interpreter #
###################

VarOrLiteral = Union[jc.Var, jc.Literal]


@dataclasses.dataclass
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


@dataclasses.dataclass
class CPSInterpreter(Pytree):
    def flatten(self):
        return (), ()

    def _eval_jaxpr_cps(
        self,
        jaxpr: jc.Jaxpr,
        consts: List[Value],
        args: List[Value],
        allowlist: List[jc.Primitive],
    ):
        env = Environment.new()
        jax_util.safe_map(env.write, jaxpr.constvars, consts)
        jax_util.safe_map(env.write, jaxpr.invars, args)

        def eval_jaxpr_iterate(eqns, env, invars, args):
            jax_util.safe_map(env.write, invars, args)

            for eqn_idx, eqn in list(enumerate(eqns)):
                in_vals = jax_util.safe_map(env.read, eqn.invars)
                subfuns, params = eqn.primitive.get_bind_params(eqn.params)
                args = subfuns + in_vals

                if eqn.primitive in allowlist:
                    # Create continuation.
                    def kont(eqns, eqn_idx, outvars, env, *args):
                        return eval_jaxpr_iterate(
                            eqns[eqn_idx + 1 :], env, outvars, [*args]
                        )

                    def _binder(env, tree_args):
                        flat_args = jtu.tree_leaves(tree_args)
                        return eqn.primitive.impl(*flat_args, **params)

                    in_tree = params["in_tree"]
                    tree_args = jtu.tree_unflatten(in_tree, args)

                    # Bind the continuation as a static parameter into an invocation
                    # of a primitive that "wants" a continuation.
                    outvals = initial_style_bind(
                        eqn.primitive,
                        kont=functools.partial(kont, eqns, eqn_idx, eqn.outvars),
                    )(_binder)(env.copy(), tree_args)

                # Otherwise, fall through -- we just use the default bind.
                else:
                    outvals = eqn.primitive.bind(*args, **params)

                jax_util.safe_map(
                    env.write,
                    eqn.outvars,
                    jtu.tree_leaves(outvals),
                )

            return jax_util.safe_map(env.read, jaxpr.outvars)

        return eval_jaxpr_iterate(jaxpr.eqns, env, jaxpr.invars, args)

    def run_interpreter(self, kont, allowlist, fn, *args, **kwargs):
        def _inner(*args, **kwargs):
            return kont(fn(*args, **kwargs))

        closed_jaxpr, (flat_args, _, out_tree) = stage(_inner)(*args, **kwargs)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        flat_out = self._eval_jaxpr_cps(jaxpr, consts, flat_args, allowlist)
        if flat_out:
            return jtu.tree_unflatten(out_tree(), flat_out)


Cont = CPSInterpreter


def cps(f, kont, allowlist):
    # Runs the interpreter.
    def _run_interpreter(*args):
        interpreter = Cont()
        return interpreter.run_interpreter(kont, allowlist, f, *args)

    # Propagates tracer values through running the interpreter.
    @functools.wraps(f)
    def wrapped(*args):
        fun = lu.wrap_init(_run_interpreter)
        flat_args, args_tree = jtu.tree_flatten(args)
        flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, args_tree)
        retvals = flat_fun.call_wrapped(*flat_args)
        out_tree_def = out_tree()
        return jtu.tree_unflatten(out_tree_def, retvals)

    return wrapped
