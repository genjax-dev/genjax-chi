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

"""This module contains a continuation passing style interpreter, along with
ways to parametrize it using user-customized rules for primitives."""

import abc
import copy
import dataclasses
from contextlib import contextmanager
from typing import Dict
from typing import List
from typing import Type
from typing import Union

import jax.core as jc
import jax.tree_util as jtu
from jax.util import safe_map

from genjax._src.core.pytree import Pytree


VarOrLiteral = Union[jc.Var, jc.Literal]


# NOTE: An abstract base class for the object values which the
# interpreter operates on.
# Useful because the interpreter often wants to assume that
# values stored in the environment are either all of the same
# cell type - the `new` interface allows the interpreter to lift
# values to the cell type after lookup in the environment.
class Cell(Pytree):
    """Base interface for objects used during interpretation."""

    @classmethod
    def new(cls, value):
        """Creates a new instance of a Cell from a value."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_val(self):
        pass


def static_map_unwrap(incells):
    def _inner(v):
        if isinstance(v, Cell):
            return v.get_val()
        else:
            return v

    return [_inner(v) for v in incells]


def flatmap_outcells(cell_type, v, **kwargs):
    return jtu.tree_map(lambda v: cell_type.new(v, **kwargs), jtu.tree_leaves(v))


class Environment:
    """Keeps track of variables and their values during propagation."""

    def __init__(self, cell_type):
        self.cell_type = cell_type
        self.env: Dict[jc.Var, Cell] = {}

    def read(self, var: VarOrLiteral) -> Cell:
        if isinstance(var, jc.Literal):
            return self.cell_type.new(var.val)  # lift to Cell.
        else:
            return self.env.get(var)

    def write(self, var: VarOrLiteral, cell: Cell) -> Cell:
        if isinstance(var, jc.Literal):
            return cell
        cur_cell = self.read(var)
        if isinstance(var, jc.DropVar):
            return cur_cell
        self.env[var] = cell
        return self.env[var]

    def __getitem__(self, var: VarOrLiteral) -> Cell:
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
class Handler(Pytree):
    """
    A handler dispatchs a `jax.core.Primitive` - and must provide
    a `Callable` with signature `def (name_of_primitive)(continuation, *args)`
    where `*args` must match the `jax.core.Primitive` declaration signature.
    """

    handles: List[jc.Primitive]

    # Allows a fallback handling rule to be applied,
    # for primitives which we don't to provide any special
    # interpretation to.
    def fallback(self, cell_type: Type[Cell], prim: jc.Primitive, args, cont, **params):
        # Strip to JAX values.
        args = static_map_unwrap(args)
        v = prim.bind(*args, **params)
        # Lift back to `Cell` type.
        v = flatmap_outcells(cell_type, v)
        return cont(*v)

    def handle(
        self, cell_type: Type[Cell], prim: jc.Primitive, args, kwargs, params, cont
    ):
        # NOTE: If the prim is registered with the handler,
        # we attempt to handle.
        if prim in self.handles:
            try:
                callable = getattr(self, repr(prim))

            # TODO: provide a better exception.
            except Exception as e:
                raise e
            return callable(cell_type, prim, args, cont, **kwargs)
        else:
            return self.fallback(cell_type, prim, args, cont, **params)


@dataclasses.dataclass
class Interpreter:
    cell_type: Type[Cell]
    handler: Handler

    # This produces an instance of `Interpreter`
    # as a context manager - to allow us to control stack traces,
    # if required.
    @classmethod
    @contextmanager
    def new(cls, cell_type: Type[Cell], handler: Handler):
        try:
            yield Interpreter(cell_type, handler)
        except Exception as e:
            raise e

    # NOTE: Understanding a little bit about continuation passing
    # style will be useful to understand this interpreter.
    def _eval_jaxpr_continuation(
        self,
        jaxpr: jc.Jaxpr,
        consts: List[Cell],
        args: List[Cell],
    ):
        env = Environment(self.cell_type)
        safe_map(env.write, jaxpr.constvars, consts)

        def eval_jaxpr_recurse(eqns, env, invars, args):
            # The rule could call the continuation multiple times so we
            # we need this function to be somewhat pure.
            # We copy `env` to ensure it isn't mutated.
            env = env.copy()

            # Bind arguments to invars.
            safe_map(env.write, invars, args)

            if eqns:
                eqn = eqns[0]
                kwargs = eqn.params
                in_vals = safe_map(env.read, eqn.invars)
                in_vals = list(in_vals)
                subfuns, params = eqn.primitive.get_bind_params(eqn.params)
                args = subfuns + in_vals

                # Create a continuation to pass to the rule.
                def continuation(*args):
                    return eval_jaxpr_recurse(
                        eqns[1:],
                        env,
                        eqn.outvars,
                        [*args],
                    )

                # Pass all the information over to the handler,
                # which gets to choose how to interpret the primitive.
                return self.handler.handle(
                    self.cell_type,
                    eqn.primitive,
                    args,
                    kwargs,
                    params,
                    cont=continuation,
                )

            else:

                # NOTE: This is the final return behavior of each
                # continuation call -- e.g. think about the
                # branches out of the recursive interpretation
                # process -- in a rule,
                # you can return directly, or you can call the
                # continuation -- the continuation runs the interpreter
                # on rest of the Jaxpr computation, stepping forward,
                # selecting a rule, etc.
                return safe_map(env.read, jaxpr.outvars)

        return eval_jaxpr_recurse(jaxpr.eqns, env, jaxpr.invars, args)

    def __call__(self, jaxpr, consts, args):
        return self._eval_jaxpr_continuation(jaxpr, consts, args)
