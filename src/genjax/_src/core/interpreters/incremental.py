# Copyright 2022 The MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module supports incremental computation using a form of JVP-inspired
computation with a type of generalized tangent values (e.g. `ChangeTangent`
below).

Incremental computation is currently a concern of Gen's `update` GFI method - and can be utilized _as a runtime performance optimization_ for computing the weight (and changes to `Trace` instances) which `update` computes.

*Change types*

By default, `genjax` provides two types of `ChangeTangent`:

* `NoChange` - indicating that a value has not changed.
* `UnknownChange` - indicating that a value has changed, without further information about the change.

`ChangeTangents` are provided along with primal values into `Diff` instances. The generative function `update` interface expects tuples of `Pytree` instances whose leaves are `Diff` instances (`argdiffs`).
"""

# TODO: Think about when tangents don't share the same Pytree shape as primals.

import abc
import dataclasses
import functools

import jax.core as jc
import jax.tree_util as jtu
from jax import util as jax_util

from genjax._src.core.datatypes.hashable_dict import HashableDict
from genjax._src.core.interpreters.forward import Environment
from genjax._src.core.interpreters.forward import StatefulHandler
from genjax._src.core.interpreters.staging import stage
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import Any
from genjax._src.core.typing import Callable
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import List
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import Value
from genjax._src.core.typing import static_check_is_concrete
from genjax._src.core.typing import typecheck


#######################################
# Change type lattice and propagation #
#######################################

###################
# Change tangents #
###################


@dataclasses.dataclass
class ChangeTangent(Pytree):
    @abc.abstractmethod
    def should_flatten(self):
        pass

    def widen(self):
        return UnknownChange


# These two classes are the bottom and top of the change lattice.
# Unknown change represents complete lack of information about
# the change to a value.
#
# No change represents complete information about the change to a value
# (namely, that it is has not changed).


@dataclasses.dataclass
class _UnknownChange(ChangeTangent):
    def flatten(self):
        return (), ()

    def should_flatten(self):
        return False


UnknownChange = _UnknownChange()


@dataclasses.dataclass
class _NoChange(ChangeTangent):
    def flatten(self):
        return (), ()

    def should_flatten(self):
        return False


NoChange = _NoChange()


@dataclasses.dataclass
class IntChange(ChangeTangent):
    dv: IntArray

    def flatten(self):
        return (self.dv,), ()

    def should_flatten(self):
        return True


@dataclasses.dataclass
class StaticIntChange(ChangeTangent):
    dv: IntArray

    def flatten(self):
        return (), (self.dv,)

    @classmethod
    def new(cls, dv):
        assert static_check_is_concrete(dv)
        return StaticIntChange(dv)

    def should_flatten(self):
        return True


def static_check_is_change_tangent(v):
    return isinstance(v, ChangeTangent)


#############################
# Diffs (generalized duals) #
#############################


@dataclasses.dataclass
class Diff(Pytree):
    primal: Any
    tangent: Any

    def flatten(self):
        return (self.primal, self.tangent), ()

    @classmethod
    def new(cls, primal, tangent):
        assert not isinstance(primal, Diff)
        static_check_is_change_tangent(tangent)
        return Diff(primal, tangent)

    def get_primal(self):
        return self.primal

    def get_tangent(self):
        return self.tangent

    def unpack(self):
        return self.primal, self.tangent


def static_check_is_diff(v):
    return isinstance(v, Diff)


def static_check_no_change(v):
    def _inner(v):
        if static_check_is_change_tangent(v):
            return isinstance(v, _NoChange)
        else:
            return True

    return all(
        jtu.tree_leaves(jtu.tree_map(_inner, v, is_leaf=static_check_is_change_tangent))
    )


def tree_diff_primal(v):
    def _inner(v):
        if static_check_is_diff(v):
            return v.get_primal()
        else:
            return v

    return jtu.tree_map(lambda v: _inner(v), v, is_leaf=static_check_is_diff)


def tree_diff_tangent(v):
    def _inner(v):
        if static_check_is_diff(v):
            return v.get_tangent()
        else:
            return v

    return jtu.tree_map(lambda v: _inner(v), v, is_leaf=static_check_is_diff)


def tree_diff_unpack_leaves(v):
    primals = tree_diff_primal(v)
    tangents = tree_diff_tangent(v)
    return jtu.tree_leaves(primals), jtu.tree_leaves(tangents)


def static_check_tree_leaves_diff(v):
    def _inner(v):
        if static_check_is_diff(v):
            return True
        else:
            return False

    return all(
        jtu.tree_leaves(
            jtu.tree_map(_inner, v, is_leaf=static_check_is_diff),
        )
    )


def tree_diff(tree, tangent_tree):
    return jtu.tree_map(
        lambda p, t: diff(p, t),
        tree,
        tangent_tree,
    )


def tree_diff_no_change(tree):
    tangent_tree = jtu.tree_map(lambda _: NoChange, tree)
    return tree_diff(tree, tangent_tree)


def tree_diff_unknown_change(tree):
    tangent_tree = jtu.tree_map(lambda _: UnknownChange, tree)
    return tree_diff(tree, tangent_tree)


#################################
# Generalized tangent transform #
#################################

# TODO: currently, only supports our default lattice
# (`Change` and `NoChange`)
def default_propagation_rule(prim, *args, **params):
    check = static_check_no_change(args)
    args = tree_diff_primal(args)
    outval = prim.bind(*args, **params)
    if check:
        return outval
    else:
        return tree_diff_unknown_change(outval)


@dataclasses.dataclass
class IncrementalInterpreter(Pytree):
    custom_rules: HashableDict[jc.Primitive, Callable]

    def flatten(self):
        return (), (self.custom_rules,)

    def _eval_jaxpr_forward(
        self,
        stateful_handler,
        jaxpr: jc.Jaxpr,
        consts: List[Value],
        primals: List[Value],
        tangents: List[ChangeTangent],
    ):
        dual_env = Environment.new()
        jax_util.safe_map(dual_env.write, jaxpr.constvars, tree_diff_no_change(consts))
        jax_util.safe_map(dual_env.write, jaxpr.invars, tree_diff(primals, tangents))
        for eqn in jaxpr.eqns:
            induals = jax_util.safe_map(dual_env.read, eqn.invars)
            subfuns, params = eqn.primitive.get_bind_params(eqn.params)
            args = subfuns + induals
            if stateful_handler.handles(eqn.primitive):
                outduals = stateful_handler.dispatch(eqn.primitive, *args, **params)
            else:
                outduals = default_propagation_rule(eqn.primitive, *args, **params)
            if not eqn.primitive.multiple_results:
                outduals = [outduals]
            jax_util.safe_map(dual_env.write, eqn.outvars, outduals)

        return jax_util.safe_map(dual_env.read, jaxpr.outvars)

    def run_interpreter(self, stateful_handler, fn, primals, tangents, **kwargs):
        def _inner(*args):
            return fn(*args, **kwargs)

        closed_jaxpr, (flat_primals, _, out_tree) = stage(_inner)(*primals)
        flat_tangents = jtu.tree_leaves(
            tangents, is_leaf=lambda v: isinstance(v, ChangeTangent)
        )
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        flat_out = self._eval_jaxpr_forward(
            stateful_handler,
            jaxpr,
            consts,
            flat_primals,
            flat_tangents,
        )
        return jtu.tree_unflatten(out_tree(), flat_out)


@typecheck
def incremental(f: Callable):
    @functools.wraps(f)
    @typecheck
    def wrapped(
        stateful_handler: StatefulHandler,
        primals: Tuple,
        tangents: Tuple,
    ):
        interpreter = IncrementalInterpreter.new()
        return interpreter.run_interpreter(
            stateful_handler,
            f,
            primals,
            tangents,
        )

    return wrapped


##############
# Shorthands #
##############

diff = Diff.new
