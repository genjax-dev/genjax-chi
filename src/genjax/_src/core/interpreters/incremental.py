# Copyright 2024 The MIT Probabilistic Computing Project
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
"""This module supports incremental computation using a form of JVP-inspired computation
with a type of generalized tangent values (e.g. `ChangeTangent` below).

Incremental computation is currently a concern of Gen's `update` GFI method - and can be utilized _as a runtime performance optimization_ for computing the weight (and changes to `Trace` instances) which `update` computes.

*Change types*

By default, `genjax` provides two types of `ChangeTangent`:

* `NoChange` - indicating that a value has not changed.
* `UnknownChange` - indicating that a value has changed, without further information about the change.

`ChangeTangents` are provided along with primal values into `Diff` instances. The generative function `update` interface expects tuples of `Pytree` instances whose leaves are `Diff` instances (`argdiffs`).
"""

# TODO: Think about when tangents don't share the same Pytree shape as primals.

import functools

import jax.core as jc
import jax.tree_util as jtu
from jax import util as jax_util

from genjax._src.core.interpreters.forward import Environment, StatefulHandler
from genjax._src.core.interpreters.staging import stage
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    Generic,
    TypeVar,
    Value,
)

R = TypeVar("R")

#######################################
# Change type lattice and propagation #
#######################################

###################
# Change tangents #
###################


class ChangeTangent(Pytree):
    pass


# These two classes are the bottom and top of the change lattice.
# Unknown change represents complete lack of information about
# the change to a value.
#
# No change represents complete information about the change to a value
# (namely, that it is has not changed).


@Pytree.dataclass
class _UnknownChange(ChangeTangent):
    pass


UnknownChange = _UnknownChange()


@Pytree.dataclass
class _NoChange(ChangeTangent):
    pass


NoChange = _NoChange()


def static_check_is_change_tangent(v):
    return isinstance(v, ChangeTangent)


#############################
# Diffs (generalized duals) #
#############################


@Pytree.dataclass
class Diff(Generic[R], Pytree):
    primal: R
    tangent: R

    def __post_init__(self):
        assert not isinstance(self.primal, Diff)
        static_check_is_change_tangent(self.tangent)

    def get_primal(self):
        return self.primal

    def get_tangent(self):
        return self.tangent

    #############
    # Utilities #
    #############

    @staticmethod
    def tree_diff(tree, tangent_tree):
        return jtu.tree_map(
            lambda p, t: Diff(p, t),
            tree,
            tangent_tree,
        )

    @staticmethod
    def tree_diff_no_change(tree):
        tangent_tree = jtu.tree_map(lambda _: NoChange, tree)
        return Diff.tree_diff(tree, tangent_tree)

    @staticmethod
    def no_change(tree):
        return Diff.tree_diff_no_change(tree)

    @staticmethod
    def tree_diff_unknown_change(tree):
        primal_tree = Diff.tree_primal(tree)
        tangent_tree = jtu.tree_map(lambda _: UnknownChange, primal_tree)
        return Diff.tree_diff(primal_tree, tangent_tree)

    @staticmethod
    def unknown_change(tree):
        return Diff.tree_diff_unknown_change(tree)

    @staticmethod
    def tree_primal(v):
        def _inner(v):
            if Diff.static_check_is_diff(v):
                return v.get_primal()
            else:
                return v

        return jtu.tree_map(_inner, v, is_leaf=Diff.static_check_is_diff)

    @staticmethod
    def tree_tangent(v):
        def _inner(v):
            if Diff.static_check_is_diff(v):
                return v.get_tangent()
            else:
                return v

        return jtu.tree_map(_inner, v, is_leaf=Diff.static_check_is_diff)

    #################
    # Static checks #
    #################

    @staticmethod
    def static_check_is_diff(v):
        return isinstance(v, Diff)

    @staticmethod
    def static_check_tree_diff(v):
        return all(
            map(
                lambda v: isinstance(v, Diff),
                jtu.tree_leaves(v, is_leaf=Diff.static_check_is_diff),
            )
        )

    @staticmethod
    def static_check_no_change(v):
        def _inner(v):
            if static_check_is_change_tangent(v):
                return isinstance(v, _NoChange)
            else:
                return True

        return all(
            jtu.tree_leaves(
                jtu.tree_map(_inner, v, is_leaf=static_check_is_change_tangent)
            )
        )


#################################
# Generalized tangent transform #
#################################


# TODO: currently, only supports our default lattice
# (`Change` and `NoChange`)
def default_propagation_rule(prim, *args, **_params):
    check = Diff.static_check_no_change(args)
    args = Diff.tree_primal(args)
    outval = prim.bind(*args, **_params)
    if check:
        return Diff.tree_diff_no_change(outval)
    else:
        return Diff.tree_diff_unknown_change(outval)


@Pytree.dataclass
class IncrementalInterpreter(Pytree):
    custom_rules: dict[jc.Primitive, Callable[..., Any]] = Pytree.static(
        default_factory=dict
    )

    def _eval_jaxpr_forward(
        self,
        _stateful_handler,
        _jaxpr: jc.Jaxpr,
        consts: list[Value],
        primals: list[Value],
        tangents: list[ChangeTangent],
    ):
        dual_env = Environment()
        jax_util.safe_map(
            dual_env.write, _jaxpr.constvars, Diff.tree_diff_no_change(consts)
        )
        jax_util.safe_map(
            dual_env.write, _jaxpr.invars, Diff.tree_diff(primals, tangents)
        )
        for _eqn in _jaxpr.eqns:
            induals = jax_util.safe_map(dual_env.read, _eqn.invars)
            # TODO: why isn't this handled automatically by the environment,
            # especially the line above with _jaxpr.constvars?
            induals = [
                Diff(v, NoChange) if not isinstance(v, Diff) else v for v in induals
            ]
            subfuns, _params = _eqn.primitive.get_bind_params(_eqn.params)
            args = subfuns + induals
            if _stateful_handler and _stateful_handler.handles(_eqn.primitive):
                outduals = _stateful_handler.dispatch(_eqn.primitive, *args, **_params)
            else:
                outduals = default_propagation_rule(_eqn.primitive, *args, **_params)
            if not _eqn.primitive.multiple_results:
                outduals = [outduals]
            jax_util.safe_map(dual_env.write, _eqn.outvars, outduals)

        return jax_util.safe_map(dual_env.read, _jaxpr.outvars)

    def run_interpreter(self, _stateful_handler, fn, primals, tangents, **kwargs):
        def _inner(*args):
            return fn(*args, **kwargs)

        _closed_jaxpr, (flat_primals, _, out_tree) = stage(_inner)(*primals)
        flat_tangents = jtu.tree_leaves(
            tangents, is_leaf=lambda v: isinstance(v, ChangeTangent)
        )
        _jaxpr, consts = _closed_jaxpr.jaxpr, _closed_jaxpr.literals
        flat_out = self._eval_jaxpr_forward(
            _stateful_handler,
            _jaxpr,
            consts,
            flat_primals,
            flat_tangents,
        )
        return jtu.tree_unflatten(out_tree(), flat_out)


def incremental(f: Callable[..., Any]):
    @functools.wraps(f)
    def wrapped(
        _stateful_handler: StatefulHandler | None,
        primals: tuple[Any, ...],
        tangents: tuple[Any, ...],
    ):
        interpreter = IncrementalInterpreter()
        return interpreter.run_interpreter(
            _stateful_handler,
            f,
            primals,
            tangents,
        )

    return wrapped
