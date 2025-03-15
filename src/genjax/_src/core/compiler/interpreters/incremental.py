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

Incremental computation is currently a concern of Gen's `edit` GFI method - and can be utilized _as a runtime performance optimization_ for computing the weight (and changes to `Trace` instances) which `edit` computes.

*Change types*

By default, `genjax` provides two types of `ChangeTangent`:

* `NoChange` - indicating that a value has not changed.
* `UnknownChange` - indicating that a value has changed, without further information about the change.

`ChangeTangents` are provided along with primal values into `Diff` instances. The generative function `edit` interface expects tuples of `Pytree` instances whose leaves are `Diff` instances (`argdiffs`).
"""

import functools

import jax.tree_util as jtu
from jax import util as jax_util
from jax.extend.core import Jaxpr, Primitive

from genjax._src.core.compiler.interpreters.environment import Environment
from genjax._src.core.compiler.interpreters.stateful import StatefulHandler
from genjax._src.core.compiler.staging import stage
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    Generic,
    TypeVar,
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


@Pytree.dataclass
class _NoChange(ChangeTangent):
    pass


UnknownChange = _UnknownChange()
NoChange = _NoChange()


#############################
# Diffs (generalized duals) #
#############################


@Pytree.dataclass(match_args=True)
class Diff(Generic[R], Pytree):
    """
    A class representing a difference type in incremental computation.

    This class pairs a value with a change tangent, which indicates whether the value
    has a known change. It is used to track changes in values during incremental computation.

    The Diff class stores a value along with a marker (ChangeTangent) saying whether or not
    that value has a known change.

    Create Diff instances with `Diff.no_change` and `Diff.unknown_change`.

    Note:
        Diff instances should only be used as leaves of an outer pytree. They should not contain nested Diff instances or be used as internal nodes in a pytree structure.

    Attributes:
        primal: The value being tracked.
        tangent: The change tangent indicating whether the value has a known change.
    """

    primal: R
    tangent: ChangeTangent

    def get_primal(self) -> R:
        return self.primal

    def get_tangent(self) -> ChangeTangent:
        return self.tangent

    #############
    # Utilities #
    #############

    @staticmethod
    def tree_diff(tree: R, tangent_tree: R) -> R:
        """
        Create a Diff tree by combining a primal tree with a tangent tree.

        This static method takes two trees of the same structure and combines them
        into a single tree where each node is a Diff instance. The primal values
        come from the `tree` argument, and the tangent values come from the
        `tangent_tree` argument.

        Args:
            tree: The tree containing primal values.
            tangent_tree: The tree containing ChangeTangent values, with the same
                          structure as `tree`.

        Returns:
            A new tree with the same structure as the input trees, where each
            node is a Diff instance combining the corresponding primal and
            tangent values.

        Note:
            The input trees must have the same structure, or a ValueError will
            be raised during the tree_map operation.
        """
        return jtu.tree_map(
            lambda p, t: Diff(p, t),
            tree,
            tangent_tree,
        )

    @staticmethod
    def no_change(tree: R) -> R:
        """
        Create a Diff tree with NoChange tangents for all nodes, used to represent a tree where no values have changed.

        Args:
            tree: The input tree to be converted.

        Returns:
            A new tree with the same structure as the input, where each
            node is a Diff instance with the original value as the primal
            and NoChange as the tangent.

        Note:
            This method first extracts the primal values from the input tree
            (in case it already contains Diff instances) before creating the
            new Diff tree. If any leaf in the input tree is already a Diff
            instance, its existing ChangeTangent will be replaced with NoChange.
        """
        primal_tree: R = Diff.tree_primal(tree)
        tangent_tree: R = jtu.tree_map(lambda _: NoChange, primal_tree)
        return Diff.tree_diff(primal_tree, tangent_tree)

    @staticmethod
    def unknown_change(tree: R) -> R:
        """
        Create a Diff tree with UnknownChange tangents for all nodes, used to represent a tree where values may have changed.

        Args:
            tree: The input tree to be converted.

        Returns:
            A new tree with the same structure as the input, where each
            node is a Diff instance with the original value as the primal
            and UnknownChange as the tangent.

        Note:
            This method first extracts the primal values from the input tree
            (in case it already contains Diff instances) before creating the
            new Diff tree. If any leaf in the input tree is already a Diff
            instance, its existing ChangeTangent will be replaced with UnknownChange.
        """
        primal_tree: R = Diff.tree_primal(tree)
        tangent_tree: R = jtu.tree_map(lambda _: UnknownChange, primal_tree)
        return Diff.tree_diff(primal_tree, tangent_tree)

    @staticmethod
    def tree_primal(v: R | "Diff[R]") -> R:
        """
        Converts a pytree that may contain Diff instances into a pytree of primal values.

        Args:
            v: A tree structure that may contain Diff instances or regular values.

        Returns:
            A new tree with the same structure as the input, where all Diff instances are replaced with their primal values.
        """

        def _inner(v) -> R:
            if isinstance(v, Diff):
                return v.get_primal()
            else:
                return v

        return jtu.tree_map(_inner, v, is_leaf=Diff.is_diff)

    @staticmethod
    def tree_tangent(v: "R | Diff[R]") -> R:
        """
        Converts a pytree that may contain Diff instances into a pytree of ChangeTangent values.

        Args:
            v: A tree structure that may contain Diff instances or regular values.

        Returns:
            A new tree with the same structure as the input, where all Diff instances are replaced with their ChangeTangent values, and all other values are replaced with UnknownChange.
        """

        def _inner(v: R | Diff[R]) -> ChangeTangent:
            if isinstance(v, Diff):
                return v.get_tangent()
            else:
                return NoChange

        return jtu.tree_map(_inner, v, is_leaf=Diff.is_diff)

    #################
    # Static checks #
    #################

    @staticmethod
    def is_diff(v: Any) -> bool:
        """
        Checks if a value is a Diff instance.

        Args:
            v: The value to check.

        Returns:
            True if the value is a Diff instance, False otherwise.
        """
        return isinstance(v, Diff)

    @staticmethod
    def is_change_tangent(v: Any) -> bool:
        """
        Checks if a value is a ChangeTangent instance.

        Args:
            v: The value to check.

        Returns:
            True if the value is a ChangeTangent instance, False otherwise.
        """
        return isinstance(v, ChangeTangent)

    @staticmethod
    def static_check_tree_diff(v) -> bool:
        """
        Returns true if all leaves in a pytree are Diff instances, False otherwise.
        """
        return all(
            map(
                Diff.is_diff,
                jtu.tree_leaves(v, is_leaf=Diff.is_diff),
            )
        )

    @staticmethod
    def static_check_no_change(v) -> bool:
        """
        Returns true if all leaves in a pytree are NoChange, False otherwise.
        """
        return all(
            map(
                lambda leaf: isinstance(leaf, _NoChange),
                jtu.tree_leaves(Diff.tree_tangent(v), is_leaf=Diff.is_change_tangent),
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
        return Diff.no_change(outval)
    else:
        return Diff.unknown_change(outval)


@Pytree.dataclass
class IncrementalInterpreter(Pytree):
    custom_rules: dict[Primitive, Callable[..., Any]] = Pytree.static(
        default_factory=dict
    )

    def eval_jaxpr_incremental(
        self,
        stateful_handler,
        jaxpr: Jaxpr,
        consts: list[Any],
        primals: list[Any],
        tangents: list[ChangeTangent],
    ):
        dual_env = Environment()
        jax_util.safe_map(dual_env.write, jaxpr.constvars, Diff.no_change(consts))
        jax_util.safe_map(
            dual_env.write, jaxpr.invars, Diff.tree_diff(primals, tangents)
        )
        for _eqn in jaxpr.eqns:
            induals = jax_util.safe_map(dual_env.read, _eqn.invars)
            # TODO: why isn't this handled automatically by the environment,
            # especially the line above with _jaxpr.constvars?
            induals = [
                Diff(v, NoChange) if not isinstance(v, Diff) else v for v in induals
            ]
            subfuns, params = _eqn.primitive.get_bind_params(_eqn.params)
            args = subfuns + induals
            if stateful_handler and stateful_handler.handles(_eqn.primitive):
                outduals = stateful_handler.dispatch(_eqn.primitive, *args, **params)
            else:
                outduals = default_propagation_rule(_eqn.primitive, *args, **params)
            if not _eqn.primitive.multiple_results:
                outduals = [outduals]
            jax_util.safe_map(dual_env.write, _eqn.outvars, outduals)

        return jax_util.safe_map(dual_env.read, jaxpr.outvars)

    def run_interpreter(self, _stateful_handler, fn, primals, tangents, **kwargs):
        def _inner(*args):
            return fn(*args, **kwargs)

        closed_jaxpr, (flat_primals, _, out_tree) = stage(_inner)(*primals)
        flat_tangents = jtu.tree_leaves(
            tangents, is_leaf=lambda v: isinstance(v, ChangeTangent)
        )
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        flat_out = self.eval_jaxpr_incremental(
            _stateful_handler,
            jaxpr,
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
