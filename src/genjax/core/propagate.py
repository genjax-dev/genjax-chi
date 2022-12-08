# Copyright 2022 The MIT Probabilistic Computing Project & the oryx authors.
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

# Note: this code was originally from two places in the JAX codebase.
# A fork by Roy Frostig and source code for `oryx`, a probabilistic
# programming library built on top of JAX.
#
# The code has been modified to enable simultaneous propagation of
# `Cell` abstract/concrete values and static dispatch handling of
# primitives for probabilistic programming.
#
# The author maintains the code attribution notice from the `oryx`
# authors above, as a derivative work.

# Note: this code was originally from two places in the JAX codebase.
# A fork by Roy Frostig and source code for `oryx`, a probabilistic
# programming library built on top of JAX.
#
# The code has been modified to enable simultaneous propagation of
# `Cell` abstract/concrete values and static dispatch handling of
# primitives for probabilistic programming.
#
# The author maintains the code attribution notice from the `oryx`
# authors above, as a derivative work.


import collections
import dataclasses
import functools
import itertools as it
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Type
from typing import Union

import plum
from jax import core as jax_core
from jax import linear_util as lu
from jax import tree_util as jtu
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla

from genjax.core.pytree import Pytree
from genjax.core.staging import extract_call_jaxpr


__all__ = [
    "Cell",
    "Equation",
    "Environment",
    "propagate",
]

State = Any
VarOrLiteral = Union[jax_core.Var, jax_core.Literal]

safe_map = jax_core.safe_map
safe_zip = jax_core.safe_zip

###########################
# Propagation interpreter #
###########################


class Cell(Pytree):
    """Base interface for objects used during propagation.

    A `Cell` represents a member of a lattice, defined by the `top`, `bottom`
    and `join` methods. Conceptually, a "top" cell represents complete
    information about a value and a "bottom" cell represents no
    information about a value.

    Cells that are neither top nor bottom thus have partial information.
    The `join` method is used to combine two cells to create a cell
    no less than the two input cells. During the propagation,
    we hope to join cells until all cells are "top".

    Transformations that use `propagate` need to pass in objects
    that are `Cell`-like.

    A `Cell` needs to specify how to create a new default cell
    from a literal value, using the `new` class method.
    A `Cell` also needs to indicate if it is a known value with
    the `is_unknown` method, but by default, `Cell` instances are known.
    """

    def __init__(self, aval):
        self.aval = aval

    def __lt__(self, other: Any) -> bool:
        raise NotImplementedError

    def top(self) -> bool:
        raise NotImplementedError

    def bottom(self) -> bool:
        raise NotImplementedError

    def join(self, other: "Cell") -> "Cell":
        raise NotImplementedError

    @property
    def shape(self) -> Tuple[int]:
        return self.aval.shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def is_unknown(self):
        # Convenient alias
        return self.bottom()

    @classmethod
    def new(cls, value):
        """Creates a new instance of a Cell from a value."""
        raise NotImplementedError

    @classmethod
    def unknown(cls, aval):
        """Creates an unknown Cell from an abstract value."""
        raise NotImplementedError


def map_outcells(cell_type, v, **kwargs):
    if isinstance(v, tuple) or isinstance(v, list):
        return list(map(lambda v: cell_type.new(v, **kwargs), v))
    else:
        return list([cell_type.new(v, **kwargs)])


@dataclasses.dataclass(frozen=True)
class Equation:
    """Hashable wrapper for :code:`jax.core.Jaxpr`."""

    invars: Tuple[jax_core.Var]
    outvars: Tuple[jax_core.Var]
    primitive: jax_core.Primitive
    params_flat: Tuple[Any]
    params_tree: Any

    @classmethod
    def from_jaxpr_eqn(cls, eqn):
        params_flat, params_tree = jtu.tree_flatten(eqn.params)
        return Equation(
            tuple(eqn.invars),
            tuple(eqn.outvars),
            eqn.primitive,
            tuple(params_flat),
            params_tree,
        )

    @property
    def params(self):
        return jtu.tree_unflatten(self.params_tree, self.params_flat)

    def __hash__(self):
        # Override __hash__ to use
        # Literal object IDs because Literals are not
        # natively hashable
        hashable_invars = tuple(
            id(invar) if isinstance(invar, jax_core.Literal) else invar
            for invar in self.invars
        )
        return hash(
            (hashable_invars, self.outvars, self.primitive, self.params_tree)
        )

    def __str__(self):
        return "{outvars} = {primitive} {invars}".format(
            invars=" ".join(map(str, self.invars)),
            outvars=" ".join(map(str, self.outvars)),
            primitive=self.primitive,
        )


class Environment:
    """Keeps track of variables and their values during propagation."""

    def __init__(self, cell_type, jaxpr):
        self.cell_type = cell_type
        self.env: Dict[jax_core.Var, Cell] = {}
        self.choice_states: Dict[Equation, Cell] = {}
        self.jaxpr: jax_core.Jaxpr = jaxpr

    def read(self, var: VarOrLiteral) -> Cell:
        if isinstance(var, jax_core.Literal):
            return self.cell_type.new(var.val)
        else:
            return self.env.get(var, self.cell_type.unknown(var.aval))

    def write(self, var: VarOrLiteral, cell: Cell) -> Cell:
        if isinstance(var, jax_core.Literal):
            return cell
        cur_cell = self.read(var)
        if isinstance(var, jax_core.DropVar):
            return cur_cell
        self.env[var] = cur_cell.join(cell)
        return self.env[var]

    def __getitem__(self, var: VarOrLiteral) -> Cell:
        return self.read(var)

    def __setitem__(self, key, val):
        raise ValueError(
            "Environments do not support __setitem__. Please use the "
            "`write` method instead."
        )

    def __contains__(self, var: VarOrLiteral):
        if isinstance(var, jax_core.Literal):
            return True
        return var in self.env

    def read_state(self, eqn: Equation) -> State:
        return self.choice_states.get(eqn, None)

    def write_state(self, eqn: Equation, state: State) -> None:
        self.choice_states[eqn] = state


@dataclasses.dataclass
class Handler(Pytree):
    """
    A handler dispatchs a :code:`jax.core.Primitive` - and must provide
    a :code:`Callable` with signature :code:`def (name_of_primitive)(continuation, *args)`
    where :code:`*args` must match the :core:`jax.core.Primitive` declaration signature.
    """

    handles: List[jax_core.Primitive]

    def flatten(self):
        return (), (self.handles,)


def construct_graph_representation(eqns):
    """Constructs a graph representation of a Jaxpr."""
    neighbors = collections.defaultdict(set)
    for eqn in eqns:
        for var in it.chain(eqn.invars, eqn.outvars):
            if isinstance(var, jax_core.Literal):
                continue
            neighbors[var].add(eqn)

    def get_neighbors(var):
        if isinstance(var, jax_core.Literal):
            return set()
        return neighbors[var]

    return get_neighbors


def update_queue_state(
    queue,
    cur_eqn,
    get_neighbor_eqns,
    incells,
    outcells,
    new_incells,
    new_outcells,
):
    """Updates the queue from the result of a propagation."""
    all_vars = cur_eqn.invars + cur_eqn.outvars
    old_cells = incells + outcells
    new_cells = new_incells + new_outcells

    for var, old_cell, new_cell in zip(all_vars, old_cells, new_cells):
        # If old_cell is less than new_cell, we know the propagation
        # has made progress.
        if old_cell < new_cell:
            # Extend left as a heuristic because in graphs
            # corresponding to chains of unary functions,
            # we immediately want to pop off these neighbors in the
            # next iteration
            neighbors = get_neighbor_eqns(var) - set(queue) - {cur_eqn}
            queue.extendleft(neighbors)

    # If the propagation failed to make progress because one of the
    # incells was bottom, and the propagation did not update that
    # incell, we want to make sure to re-visit it later on in the
    # propagation.
    for var, new_cell in zip(cur_eqn.invars, new_incells):
        if new_cell.bottom():
            neighbors = get_neighbor_eqns(var) - set(queue) - {cur_eqn}
            queue.extend(neighbors)


def identity_reducer(env, eqn, state, new_state):
    del env, eqn, new_state
    return state


PropagationRule = Callable[
    [List[Any], List[Cell]],
    Tuple[List[Cell], List[Cell], State],
]


def propagate(
    cell_type: Type[Cell],
    jaxpr: pe.Jaxpr,
    constcells: List[Cell],
    incells: List[Cell],
    outcells: List[Cell],
    reducer: Callable[
        [Environment, Equation, State, State], State
    ] = identity_reducer,
    initial_state: State = None,
    handler: Union[None, Handler] = None,
) -> Tuple[Environment, State]:
    """This interpreter converts a `Jaxpr` to a directed graph where
    `jax.core.Var` instances are nodes and primitives are edges.

    It initializes `invars` and `outvars` with `Cell` instances,
    where a `Cell` encapsulates a value (or a set of values)
    that a node in the graph can take on,
    and the `Cell` is computed from neighboring `Cell` instances,
    using a set of propagation rules for each primitive.

    Each rule indicates whether the propagation has been completed for
    the given edge. If so, the interpreter continues on to that
    primitive's neighbors in the graph. Propagation continues until
    there are `Cell` instances for every node, or when no
    further progress can be made.

    Finally, `Cell` values for all nodes in the graph are returned.

    Args:
        cell_type: used to instantiate literals into cells
        jaxpr: used to construct the propagation graph
        constcells: used to populate the Jaxpr's constvars
        incells: used to populate the Jaxpr's invars
        outcells: used to populate the Jaxpr's outcells
        reducer: An optional callable used to reduce over the state at each
            equation in the Jaxpr. :code:`reducer`: takes in
            :code:`(env, eqn, state, new_state)` as arguments and should
            return an updated state. The :code:`new_state` value is provided
            by each equation.
        initial_state: The initial :code:`state` value used in the reducer

    Returns:
        The :code:`Jaxpr` environment after propagation has terminated
    """

    env = Environment(cell_type, jaxpr)
    safe_map(env.write, jaxpr.constvars, constcells)
    safe_map(env.write, jaxpr.outvars, outcells)
    safe_map(env.write, jaxpr.invars, incells)

    eqns = safe_map(Equation.from_jaxpr_eqn, jaxpr.eqns)
    get_neighbor_eqns = construct_graph_representation(eqns)

    out_eqns = set()
    for eqn in jaxpr.eqns:
        for var in it.chain(eqn.invars, eqn.outvars):
            env.write(var, cell_type.unknown(var.aval))

    for var in it.chain(jaxpr.outvars, jaxpr.invars, jaxpr.constvars):
        out_eqns.update(get_neighbor_eqns(var))

    queue = collections.deque(out_eqns)

    while queue:
        eqn = queue.popleft()
        incells = safe_map(env.read, eqn.invars)
        outcells = safe_map(env.read, eqn.outvars)

        # If we have top information about all incells and outcells
        # for this equation, skip to the next in the queue.
        if all(map(lambda cell: cell.top(), incells)) and all(
            map(lambda cell: cell.top(), outcells)
        ):
            continue

        call_jaxpr, params = extract_call_jaxpr(eqn.primitive, eqn.params)
        if call_jaxpr:
            subfuns = [
                lu.wrap_init(
                    functools.partial(
                        propagate,
                        cell_type,
                        call_jaxpr,
                        (),
                        initial_state=initial_state,
                        reducer=reducer,
                        handler=handler,
                    )
                )
            ]
            if eqn.primitive in default_call_rules:
                rule = default_call_rules.get(eqn.primitive)
                # Just ignore the primitive - it's handled by our
                # default_call_rules dictionary.
                rule = lambda _, incells, outcells, **params: rule(
                    incells, outcells, **params
                )
            else:
                rule = propagation_rule
        else:
            subfuns = []

            ############################################
            #   Static handler dispatch occurs here.   #
            ############################################

            if hasattr(eqn.primitive, "must_handle"):
                assert eqn.primitive in handler.handles
                rule = getattr(handler, repr(eqn.primitive))

            ############################################
            #    Static handler dispatch ends here.    #
            ############################################

            else:
                rule = propagation_rule

        # Apply a propagation rule.
        new_incells, new_outcells, eqn_state = rule(
            eqn.primitive,
            subfuns + incells,
            outcells,
            **params,
        )

        env.write_state(eqn, eqn_state)
        new_incells = safe_map(env.write, eqn.invars, new_incells)
        new_outcells = safe_map(env.write, eqn.outvars, new_outcells)

        update_queue_state(
            queue,
            eqn,
            get_neighbor_eqns,
            incells,
            outcells,
            new_incells,
            new_outcells,
        )

    state = initial_state
    for eqn in eqns:
        state = reducer(env, eqn, state, env.read_state(eqn))

    return env, state


@lu.transformation_with_aux
def flat_propagate(tree, *flat_invals):
    invals, outvals = jtu.tree_unflatten(tree, flat_invals)
    env, state = yield ((invals, outvals), {})
    new_incells = [env.read(var) for var in env.jaxpr.invars]
    new_outcells = [env.read(var) for var in env.jaxpr.outvars]
    flat_out, out_tree = jtu.tree_flatten((new_incells, new_outcells, state))
    yield flat_out, out_tree


def call_rule(prim, incells, outcells, **params):
    """Propagation rule for JAX/XLA call primitives."""
    f, incells = incells[0], incells[1:]
    flat_vals, in_tree = jtu.tree_flatten((incells, outcells))
    new_params = dict(params)
    if "donated_invars" in params:
        new_params["donated_invars"] = (False,) * len(flat_vals)
    f, aux = flat_propagate(f, in_tree)
    flat_out = prim.bind(f, *flat_vals, **new_params)
    out_tree = aux()
    return jtu.tree_unflatten(out_tree, flat_out)


default_call_rules = {}
default_call_rules[xla.xla_call_p] = functools.partial(
    call_rule, xla.xla_call_p
)
default_call_rules[jax_core.call_p] = functools.partial(
    call_rule, jax_core.call_p
)

######
# Propagation rules
######

# We use multiple dispatch to support overloading propagation rules.
# At tracing time, the dispatch arguments will contain abstract
# tracer values.
abstract = plum.dispatch

# Fallback: we should error -- we're encountering a primitive with
# types that we don't have a rule for.
@abstract
def propagation_rule(prim: Any, incells: Any, outcells: Any, **params):
    raise Exception(
        f"({prim}, {(*incells,)}) Propagation rule not implemented."
    )
