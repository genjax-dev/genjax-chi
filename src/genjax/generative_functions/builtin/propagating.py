# Copyright 2022 The oryx Authors & MIT Probabilistic Computing Project
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
# `Cell` abstract/concrete values and static dispatch handling of primitives
# for probabilistic programming.
#
# The author maintains the code attribution notice from the `oryx`
# authors above, as a derivative work.

import abc
import contextlib
import copy
import plum

import collections
import dataclasses
import functools
import itertools as it
from typing import Any
from typing import Callable
from typing import Generator
from typing import Sequence
from typing import Dict
from typing import List
from typing import Tuple
from typing import Type
from typing import Union

from jax import api_util
from jax import abstract_arrays
from jax import core as jax_core
from jax import linear_util as lu
from jax import tree_util as jtu
from jax import util as jax_util
from jax._src import dtypes
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
from typing import Any
from typing import Dict
from typing import Generator
from typing import List

from genjax.core import Pytree
from genjax.core.datatypes import EmptyChoiceMap
from genjax.core.hashabledict import hashabledict
from genjax.core.masks import BooleanMask
from genjax.core.specialization import concrete_and
from genjax.core.specialization import concrete_cond
from genjax.core.specialization import is_concrete
from genjax.generative_functions.builtin.builtin_datatypes import (
    BuiltinChoiceMap,
)
from genjax.generative_functions.builtin.intrinsics import gen_fn_p

__all__ = [
    "Cell",
    "Equation",
    "Environment",
    "propagate",
    "get_shaped_aval",
    "pv_like",
    "stage",
    "trees",
    "Diff",
    "UnknownChange",
    "NoChange",
]

State = Any
VarOrLiteral = Union[jax_core.Var, jax_core.Literal]

safe_map = jax_core.safe_map
safe_zip = jax_core.safe_zip

#####
# Utilities
#####


def get_shaped_aval(x):
    """Converts a JAX value type into a shaped abstract value."""
    if hasattr(x, "dtype") and hasattr(x, "shape"):
        return abstract_arrays.ShapedArray(
            x.shape, dtypes.canonicalize_dtype(x.dtype)
        )
    return abstract_arrays.raise_to_shaped(jax_core.get_aval(x))


def pv_like(x, abstract=True):
    """Converts a JAX value type into a JAX `PartialVal`."""
    if abstract:
        return pe.PartialVal.unknown(get_shaped_aval(x))
    else:
        return pe.PartialVal((None, x))  # pytype: disable=wrong-arg-types


def stage(f, dynamic=True):
    """Returns a function that stages a function to a ClosedJaxpr."""

    def wrapped(*args, **kwargs):
        fun = lu.wrap_init(f, kwargs)
        flat_args, in_tree = jtu.tree_flatten(args)
        flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
        flat_avals = safe_map(get_shaped_aval, flat_args)
        if dynamic:
            jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, flat_avals)
        else:
            pvals = [pe.PartialVal.unknown(aval) for aval in flat_avals]
            jaxpr, _, consts = pe.trace_to_jaxpr(
                flat_fun, pvals, instantiate=True
            )
        typed_jaxpr = jax_core.ClosedJaxpr(jaxpr, consts)
        return typed_jaxpr, (in_tree, out_tree())

    return wrapped


def trees(f):
    """Returns a function that determines input and output pytrees from inputs."""

    def wrapped(*args, **kwargs):
        return stage(f)(*args, **kwargs)[1]

    return wrapped


def extract_call_jaxpr(primitive, params):
    if not (primitive.call_primitive or primitive.map_primitive):
        return None, params
    else:
        params = dict(params)
        return params.pop("call_jaxpr"), params


#####
# Cell
#####


class Cell(Pytree):
    """
    Base interface for objects used during propagation.

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


#####
# Propagation interpreter
#####


@dataclasses.dataclass(frozen=True)
class Equation:
    """
    Hashable wrapper for :code:`jax.core.Jaxpr`.
    """

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
    """
    Keeps track of variables and their values during propagation.
    """

    def __init__(self, cell_type, jaxpr):
        self.cell_type = cell_type
        self.env: Dict[jax_core.Var, Cell] = {}
        self.states: Dict[Equation, Cell] = {}
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
        return self.states.get(eqn, None)

    def write_state(self, eqn: Equation, state: State) -> None:
        self.states[eqn] = state


@dataclasses.dataclass
class Handler(Pytree):
    """
    A handler dispatchs a :code:`jax.core.Primitive` - and must provide
    a :code:`Callable` with signature :code:`def (name_of_primitive)(continuation, *args)`
    where :code:`*args` must match the :core:`jax.core.Primitive` declaration signature.
    """

    handles: jax_core.Primitive

    def flatten(self):
        return (), (self.handles,)

    @abc.abstractmethod
    def handle(self, incells, outcells, **kwargs):
        pass


def construct_graph_representation(eqns):
    """
    Constructs a graph representation of a Jaxpr.
    """
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
    """
    Updates the queue from the result of a propagation.
    """
    all_vars = cur_eqn.invars + cur_eqn.outvars
    old_cells = incells + outcells
    new_cells = new_incells + new_outcells

    for var, old_cell, new_cell in zip(all_vars, old_cells, new_cells):
        # If old_cell is less than new_cell, we know the propagation has made
        # progress.
        if old_cell < new_cell:
            # Extend left as a heuristic because in graphs corresponding to
            # chains of unary functions, we immediately want to pop off these
            # neighbors in the next iteration
            neighbors = get_neighbor_eqns(var) - set(queue) - {cur_eqn}
            queue.extendleft(neighbors)


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
    """
    This interpreter converts a `Jaxpr` to a directed graph where
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

    # Initialize propagation queue with equations neighboring
    # constvars, invars, and outvars.
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
            else:
                rule = propagation_rule
        else:
            subfuns = []

            ############################################
            #   Static handler dispatch occurs here.   #
            ############################################

            if hasattr(eqn.primitive, "must_handle"):
                assert eqn.primitive == handler.handles
                rule = handler.handle

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
    """
    Propagation rule for JAX/XLA call primitives.
    """
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

#######################################
# Change type lattice and propagation #
#######################################

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


class Change(Pytree):
    pass


class _UnknownChange(Change):
    def flatten(self):
        return (), ()


UnknownChange = _UnknownChange()


class _NoChange(Change):
    def flatten(self):
        return (), ()


NoChange = _NoChange()


@dataclasses.dataclass
class Diff(Cell):
    val: Any
    change: Change

    def __init__(self, aval, val, change):
        super().__init__(aval)
        self.val = val
        self.change = change

    def flatten(self):
        return (self.val, self.change), (self.aval,)

    def __lt__(self, other):
        return self.bottom() and other.top()

    def top(self):
        return self.val is not None

    def bottom(self):
        return self.val is None

    def join(self, other):
        if other.bottom():
            return self
        else:
            return other

    @classmethod
    def new(cls, val, change: Change = UnknownChange):
        aval = get_shaped_aval(val)
        return Diff(aval, val, change)

    @classmethod
    def unknown(cls, aval):
        return Diff(aval, None, UnknownChange)

    def get_change(self):
        return self.change

    def get_val(self):
        return self.val


def diff_strip(diff):
    return diff.get_val()


@abstract
def propagation_rule(
    prim: Any, incells: Sequence[Diff], outcells: Any, **params
):
    if all(map(lambda v: v.top(), incells)):
        in_vals = list(map(lambda v: v.get_val(), incells))
        flat_out = prim.bind(*in_vals, **params)
        if all(map(lambda v: v.get_change() == NoChange, incells)):
            new_out = [Diff.new(flat_out, change=NoChange)]
        else:
            new_out = [Diff.new(flat_out)]
    else:
        new_out = outcells
    return incells, new_out, None


def check_no_change(diff):
    return diff.get_change() == NoChange


######################################
#  Generative function interpreters  #
######################################


class Update(Handler):
    def __init__(self, prev, new):
        self.handles = gen_fn_p
        self.state = BuiltinChoiceMap(hashabledict())
        self.discard = BuiltinChoiceMap(hashabledict())
        self.weight = 0.0
        self.prev = prev
        self.choice_change = new
        self.return_or_continue = False

    def handle(self, _, incells, outcells, addr, gen_fn, args_form, **kwargs):
        key, *args = incells
        key = key.get_val()

        # We haven't handled the predecessors of this trace
        # call yet, so we return back to the abstract interpreter
        # to continue propagation.
        if key is None:
            return incells, outcells, None

        has_previous = self.prev.has_subtree(addr)
        constrained = self.choice_change.has_subtree(addr)

        # If no changes, we can just short-circuit.
        if (
            is_concrete(has_previous)
            and is_concrete(constrained)
            and has_previous
            and not constrained
            and map(check_no_change, incells)
        ):
            prev = self.prev.get_subtree(addr)
            return (
                incells,
                [
                    Diff.new(key, change=NoChange),
                    Diff.new(prev.get_retval(), change=NoChange),
                ],
                None,
            )

        # Otherwise, we send the update down to the callee.
        def _update_branch(key, args):
            prev_tr = self.prev.get_subtree(addr)
            chm = self.choice_change.get_subtree(addr)
            key, (retval, w, tr, discard) = gen_fn.update(
                key, prev_tr, chm, args, **kwargs
            )
            discard = discard.strip()
            return key, (retval, w, tr, discard)

        def _has_prev_branch(key, args):
            prev_tr = self.prev.get_subtree(addr)
            key, (retval, w, _, _) = gen_fn.update(
                key, prev_tr, EmptyChoiceMap(), args, **kwargs
            )
            discard = BooleanMask.new(False, prev_tr.strip())
            return key, (retval, w, prev_tr, discard)

        def _constrained_branch(key, args):
            args = map(diff_strip, args)
            prev_tr = self.prev.get_subtree(addr)
            chm = self.choice_change.get_subtree(addr)
            key, (w, tr) = gen_fn.importance(key, chm, args, **kwargs)
            discard = BooleanMask.new(False, prev_tr.strip())
            retval = tr.get_retval()
            return key, (retval, w, tr, discard)

        key, (retval, w, tr, discard) = concrete_cond(
            concrete_and(has_previous, constrained),
            _update_branch,
            lambda key, args: concrete_cond(
                has_previous,
                _has_prev_branch,
                _constrained_branch,
                key,
                args,
            ),
            key,
            args,
        )

        key = Diff.new(key)
        new_outcells = [key, retval]

        self.weight += w
        self.state[addr] = tr
        self.discard[addr] = discard

        return incells, new_outcells, None


def handler_update(f, **kwargs):
    def _inner(key, prev, new, args):
        vals = tuple(map(diff_strip, args))
        jaxpr, _ = stage(f)(key, *vals, **kwargs)
        jaxpr, consts = jaxpr.jaxpr, jaxpr.literals
        handler = Update(prev, new)
        final_env, ret_state = propagate(
            Diff,
            jaxpr,
            [Diff.new(v, change=NoChange) for v in consts],
            [Diff.new(key), *args],
            [Diff.unknown(var.aval) for var in jaxpr.outvars],
            handler=handler,
        )
        key, retval_diff = safe_map(final_env.read, jaxpr.outvars)
        w = handler.weight
        chm = handler.state
        discard = handler.discard
        key = key.get_val()
        retval = diff_strip(retval_diff)
        return (
            key,
            (
                w,
                retval_diff,
                (f, vals, retval, chm, prev.get_score() + w),
                discard,
            ),
        )

    return _inner
