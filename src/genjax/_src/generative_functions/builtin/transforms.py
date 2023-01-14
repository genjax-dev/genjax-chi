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

import dataclasses
import functools
from typing import Any
from typing import List

import jax.core as core
import jax.tree_util as jtu
from jax.interpreters import xla

from genjax._src.core.datatypes import ChoiceMap
from genjax._src.core.datatypes import EmptyChoiceMap
from genjax._src.core.datatypes import Trace
from genjax._src.core.diff_rules import Diff
from genjax._src.core.diff_rules import NoChange
from genjax._src.core.diff_rules import check_is_diff
from genjax._src.core.diff_rules import check_no_change
from genjax._src.core.diff_rules import diff_propagation_rules
from genjax._src.core.diff_rules import strip_diff
from genjax._src.core.propagate import Cell
from genjax._src.core.propagate import Handler
from genjax._src.core.propagate import PropagationInterpreter
from genjax._src.core.propagate import PropagationRules
from genjax._src.core.propagate import flat_propagate
from genjax._src.core.propagate import flatmap_outcells
from genjax._src.core.specialization import is_concrete
from genjax._src.core.staging import get_shaped_aval
from genjax._src.core.staging import stage
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import PRNGKey
from genjax._src.generative_functions.builtin.builtin_datatypes import Trie
from genjax._src.generative_functions.builtin.intrinsics import cache_p
from genjax._src.generative_functions.builtin.intrinsics import gen_fn_p


safe_map = core.safe_map
safe_zip = core.safe_zip

##################################################
# Bare values (for interfaces sans change types) #
##################################################


@dataclasses.dataclass
class Bare(Cell):
    val: Any

    def __init__(self, aval, val):
        super().__init__(aval)
        self.val = val

    def flatten(self):
        return (self.val,), (self.aval,)

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
    def new(cls, val):
        aval = get_shaped_aval(val)
        return Bare(aval, val)

    @classmethod
    def unknown(cls, aval):
        return Bare(aval, None)

    def get_val(self):
        return self.val


def bare_fallback_rule(
    prim: Any, incells: List[Bare], outcells: Any, **params
):
    if all(map(lambda v: v.top(), incells)):
        in_vals = list(map(lambda v: v.get_val(), incells))
        out = prim.bind(*in_vals, **params)
        new_out = jtu.tree_map(Bare.new, out)
        if not prim.multiple_results:
            new_out = [new_out]
    else:
        new_out = outcells
    return incells, new_out, None


def bare_call_p_rule(prim, incells, outcells, **params):
    """Propagation rule for JAX/XLA call primitives."""
    f, incells = incells[0], incells[1:]
    if all(map(lambda v: v.top(), incells)):
        flat_vals, in_tree = jtu.tree_flatten((incells, outcells))
        new_params = dict(params)
        if "donated_invars" in params:
            new_params["donated_invars"] = (False,) * len(flat_vals)
        f, aux = flat_propagate(f, in_tree)
        flat_out = prim.bind(f, *flat_vals, **new_params)
        out_tree = aux()
        return jtu.tree_unflatten(out_tree, flat_out)
    else:
        return incells, outcells, None


bare_call_rules = {}
bare_call_rules[xla.xla_call_p] = functools.partial(
    bare_call_p_rule, xla.xla_call_p
)
bare_call_rules[core.call_p] = functools.partial(bare_call_p_rule, core.call_p)

bare_propagation_rules = PropagationRules(bare_fallback_rule, bare_call_rules)

######################################
#  Generative function interpreters  #
######################################


def static_map_check_bottom(incells):
    def _inner(v):
        if isinstance(v, Cell):
            return v.bottom()
        else:
            return False

    return any(map(_inner, incells))


def static_map_unwrap(incells):
    def _inner(v):
        if isinstance(v, Cell):
            return v.get_val()
        else:
            return v

    return [_inner(v) for v in incells]


#####
# Simulate
#####


@dataclasses.dataclass
class Simulate(Handler):
    handles: List[core.Primitive]
    key: PRNGKey
    score: FloatArray
    choice_state: Trie
    cache_state: Trie

    def flatten(self):
        return (
            self.key,
            self.score,
            self.choice_state,
            self.cache_state,
        ), (self.handles,)

    @classmethod
    def new(cls, key: PRNGKey):
        score = 0.0
        choice_state = Trie.new()
        cache_state = Trie.new()
        handles = [gen_fn_p, cache_p]
        return Simulate(handles, key, score, choice_state, cache_state)

    def trace(self, incells, outcells, addr, tree_in, **kwargs):
        # We haven't handled the predecessors of this trace
        # call yet, so we return back to the abstract interpreter
        # to continue propagation.
        if static_map_check_bottom(incells):
            return incells, outcells, None

        values = static_map_unwrap(incells)
        gen_fn, args = jtu.tree_unflatten(tree_in, values)
        args = tuple(args)

        # Otherwise, we send simulate down to the generative function
        # callee.
        self.key, tr = gen_fn.simulate(self.key, args, **kwargs)
        score = tr.get_score()
        self.choice_state[addr] = tr
        self.score += score

        # Here, we keep the outcells dimension (e.g. the same as
        # the number of outgoing variables in the equation) invariant
        # under our expansion.
        v = tr.get_retval()
        new_outcells = flatmap_outcells(Bare, v)

        return incells, new_outcells, None

    def cache(self, incells, outcells, addr, fn, tree_in, **kwargs):
        # We haven't handled the predecessors of this trace
        # call yet, so we return back to the abstract interpreter
        # to continue propagation.
        if static_map_check_bottom(incells):
            return incells, outcells, None

        values = static_map_unwrap(incells)
        args = jtu.tree_unflatten(tree_in, values)

        # Otherwise, we codegen by calling into the function.
        retval = fn(*args)
        self.cache_state[addr] = retval

        # Here, we keep the outcells dimension (e.g. the same as
        # the number of outgoing variables in the equation) invariant
        # under our expansion.
        new_outcells = flatmap_outcells(Bare, retval)

        return incells, new_outcells, None


def simulate_transform(source_fn, **kwargs):
    @functools.wraps(source_fn)
    def _inner(key, args):
        closed_jaxpr, (flat_args, _, out_tree) = stage(source_fn)(
            *args, **kwargs
        )
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        handler = Simulate.new(key)
        with PropagationInterpreter.new(
            Bare, bare_propagation_rules, handler
        ) as interpreter:
            final_env, _ = interpreter(
                jaxpr,
                [Bare.new(v) for v in consts],
                list(map(Bare.new, flat_args)),
                [Bare.unknown(var.aval) for var in jaxpr.outvars],
            )
            flat_out = safe_map(final_env.read, jaxpr.outvars)
            flat_out = map(lambda v: v.get_val(), flat_out)
            retvals = jtu.tree_unflatten(out_tree, flat_out)
            score = handler.score
            constraints = handler.choice_state
            cache = handler.cache_state
            key = handler.key

        return key, (source_fn, args, retvals, constraints, score), cache

    return _inner


#####
# Importance
#####


@dataclasses.dataclass
class Importance(Handler):
    handles: List[core.Primitive]
    key: PRNGKey
    score: FloatArray
    weight: FloatArray
    constraints: ChoiceMap
    choice_state: Trie
    cache_state: Trie

    def flatten(self):
        return (
            self.key,
            self.score,
            self.weight,
            self.constraints,
            self.choice_state,
            self.cache_state,
        ), (self.handles,)

    @classmethod
    def new(cls, key, constraints):
        handles = [gen_fn_p, cache_p]
        score = 0.0
        weight = 0.0
        choice_state = Trie.new()
        cache_state = Trie.new()
        return Importance(
            handles, key, score, weight, constraints, choice_state, cache_state
        )

    def trace(self, incells, outcells, addr, tree_in, **kwargs):
        # We haven't handled the predecessors of this trace
        # call yet, so we return back to the abstract interpreter
        # to continue propagation.
        if static_map_check_bottom(incells):
            return incells, outcells, None

        # Otherwise, we proceed with code generation.
        values = static_map_unwrap(incells)
        gen_fn, args = jtu.tree_unflatten(tree_in, values)
        args = tuple(args)
        if self.constraints.has_subtree(addr):
            sub_map = self.constraints.get_subtree(addr)
        else:
            sub_map = EmptyChoiceMap()
        self.key, (w, tr) = gen_fn.importance(self.key, sub_map, args)
        self.choice_state[addr] = tr
        self.score += tr.get_score()
        self.weight += w

        # Here, we keep the outcells dimension (e.g. the same as
        # the number of outgoing variables in the equation) invariant
        # under our expansion.
        v = tr.get_retval()
        new_outcells = flatmap_outcells(Bare, v)

        return incells, new_outcells, None

    def cache(self, incells, outcells, addr, fn, **kwargs):
        # We haven't handled the predecessors of this trace
        # call yet, so we return back to the abstract interpreter
        # to continue propagation.
        if static_map_check_bottom(incells):
            return incells, outcells, None

        # Otherwise, we proceed with code generation.
        values = static_map_unwrap(incells)
        args = tuple(values)
        retval = fn(*args)
        self.cache_state[addr] = retval

        # Here, we keep the outcells dimension (e.g. the same as
        # the number of outgoing variables in the equation) invariant
        # under our expansion.
        new_outcells = flatmap_outcells(Bare, retval)

        return incells, new_outcells, None


def importance_transform(source_fn, **kwargs):
    @functools.wraps(source_fn)
    def _inner(key, constraints, args):
        closed_jaxpr, (flat_args, _, out_tree) = stage(source_fn)(
            *args, **kwargs
        )
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        handler = Importance.new(key, constraints)
        with PropagationInterpreter.new(
            Bare, bare_propagation_rules, handler
        ) as interpreter:
            final_env, _ = interpreter(
                jaxpr,
                [Bare.new(v) for v in consts],
                list(map(Bare.new, flat_args)),
                [Bare.unknown(var.aval) for var in jaxpr.outvars],
            )
            flat_out = safe_map(final_env.read, jaxpr.outvars)
            flat_out = map(lambda v: v.get_val(), flat_out)
            retvals = jtu.tree_unflatten(out_tree, flat_out)
            w = handler.weight
            score = handler.score
            constraints = handler.choice_state
            cache = handler.cache_state
            key = handler.key

        return key, (w, (source_fn, args, retvals, constraints, score)), cache

    return _inner


#####
# Update
#####


@dataclasses.dataclass
class Update(Handler):
    handles: List[core.Primitive]
    key: PRNGKey
    score: FloatArray
    weight: FloatArray
    previous_trace: Trace
    constraints: ChoiceMap
    discard: Trie
    choice_state: Trie
    cache_state: Trie

    def flatten(self):
        return (
            self.key,
            self.score,
            self.weight,
            self.previous_trace,
            self.constraints,
            self.discard,
            self.choice_state,
            self.cache_state,
        ), (self.handles,)

    @classmethod
    def new(cls, key, previous_trace, constraints):
        handles = [gen_fn_p, cache_p]
        score = 0.0
        weight = 0.0
        choice_state = Trie.new()
        cache_state = Trie.new()
        discard = Trie.new()
        return Update(
            handles,
            key,
            score,
            weight,
            previous_trace,
            constraints,
            discard,
            choice_state,
            cache_state,
        )

    def trace(self, incells, outcells, addr, tree_in, **kwargs):
        # We haven't handled the predecessors of this trace
        # call yet, so we return back to the abstract interpreter
        # to continue propagation.
        if static_map_check_bottom(incells):
            return incells, outcells, None

        gen_fn, argdiffs = jtu.tree_unflatten(tree_in, incells)
        argdiffs = tuple(argdiffs)
        has_previous = self.previous_trace.choices.has_subtree(addr)
        constrained = self.constraints.has_subtree(addr)

        # If no changes, we can just short-circuit.
        if (
            is_concrete(has_previous)
            and is_concrete(constrained)
            and has_previous
            and not constrained
            and all(map(check_no_change, incells))
        ):
            subtrace = self.previous_trace.choices.get_subtree(addr)
            self.choice_state[addr] = subtrace

            # Here, we keep the outcells dimension (e.g. the same as
            # the number of outgoing variables in the equation) invariant
            # under our expansion.
            v = subtrace.get_retval()
            new_outcells = flatmap_outcells(Bare, v, change=NoChange)
            return (
                incells,
                new_outcells,
                None,
            )

        # Otherwise, we proceed with code generation.
        subtrace = self.previous_trace.choices.get_subtree(addr)
        if constrained:
            subconstraints = self.constraints.get_subtree(addr)
        else:
            subconstraints = EmptyChoiceMap()
        self.key, (retval_diff, w, tr, discard) = gen_fn.update(
            self.key, subtrace, subconstraints, argdiffs
        )

        self.weight += w
        self.choice_state[addr] = tr
        self.discard[addr] = discard

        # Here, we keep the outcells dimension (e.g. the same as
        # the number of outgoing variables in the equation) invariant
        # under our expansion.
        new_outcells = flatmap_outcells(Diff, retval_diff)

        return incells, new_outcells, None

    def cache(self, incells, outcells, addr, fn, **kwargs):
        # We haven't handled the predecessors of this trace
        # call yet, so we return back to the abstract interpreter
        # to continue propagation.
        if static_map_check_bottom(incells):
            return incells, outcells, None

        values = static_map_unwrap(incells)
        has_value = self.previous_trace.has_cached_value(addr)

        # If no changes, we can just fetch from trace.
        if (
            is_concrete(has_value)
            and has_value
            and all(map(check_no_change, incells))
        ):
            cached_value = self.previous_trace.get_cached_value(addr)
            self.cache_state[addr] = cached_value

            # Here, we keep the outcells dimension (e.g. the same as
            # the number of outgoing variables in the equation) invariant
            # under our expansion.
            new_outcells = flatmap_outcells(
                Diff,
                cached_value,
                change=NoChange,
            )

            return incells, new_outcells, None

        # Otherwise, we codegen by calling into the function.
        retval = fn(*values)
        self.cache_state[addr] = retval

        # Here, we keep the outcells dimension (e.g. the same as
        # the number of outgoing variables in the equation) invariant
        # under our expansion.
        new_outcells = flatmap_outcells(Bare, retval)

        return incells, new_outcells, None


def update_transform(source_fn, **kwargs):
    @functools.wraps(source_fn)
    def _inner(key, previous_trace, constraints, argdiffs):
        vals = jtu.tree_map(strip_diff, argdiffs, is_leaf=check_is_diff)
        jaxpr, (_, _, out_tree) = stage(source_fn)(*vals, **kwargs)
        jaxpr, consts = jaxpr.jaxpr, jaxpr.literals
        handler = Update.new(key, previous_trace, constraints)
        with PropagationInterpreter.new(
            Diff, diff_propagation_rules, handler
        ) as interpreter:
            flat_argdiffs, _ = jtu.tree_flatten(
                argdiffs, is_leaf=check_is_diff
            )
            final_env, _ = interpreter(
                jaxpr,
                [Diff.new(v, change=NoChange) for v in consts],
                flat_argdiffs,
                [Diff.unknown(var.aval) for var in jaxpr.outvars],
            )

            flat_retval_diffs = safe_map(final_env.read, jaxpr.outvars)
            retval_diffs = jtu.tree_unflatten(
                out_tree,
                flat_retval_diffs,
            )
            retvals = tuple(map(strip_diff, flat_retval_diffs))
            retvals = jtu.tree_unflatten(out_tree, retvals)
            w = handler.weight
            constraints = handler.choice_state
            cache = handler.cache_state
            discard = handler.discard
            key = handler.key

        return (
            key,
            (
                retval_diffs,
                w,
                (
                    source_fn,
                    vals,
                    retvals,
                    constraints,
                    previous_trace.get_score() + w,
                ),
                discard,
            ),
            cache,
        )

    return _inner


#####
# Assess
#####


@dataclasses.dataclass
class Assess(Handler):
    handles: List[core.Primitive]
    key: PRNGKey
    score: FloatArray
    constraints: ChoiceMap

    def flatten(self):
        return (
            self.key,
            self.score,
            self.constraints,
        ), (self.handles,)

    @classmethod
    def new(cls, key, constraints):
        handles = [gen_fn_p, cache_p]
        score = 0.0
        return Assess(handles, key, score, constraints)

    def trace(self, incells, outcells, addr, tree_in, **kwargs):
        assert self.constraints.has_subtree(addr)

        # We haven't handled the predecessors of this trace
        # call yet, so we return back to the abstract interpreter
        # to continue propagation.
        if static_map_check_bottom(incells):
            return incells, outcells, None

        # Otherwise, we continue with code generation.
        values = static_map_unwrap(incells)
        gen_fn, args = jtu.tree_unflatten(tree_in, values)
        args = tuple(args)
        submap = self.constraints.get_subtree(addr)
        self.key, (v, score) = gen_fn.assess(self.key, submap, args)
        self.score += score

        # Here, we keep the outcells dimension (e.g. the same as
        # the number of outgoing variables in the equation) invariant
        # under our expansion.
        new_outcells = flatmap_outcells(Bare, v)

        return incells, new_outcells, None

    def cache(self, incells, outcells, addr, fn, **kwargs):
        # We haven't handled the predecessors of this trace
        # call yet, so we return back to the abstract interpreter
        # to continue propagation.
        if any(map(lambda v: v.bottom(), incells)):
            return incells, outcells, None

        # Otherwise, we codegen by calling into the function.
        args = tuple(map(lambda v: v.get_val(), incells))
        retval = fn(*args)

        # Here, we keep the outcells dimension (e.g. the same as
        # the number of outgoing variables in the equation) invariant
        # under our expansion.
        new_outcells = flatmap_outcells(Bare, retval)

        return incells, new_outcells, None


def assess_transform(source_fn, **kwargs):
    @functools.wraps(source_fn)
    def _inner(key, constraints, args):
        closed_jaxpr, (flat_args, _, out_tree) = stage(source_fn)(
            *args, **kwargs
        )
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        handler = Assess.new(key, constraints)
        with PropagationInterpreter.new(
            Bare, bare_propagation_rules, handler
        ) as interpreter:
            final_env, _ = interpreter(
                jaxpr,
                [Bare.new(v) for v in consts],
                list(map(Bare.new, flat_args)),
                [Bare.unknown(var.aval) for var in jaxpr.outvars],
            )
            flat_out = safe_map(final_env.read, jaxpr.outvars)
            flat_out = map(lambda v: v.get_val(), flat_out)
            retvals = jtu.tree_unflatten(out_tree, flat_out)
            score = handler.score
            key = handler.key

        return key, (retvals, score)

    return _inner
