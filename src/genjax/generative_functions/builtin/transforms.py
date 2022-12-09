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

import jax.tree_util as jtu
from jax import core as jax_core
from jax.interpreters import xla

from genjax.core.datatypes import EmptyChoiceMap
from genjax.core.hashabledict import hashabledict
from genjax.core.propagate import Cell
from genjax.core.propagate import Handler
from genjax.core.propagate import PropagationRules
from genjax.core.propagate import flat_propagate
from genjax.core.propagate import map_outcells
from genjax.core.propagate import propagate
from genjax.core.specialization import is_concrete
from genjax.core.staging import get_shaped_aval
from genjax.core.staging import stage
from genjax.generative_functions.builtin.builtin_datatypes import (
    BuiltinChoiceMap,
)
from genjax.generative_functions.builtin.builtin_datatypes import BuiltinTrie
from genjax.generative_functions.builtin.intrinsics import cache_p
from genjax.generative_functions.builtin.intrinsics import gen_fn_p
from genjax.generative_functions.diff_rules import Diff
from genjax.generative_functions.diff_rules import NoChange
from genjax.generative_functions.diff_rules import check_no_change
from genjax.generative_functions.diff_rules import diff_propagation_rules
from genjax.generative_functions.diff_rules import strip_diff


safe_map = jax_core.safe_map
safe_zip = jax_core.safe_zip

#####
# Utilities
#####


def static_check_not_none(key, retvals):
    assert key is not None
    for r in retvals:
        assert r is not None


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
        flat_out = prim.bind(*in_vals, **params)
        new_out = [Bare.new(flat_out)]
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
bare_call_rules[jax_core.call_p] = functools.partial(
    bare_call_p_rule, jax_core.call_p
)

bare_propagation_rules = PropagationRules(bare_fallback_rule, bare_call_rules)

######################################
#  Generative function interpreters  #
######################################

#####
# Simulate
#####


class Simulate(Handler):
    def __init__(self):
        self.handles = [gen_fn_p, cache_p]
        self.choice_state = BuiltinChoiceMap(hashabledict())
        self.cache_state = BuiltinTrie(hashabledict())
        self.score = 0.0

    def trace(self, incells, outcells, addr, gen_fn, tree_in, **kwargs):
        # We haven't handled the predecessors of this trace
        # call yet, so we return back to the abstract interpreter
        # to continue propagation.
        if any(map(lambda v: v.bottom(), incells)):
            return incells, outcells, None

        values = map(lambda v: v.get_val(), incells)
        key, *args = jtu.tree_unflatten(tree_in, values)
        args = tuple(args)

        # Otherwise, we send simulate down to the generative function
        # callee.
        key, tr = gen_fn.simulate(key, args, **kwargs)
        score = tr.get_score()
        self.choice_state[addr] = tr
        self.score += score

        # Here, we keep the outcells dimension (e.g. the same as
        # the number of outgoing variables in the equation) invariant
        # under our expansion.
        v = tr.get_retval()
        new_v_outcells = map_outcells(Bare, v)
        new_outcells = [Bare.new(key), *new_v_outcells]

        return incells, new_outcells, None

    def cache(self, incells, outcells, addr, fn, tree_in, **kwargs):
        # We haven't handled the predecessors of this trace
        # call yet, so we return back to the abstract interpreter
        # to continue propagation.
        if any(map(lambda v: v.bottom(), incells)):
            return incells, outcells, None

        values = map(lambda v: v.get_val(), incells)
        args = jtu.tree_unflatten(tree_in, values)

        # Otherwise, we codegen by calling into the function.
        retval = fn(*args)
        self.cache_state[addr] = retval

        # Here, we keep the outcells dimension (e.g. the same as
        # the number of outgoing variables in the equation) invariant
        # under our expansion.
        new_outcells = map_outcells(Bare, retval)

        return incells, new_outcells, None


def simulate_transform(f, **kwargs):
    def _inner(key, args):
        closed_jaxpr, (flat_args, in_tree, out_tree) = stage(f)(
            key, *args, **kwargs
        )
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        handler = Simulate()
        final_env, ret_state = propagate(
            Bare,
            bare_propagation_rules,
            jaxpr,
            [Bare.new(v) for v in consts],
            list(map(Bare.new, flat_args)),
            [Bare.unknown(var.aval) for var in jaxpr.outvars],
            handler=handler,
        )
        flat_out = safe_map(final_env.read, jaxpr.outvars)
        flat_out = map(lambda v: v.get_val(), flat_out)
        key_and_returns = jtu.tree_unflatten(out_tree, flat_out)
        key, *retvals = key_and_returns
        retvals = tuple(retvals)
        score = handler.score
        chm = handler.choice_state
        cache = handler.cache_state

        # If propagation succeeded, no value should be
        # None at this point.
        static_check_not_none(key, retvals)
        if len(retvals) == 1:
            retvals = retvals[0]

        return key, (f, args, retvals, chm, score), cache

    return _inner


#####
# Importance
#####


class Importance(Handler):
    def __init__(self, constraints):
        self.handles = [gen_fn_p, cache_p]
        self.choice_state = BuiltinChoiceMap(hashabledict())
        self.cache_state = BuiltinTrie(hashabledict())
        self.score = 0.0
        self.weight = 0.0
        self.constraints = constraints

    def trace(self, incells, outcells, addr, gen_fn, tree_in, **kwargs):
        # We haven't handled the predecessors of this trace
        # call yet, so we return back to the abstract interpreter
        # to continue propagation.
        if any(map(lambda v: v.bottom(), incells)):
            return incells, outcells, None

        # Otherwise, we proceed with code generation.
        values = map(lambda v: v.get_val(), incells)
        key, *args = jtu.tree_unflatten(tree_in, values)
        args = tuple(args)
        if self.constraints.has_subtree(addr):
            sub_map = self.constraints.get_subtree(addr)
        else:
            sub_map = EmptyChoiceMap()
        key, (w, tr) = gen_fn.importance(key, sub_map, args)
        self.choice_state[addr] = tr
        self.score += tr.get_score()
        self.weight += w

        # Here, we keep the outcells dimension (e.g. the same as
        # the number of outgoing variables in the equation) invariant
        # under our expansion.
        v = tr.get_retval()
        new_v_outcells = map_outcells(Bare, v)
        new_outcells = [Bare.new(key), *new_v_outcells]

        return incells, new_outcells, None

    def cache(self, incells, outcells, addr, fn, **kwargs):
        # We haven't handled the predecessors of this trace
        # call yet, so we return back to the abstract interpreter
        # to continue propagation.
        if any(map(lambda v: v.bottom(), incells)):
            return incells, outcells, None

        # Otherwise, we proceed with code generation.
        args = tuple(map(lambda v: v.get_val(), incells))
        retval = fn(*args)
        self.cache_state[addr] = retval

        # Here, we keep the outcells dimension (e.g. the same as
        # the number of outgoing variables in the equation) invariant
        # under our expansion.
        new_outcells = map_outcells(Bare, retval)

        return incells, new_outcells, None


def importance_transform(f, **kwargs):
    def _inner(key, chm, args):
        closed_jaxpr, (flat_args, in_tree, out_tree) = stage(f)(
            key, *args, **kwargs
        )
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        handler = Importance(chm)
        final_env, ret_state = propagate(
            Bare,
            bare_propagation_rules,
            jaxpr,
            [Bare.new(v) for v in consts],
            list(map(Bare.new, flat_args)),
            [Bare.unknown(var.aval) for var in jaxpr.outvars],
            handler=handler,
        )
        flat_out = safe_map(final_env.read, jaxpr.outvars)
        flat_out = map(lambda v: v.get_val(), flat_out)
        key_and_returns = jtu.tree_unflatten(out_tree, flat_out)
        key, *retvals = key_and_returns
        retvals = tuple(retvals)
        w = handler.weight
        score = handler.score
        chm = handler.choice_state
        cache = handler.cache_state

        # If propagation succeeded, no value should be
        # None at this point.
        static_check_not_none(key, retvals)
        if len(retvals) == 1:
            retvals = retvals[0]

        return key, (w, (f, args, retvals, chm, score)), cache

    return _inner


#####
# Update
#####


class Update(Handler):
    def __init__(self, prev, new):
        self.handles = [gen_fn_p, cache_p]
        self.choice_state = BuiltinChoiceMap(hashabledict())
        self.cache_state = BuiltinTrie(hashabledict())
        self.discard = BuiltinChoiceMap(hashabledict())
        self.weight = 0.0
        self.prev = prev
        self.choice_change = new

    def trace(self, incells, outcells, addr, gen_fn, tree_in, **kwargs):
        # We haven't handled the predecessors of this trace
        # call yet, so we return back to the abstract interpreter
        # to continue propagation.
        if any(map(lambda v: v.bottom(), incells)):
            return incells, outcells, None

        key, *diffs = jtu.tree_unflatten(tree_in, incells)
        key = key.get_val()
        diffs = tuple(diffs)
        has_previous = self.prev.has_subtree(addr)
        constrained = self.choice_change.has_subtree(addr)

        # If no changes, we can just short-circuit.
        if (
            is_concrete(has_previous)
            and is_concrete(constrained)
            and has_previous
            and not constrained
            and all(map(check_no_change, incells))
        ):
            prev = self.prev.get_subtree(addr)
            self.choice_state[addr] = prev

            # Here, we keep the outcells dimension (e.g. the same as
            # the number of outgoing variables in the equation) invariant
            # under our expansion.
            v = prev.get_retval()
            new_outcells = [
                Diff.new(key, change=NoChange),
                *map_outcells(Bare, v, change=NoChange),
            ]
            return (
                incells,
                new_outcells,
                None,
            )

        # Otherwise, we proceed with code generation.
        prev_tr = self.prev.get_subtree(addr)
        diffs = tuple(diffs)
        if constrained:
            chm = self.choice_change.get_subtree(addr)
        else:
            chm = EmptyChoiceMap()
        key, (retval_diff, w, tr, discard) = gen_fn.update(
            key, prev_tr, chm, diffs
        )

        self.weight += w
        self.choice_state[addr] = tr
        self.discard[addr] = discard

        # Here, we keep the outcells dimension (e.g. the same as
        # the number of outgoing variables in the equation) invariant
        # under our expansion.
        new_outcells = [
            Diff.new(key),
            *map_outcells(Diff, retval_diff),
        ]

        return incells, new_outcells, None

    def cache(self, incells, outcells, addr, fn, **kwargs):
        # We haven't handled the predecessors of this trace
        # call yet, so we return back to the abstract interpreter
        # to continue propagation.
        if any(map(lambda v: v.bottom(), incells)):
            return incells, outcells, None

        args = tuple(map(lambda v: v.get_val(), incells))
        has_value = self.prev.has_cached_value(addr)

        # If no changes, we can just fetch from trace.
        if (
            is_concrete(has_value)
            and has_value
            and all(map(check_no_change, incells))
        ):
            cached_value = self.prev.get_cached_value(addr)
            self.cache_state[addr] = cached_value

            # Here, we keep the outcells dimension (e.g. the same as
            # the number of outgoing variables in the equation) invariant
            # under our expansion.
            new_outcells = map_outcells(
                Diff,
                cached_value,
                change=NoChange,
            )

            return incells, new_outcells, None

        # Otherwise, we codegen by calling into the function.
        retval = fn(*args)
        self.cache_state[addr] = retval

        # Here, we keep the outcells dimension (e.g. the same as
        # the number of outgoing variables in the equation) invariant
        # under our expansion.
        new_outcells = map_outcells(Bare, retval)

        return incells, new_outcells, None


def check_diff_leaf(v):
    return isinstance(v, Diff) or isinstance(v, Cell)


def update_transform(f, **kwargs):
    def _inner(key, prev, new, diffs):
        vals = jtu.tree_map(strip_diff, diffs, is_leaf=check_diff_leaf)
        _, diff_tree = jtu.tree_flatten(diffs)
        jaxpr, (flat_args, in_tree, out_tree) = stage(f)(key, *vals, **kwargs)
        flat_diffs = jtu.tree_flatten(diffs, is_leaf=check_diff_leaf)
        jaxpr, consts = jaxpr.jaxpr, jaxpr.literals
        handler = Update(prev, new)
        flat_diffs = jtu.tree_flatten(diffs, is_leaf=check_diff_leaf)
        final_env, _ = propagate(
            Diff,
            diff_propagation_rules,
            jaxpr,
            [Diff.new(v, change=NoChange) for v in consts],
            [
                Diff.new(key),
                *jtu.tree_unflatten(diff_tree, flat_args[1:]),
            ],
            [Diff.unknown(var.aval) for var in jaxpr.outvars],
            handler=handler,
        )

        key, *retval_diffs = safe_map(final_env.read, jaxpr.outvars)
        w = handler.weight
        chm = handler.choice_state
        cache = handler.cache_state
        discard = handler.discard
        key = key.get_val()
        retvals = tuple(map(strip_diff, retval_diffs))
        retval_diffs = tuple(retval_diffs)

        # If propagation succeeded, no value should be
        # None at this point.
        static_check_not_none(key, retvals)
        if len(retvals) == 1:
            retvals = retvals[0]

        return (
            key,
            (
                retval_diffs,
                w,
                (f, vals, retvals, chm, prev.get_score() + w),
                discard,
            ),
            cache,
        )

    return _inner


#####
# Assess
#####


class Assess(Handler):
    def __init__(self, provided):
        self.handles = [gen_fn_p, cache_p]
        self.provided = provided
        self.score = 0.0

    def trace(self, incells, outcells, addr, gen_fn, tree_in, **kwargs):
        assert self.provided.has_subtree(addr)

        # We haven't handled the predecessors of this trace
        # call yet, so we return back to the abstract interpreter
        # to continue propagation.
        if any(map(lambda v: v.bottom(), incells)):
            return incells, outcells, None

        # Otherwise, we continue with code generation.
        values = map(lambda v: v.get_val(), incells)
        key, *args = jtu.tree_unflatten(tree_in, values)
        args = tuple(args)
        submap = self.provided.get_subtree(addr)
        key, (v, score) = gen_fn.assess(key, submap, args)
        self.score += score

        # Here, we keep the outcells dimension (e.g. the same as
        # the number of outgoing variables in the equation) invariant
        # under our expansion.
        new_v_outcells = map_outcells(Bare, v)
        new_outcells = [Bare.new(key), *new_v_outcells]

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
        new_outcells = map_outcells(Bare, retval)

        return incells, new_outcells, None


def assess_transform(f, **kwargs):
    def _inner(key, chm, args):
        closed_jaxpr, (flat_args, in_tree, out_tree) = stage(f)(
            key, *args, **kwargs
        )
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        handler = Assess(chm)
        final_env, ret_state = propagate(
            Bare,
            bare_propagation_rules,
            jaxpr,
            [Bare.new(v) for v in consts],
            list(map(Bare.new, flat_args)),
            [Bare.unknown(var.aval) for var in jaxpr.outvars],
            handler=handler,
        )
        flat_out = safe_map(final_env.read, jaxpr.outvars)
        flat_out = map(lambda v: v.get_val(), flat_out)
        key_and_returns = jtu.tree_unflatten(out_tree, flat_out)
        key, *retvals = key_and_returns
        retvals = tuple(retvals)
        score = handler.score

        # If propagation succeeded, no value should be
        # None at this point.
        static_check_not_none(key, retvals)
        if len(retvals) == 1:
            retvals = retvals[0]

        return key, (retvals, score)

    return _inner
