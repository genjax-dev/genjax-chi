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

"""This module supports incremental computation using generalized tangents
(e.g. `ChangeTangent` below).

The implementation follows a forward-mode AD approach, with the ability
to customize the primitive rule registry (c.f. Autodidax).
"""

# TODO: Think about when tangents don't share the same Pytree shape as primals.

import abc
import dataclasses
import functools

import jax.core as jc
import jax.tree_util as jtu
from jax import abstract_arrays
from jax import api_util
from jax import linear_util as lu
from jax import util as jax_util

from genjax._src.core.interpreters import staging
from genjax._src.core.interpreters.context import Context
from genjax._src.core.interpreters.context import ContextualTrace
from genjax._src.core.interpreters.context import Fwd
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import Any
from genjax._src.core.typing import Dict
from genjax._src.core.typing import IntArray
from genjax._src.core.typing import Iterable
from genjax._src.core.typing import List
from genjax._src.core.typing import String
from genjax._src.core.typing import Union
from genjax._src.core.typing import typecheck


#######################################
# Change type lattice and propagation #
#######################################

Value = Any


class DiffTracer(jc.Tracer):
    """A `DiffTracer` encapsulates a single value."""

    def __init__(self, trace: "DiffTrace", val: Value, tangent: "ChangeTangent"):
        self._trace = trace
        self.val = val
        self.tangent = tangent

    @property
    def aval(self):
        return abstract_arrays.raise_to_shaped(jc.get_aval(self.val))

    def full_lower(self):
        return self


class DiffTrace(ContextualTrace):
    def pure(self, val: Value) -> DiffTracer:
        return DiffTracer(self, val, NoChange)

    def sublift(self, tracer: DiffTracer) -> DiffTracer:
        return self.pure(tracer.val)

    def lift(self, val: Value) -> DiffTracer:
        return self.pure(val)

    def default_process_primitive(
        self,
        primitive: jc.Primitive,
        tracers: List[DiffTracer],
        params: Dict[String, Any],
    ) -> Union[DiffTracer, List[DiffTracer]]:
        context = staging.get_dynamic_context(self)
        vals = [v.val for v in tracers]
        tangents = [v.tangent for v in tracers]
        check = static_check_no_change(tangents)
        if context.can_process(primitive):
            outvals = context.process_primitive(primitive, *vals, **params)
            return jax_util.safe_map(self.pure, outvals)
        outvals = primitive.bind(*vals, **params)
        if not primitive.multiple_results:
            outvals = [outvals]
        out_tracers = jax_util.safe_map(self.full_raise, outvals)
        if primitive.multiple_results:
            return out_tracers
        if not check:
            for tracer in out_tracers:
                tracer.tangent = UnknownChange
        return out_tracers[0]

    def post_process_call(self, call_primitive, out_tracers, params):
        vals = tuple(t.val for t in out_tracers)
        master = self.main

        def todo(x):
            trace = DiffTrace(master, jc.cur_sublevel())
            return jax_util.safe_map(functools.partial(DiffTracer, trace), x)

        return vals, todo


#####
# Change types
#####


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
        return (self.tangent,), ()

    def should_flatten(self):
        return True


def static_check_is_change_tangent(v):
    return isinstance(v, ChangeTangent)


#####
# Diffs (generalized duals)
#####


@dataclasses.dataclass
class Diff(Pytree):
    primal: Any
    tangent: Any

    def flatten(self):
        return (self.primal, self.tangent), ()

    @classmethod
    def new(cls, primal):
        assert not isinstance(primal, Diff)
        return Diff(primal, UnknownChange)

    def get_primal(self):
        return self.primal

    def get_tracers(self, trace):
        # If we're not in a `DiffTrace` context -
        # we shouldn't try and make DiffTracers.
        if not isinstance(trace, DiffTrace):
            return self.primal
        return DiffTracer(trace, self.primal, self.tangent)

    @typecheck
    @classmethod
    def from_tracer(cls, tracer: DiffTracer):
        if tracer.tangent is None:
            tangent = NoChange
        else:
            tangent = tracer.tangent
        return Diff(tracer.val, tangent)

    @classmethod
    def inflate(cls, tree, change_tangent):
        """Create an instance of `type(tree)` with the same structure as tree,
        but with all values replaced with `change_tangent`"""
        return jtu.tree_map(lambda _: change_tangent, tree)


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


def tree_diff_get_tracers(v, trace):
    def _inner(v):
        if static_check_is_diff(v):
            return v.get_tracers(trace)
        else:
            return v

    return jtu.tree_map(lambda v: _inner(v), v, is_leaf=static_check_is_diff)


#################################
# Generalized tangent transform #
#################################


@lu.transformation
def _jvp(main: jc.MainTrace, ctx: Context, diffs: Iterable[Diff]):
    trace = DiffTrace(main, jc.cur_sublevel())
    in_tracers = jtu.tree_leaves([d.get_tracers(trace) for d in diffs])
    with staging.new_dynamic_context(main, ctx):
        # Give ctx main so that we can new up
        # tracers at the correct level when required.
        ctx.main_trace = main
        ans = yield in_tracers, {}
        out_tracers = jax_util.safe_map(trace.full_raise, ans)
        stateful_tracers = jtu.tree_map(trace.full_raise, ctx.yield_state())
        del main
    stateful_values = jtu.tree_map(lambda x: x.val, stateful_tracers)
    out_diffs = jtu.tree_map(Diff.from_tracer, out_tracers)
    yield out_diffs, stateful_values


# NOTE: There's no constraint on Pytree equality between primals and tangents.
# I'm really not sure about this, more generally.
# How to specify custom tangents for arbitrary Pytree types?
# This solution works for array-like values (e.g. JAX native values)
# but I'm not sure how to generalize to custom tangents for structure.
def jvp(f, ctx: Context):
    # Runs the interpreter.
    def _run_interpreter(main, *args, **kwargs):
        with Fwd.new() as interpreter:
            return interpreter(DiffTrace, main, f, *args, **kwargs)

    # Propagates tracer values through running the interpreter.
    @functools.wraps(f)
    def wrapped(*diffs, **kwargs):
        with jc.new_main(DiffTrace) as main:
            fun = lu.wrap_init(functools.partial(_run_interpreter, main), kwargs)
            primals = jax_util.safe_map(lambda d: d.get_primal(), diffs)
            _, primal_tree = jtu.tree_flatten(primals)
            flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, primal_tree)
            flat_fun = _jvp(flat_fun, main, ctx)
            out_diffs, ctx_statefuls = flat_fun.call_wrapped(diffs)
            del main
        return jtu.tree_unflatten(out_tree(), out_diffs), ctx_statefuls

    return wrapped


##############
# Shorthands #
##############

diff = Diff.new
transform = jvp
