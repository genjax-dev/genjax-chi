# Copyright 2024 The MIT Probabilistic Computing Project & the oryx authors.
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

import functools
from abc import abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from genjax._src.core.generative import (
    Address,
    ChangeTargetUpdateSpec,
    ChoiceMap,
    Constraint,
    GenerativeFunction,
    RemoveSelectionUpdateSpec,
    Trace,
    UpdateSpec,
)
from genjax._src.core.interpreters.forward import (
    InitialStylePrimitive,
    StatefulHandler,
    forward,
    initial_style_bind,
)
from genjax._src.core.interpreters.incremental import (
    Diff,
    incremental,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.traceback_util import register_exclusion
from genjax._src.core.typing import (
    FloatArray,
    List,
    PRNGKey,
    Tuple,
    static_check_is_concrete,
    typecheck,
)
from genjax.core.exceptions import AddressReuse, StaticAddressJAX

register_exclusion(__file__)

##############
# Primitives #
##############

# Generative function trace intrinsic.
trace_p = InitialStylePrimitive("trace")


##################
# Address checks #
##################


# Usage in intrinsics: ensure that addresses do not contain JAX traced values.
def static_check_address_type(addr):
    check = all(jtu.tree_leaves(jtu.tree_map(static_check_is_concrete, addr)))
    if not check:
        raise StaticAddressJAX(addr)


############################################################
# Trace call (denotes invocation of a generative function) #
############################################################


# We defer the abstract call here so that, when we
# stage, any traced values stored in `gen_fn`
# get lifted to by `get_shaped_aval`.
def _abstract_gen_fn_call(addr, gen_fn, args):
    return gen_fn.__abstract_call__(*args)


@typecheck
def trace(
    addr: Address,
    gen_fn: GenerativeFunction,
    args: Tuple,
):
    """Invoke a generative function, binding its generative semantics with the current
    caller.

    Arguments:
        addr: An address denoting the site of a generative function invocation.
        gen_fn: A generative function invoked as a callee of `StaticGenerativeFunction`.
    """
    static_check_address_type(addr)
    addr = Pytree.tree_const(addr)
    return initial_style_bind(trace_p)(_abstract_gen_fn_call)(
        addr,
        gen_fn,
        args,
    )


######################################
#  Generative function interpreters  #
######################################


# Usage in transforms: checks for duplicate addresses.
@Pytree.dataclass
class AddressVisitor(Pytree):
    visited: List = Pytree.field(default_factory=list)

    def visit(self, addr):
        if addr in self.visited:
            raise AddressReuse(addr)
        else:
            self.visited.append(addr)

    def get_visited(self):
        return self.visited


###########################
# Static language handler #
###########################


# This explicitly makes assumptions about some common fields:
# e.g. it assumes if you are using `StaticHandler.get_submap`
# in your code, that your derived instance has a `constraints` field.
@dataclass
class StaticHandler(StatefulHandler):
    @abstractmethod
    def handle_trace(*traces, **_params):
        raise NotImplementedError

    # By default, the interpreter handlers for this language
    # handle the two primitives we defined above
    # (`trace_p`, for random choices)
    def handles(self, prim):
        return prim == trace_p

    def dispatch(self, prim, *tracers, **_params):
        in_tree = _params.get("in_tree")
        num_consts = _params.get("num_consts")
        non_const_tracers = tracers[num_consts:]
        addr, gen_fn, args = jtu.tree_unflatten(in_tree, non_const_tracers)
        if prim == trace_p:
            v = self.handle_trace(addr, gen_fn, args)
            return jtu.tree_leaves(v)
        else:
            raise Exception("Illegal primitive: {}".format(prim))


############
# Simulate #
############


@dataclass
class SimulateHandler(StaticHandler):
    key: PRNGKey
    score: FloatArray = Pytree.field(default_factory=lambda: jnp.zeros(()))
    address_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)
    address_traces: List[Trace] = Pytree.field(default_factory=list)

    def visit(self, addr):
        self.address_visitor.visit(addr)

    def yield_state(self):
        return (
            self.address_visitor,
            self.address_traces,
            self.score,
        )

    def handle_trace(self, addr, gen_fn, args):
        self.visit(addr)
        self.key, sub_key = jax.random.split(self.key)
        tr = gen_fn.simulate(sub_key, args)
        score = tr.get_score()
        self.address_traces.append(tr)
        self.score += score
        v = tr.get_retval()
        return v


def simulate_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(key, args):
        stateful_handler = SimulateHandler(key)
        retval = forward(source_fn)(stateful_handler, *args)
        (
            address_visitor,
            address_traces,
            score,
        ) = stateful_handler.yield_state()
        return (
            args,
            retval,
            address_visitor,
            address_traces,
            score,
        )

    return wrapper


##############
# Importance #
##############


@dataclass
class ImportanceHandler(StaticHandler):
    key: PRNGKey
    constraint: Constraint
    score: FloatArray = Pytree.field(default_factory=lambda: jnp.zeros(()))
    weight: FloatArray = Pytree.field(default_factory=lambda: jnp.zeros(()))
    address_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)
    address_traces: List[Trace] = Pytree.field(default_factory=list)
    bwd_specs: List[ChoiceMap] = Pytree.field(default_factory=list)

    def yield_state(self):
        return (
            self.score,
            self.weight,
            self.address_visitor,
            self.address_traces,
            self.bwd_specs,
        )

    def visit(self, addr):
        self.address_visitor.visit(addr)

    def get_subconstraint(self, addr):
        addr = Pytree.tree_unwrap_const(addr)
        match self.constraint:
            case ChoiceMap():
                return self.constraint.get_submap(addr)

            case _:
                raise ValueError(f"Not implemented fwd_spec: {self.fwd_spec}")

    def handle_trace(self, *tracers, **_params):
        in_tree = _params.get("in_tree")
        num_consts = _params.get("num_consts")
        passed_in_tracers = tracers[num_consts:]
        gen_fn, addr = jtu.tree_unflatten(in_tree, passed_in_tracers)
        self.visit(addr)
        sub_map = self.get_subconstraint(addr)
        self.key, sub_key = jax.random.split(self.key)
        tr, w, bwd_spec = gen_fn.importance(sub_key, sub_map)
        self.address_traces.append(tr)
        self.bwd_specs.append(bwd_spec)
        self.score += tr.get_score()
        self.weight += w
        v = tr.get_retval()
        return jtu.tree_leaves(v)


def importance_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(key, constraints, args):
        stateful_handler = ImportanceHandler(key, constraints)
        retval = forward(source_fn)(stateful_handler, *args)
        (
            score,
            weight,
            address_visitor,
            address_traces,
            bwd_specs,
        ) = stateful_handler.yield_state()
        return (
            weight,
            (
                args,
                retval,
                address_visitor,
                address_traces,
                score,
            ),
            bwd_specs,
        )

    return wrapper


##########
# Update #
##########


@dataclass
class UpdateHandler(StaticHandler):
    key: PRNGKey
    previous_trace: Trace
    fwd_spec: UpdateSpec
    address_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)
    score: FloatArray = Pytree.field(default_factory=lambda: jnp.zeros(()))
    weight: FloatArray = Pytree.field(default_factory=lambda: jnp.zeros(()))
    address_traces: List[Trace] = Pytree.field(default_factory=list)
    bwd_specs: List[ChoiceMap] = Pytree.field(default_factory=list)

    def yield_state(self):
        return (
            self.score,
            self.weight,
            self.address_visitor,
            self.address_traces,
            self.bwd_specs,
        )

    def visit(self, addr):
        self.address_visitor.visit(addr)

    def get_subspec(self, addr):
        addr = Pytree.tree_unwrap_const(addr)
        match self.fwd_spec:
            case ChoiceMap():
                return self.fwd_spec.get_submap(addr)

            case RemoveSelectionUpdateSpec(selection):
                subselection = selection.step(addr)
                return RemoveSelectionUpdateSpec(subselection)

            case _:
                raise ValueError(f"Not implemented fwd_spec: {self.fwd_spec}")

    def get_subtrace(self, addr):
        addr = Pytree.tree_unwrap_const(addr)
        return self.previous_trace.get_subtrace(addr)

    def handle_trace(self, *tracers, **_params):
        in_tree = _params.get("in_tree")
        num_consts = _params.get("num_consts")
        argdiffs = tracers[num_consts:]
        primals = Diff.tree_primal(argdiffs)
        gen_fn, addr = jtu.tree_unflatten(in_tree, primals)
        self.visit(addr)

        # Run the update step.
        subtrace = self.get_subtrace(addr)
        subspec = self.get_subspec(addr)
        self.key, sub_key = jax.random.split(self.key)
        fwd_spec = ChangeTargetUpdateSpec(argdiffs, subspec)
        (tr, w, retval_diff, bwd_spec) = gen_fn.update(sub_key, subtrace, fwd_spec)
        self.score += tr.get_score()
        self.weight += w
        self.address_traces.append(tr)
        self.bwd_specs.append(bwd_spec)

        # We have to convert the Diff back to tracers to return
        # from the primitive.
        return jtu.tree_leaves(retval_diff, is_leaf=lambda v: isinstance(v, Diff))


def update_transform(source_fn):
    @functools.wraps(source_fn)
    @typecheck
    def wrapper(key, previous_trace, constraints, diffs: Tuple):
        stateful_handler = UpdateHandler(key, previous_trace, constraints)
        diff_primals = Diff.tree_primal(diffs)
        diff_tangents = Diff.tree_tangent(diffs)
        retval_diffs = incremental(source_fn)(
            stateful_handler, diff_primals, diff_tangents
        )
        retval_primals = Diff.tree_primal(retval_diffs)
        (
            score,
            weight,
            address_visitor,
            address_traces,
            bwd_specs,
        ) = stateful_handler.yield_state()
        return (
            (
                retval_diffs,
                weight,
                # Trace.
                (
                    diff_primals,
                    retval_primals,
                    address_visitor,
                    address_traces,
                    score,
                ),
                # Backward update spec.
                bwd_specs,
            ),
        )

    return wrapper


##########
# Assess #
##########


@dataclass
class AssessHandler(StaticHandler):
    constraints: ChoiceMap
    score: FloatArray = Pytree.field(default_factory=lambda: jnp.zeros(()))
    address_visitor: AddressVisitor = Pytree.field(default_factory=AddressVisitor)

    def yield_state(self):
        return (self.score,)

    def handle_trace(self, *tracers, **_params):
        in_tree = _params.get("in_tree")
        num_consts = _params.get("num_consts")
        passed_in_tracers = tracers[num_consts:]
        gen_fn, addr, *args = jtu.tree_unflatten(in_tree, passed_in_tracers)
        self.visit(addr)
        args = tuple(args)
        submap = self.get_submap(addr)
        (score, v) = gen_fn.assess(submap, args)
        self.score += score
        return jtu.tree_leaves(v)


def assess_transform(source_fn):
    @functools.wraps(source_fn)
    def wrapper(constraints, args):
        stateful_handler = AssessHandler(constraints)
        retval = forward(source_fn)(stateful_handler, *args)
        (score,) = stateful_handler.yield_state()
        return (retval, score)

    return wrapper
