# Copyright 2024 MIT Probabilistic Computing Project
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


import jax
import jax.core as jc
import jax.tree_util as jtu
from jax import util as jax_util
from jax.extend import source_info_util as src_util

from genjax._src.core.interpreters.forward import (
    Environment,
    InitialStylePrimitive,
    initial_style_bind,
)
from genjax._src.core.interpreters.staging import stage
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    ArrayLike,
    BoolArray,
    Callable,
    Generic,
    TypeVar,
)

R = TypeVar("R")
S = TypeVar("S")

record_p = InitialStylePrimitive("record_p")
contract_p = InitialStylePrimitive("contract_p")


@Pytree.dataclass
class Contract(Generic[R, S], Pytree):
    f: Callable[..., R] = Pytree.static()
    pre: Callable[..., BoolArray] = Pytree.static()
    post: Callable[..., BoolArray] = Pytree.static()

    # Default implementation -- just call the function
    # as we would ordinarily do!
    def default_call(self, *args) -> R:
        return self.f(*args)

    def __call__(self, *args):
        def _cont_prim_call(contract, *args):
            return contract.default_call(*args)

        # Bind the primitive with the default implementation.
        return initial_style_bind(contract_p)(_cont_prim_call)(self, *args)


@Pytree.dataclass
class ClosedContract(Generic[R, S], Pytree):
    """
    A closed contract represents a contract to be applied to a set of arguments, and a return value.

    These are created by the `TimeTravelCPSInterpreter` below, they store information from the recording of the execution, and can be checked against the recorded values.
    """

    contract: Contract[R, S]
    args: tuple[Any, ...]
    result: R

    def check(self):
        return self.contract.pre(*self.args) and self.contract.post(self.result)


# Used as a decorator to create a contract on a function.
def contract(pre, post):
    def inner(fn):
        return Contract(fn, pre, post)

    return inner


@Pytree.dataclass
class Frame(Generic[R, S], Pytree):
    """
    A frame represents a point in the computation preceding the application of a function `f` to arguments `args`.

    A frame also stores the contracts which have been accumulated between this point, and the previous frame.
    """

    args: tuple[Any, ...]
    contracts: list[ClosedContract[Any, Any]]
    f: Callable[..., R] = Pytree.static()
    cont: Callable[..., S] = Pytree.static()

    def check(self):
        return all(contract.check() for contract in self.contracts)


@Pytree.dataclass
class Breakpoint(Generic[R, S], Pytree):
    callable: Callable[..., R] = Pytree.static()
    debug_tag: str | None = Pytree.static()

    def default_call(self, *args) -> R:
        return self.callable(*args)

    def handle(
        self,
        contracts: list[ClosedContract[Any, Any]],
        cont: Callable[[R], tuple[S, Any]],
        *args,
    ):
        def _cont(*args) -> S:
            final_ret, _ = cont(self.callable(*args))
            return final_ret

        # Normal execution.
        final_ret = _cont(*args)
        return final_ret, (
            self.debug_tag,
            Frame(args, contracts, self.callable, _cont),
        )

    def __call__(self, *args):
        def _cont_prim_call(brk_pt, *args):
            return brk_pt.default_call(*args)

        return initial_style_bind(record_p)(_cont_prim_call)(self, *args)


def brk(
    callable: Callable[..., R],
    debug_tag: str | None = None,
):
    return Breakpoint(callable, debug_tag)


def tag(v, name=None):
    return brk(lambda v: v, name)(v)


##########################
# Hybrid CPS interpreter #
##########################


@Pytree.dataclass
class TimeTravelCPSInterpreter(Pytree):
    @staticmethod
    def _eval_jaxpr_hybrid_cps(
        jaxpr: jc.Jaxpr,
        consts: list[ArrayLike],
        flat_args: list[ArrayLike],
        out_tree,
    ):
        env = Environment()
        jax_util.safe_map(env.write, jaxpr.constvars, consts)
        jax_util.safe_map(env.write, jaxpr.invars, flat_args)
        closed_contracts = []

        # Hybrid CPS evaluation.
        def eval_jaxpr_iterate_cps(
            eqns,
            env: Environment,
            invars,
            flat_args,
            rebind=False,
        ):
            jax_util.safe_map(env.write, invars, flat_args)

            for eqn_idx, eqn in enumerate(eqns):
                with src_util.user_context(eqn.source_info.traceback):
                    invals = jax_util.safe_map(env.read, eqn.invars)
                    subfuns, params = eqn.primitive.get_bind_params(eqn.params)
                    args = subfuns + invals

                    if eqn.primitive == record_p:
                        env = env.copy()

                        def _kont(*args):
                            leaves = jtu.tree_leaves(args)
                            return eval_jaxpr_iterate_cps(
                                eqns[eqn_idx + 1 :],
                                env,
                                eqn.outvars,
                                leaves,
                                rebind=True,
                            )

                        in_tree = params["in_tree"]
                        num_consts = params["num_consts"]
                        record_pt, *args = jtu.tree_unflatten(
                            in_tree, args[num_consts:]
                        )
                        assert isinstance(record_pt, Breakpoint)
                        if rebind:
                            return _kont(record_pt(*args))

                        else:
                            return record_pt.handle(closed_contracts, _kont, *args)

                    elif eqn.primitive == contract_p:
                        in_tree = params["in_tree"]
                        num_consts = params["num_consts"]
                        contract, *args = jtu.tree_unflatten(in_tree, args[num_consts:])
                        retval = contract.f(*args)
                        closed_contracts.append(
                            ClosedContract(contract, tuple(args), retval)
                        )
                        outs = jtu.tree_leaves(retval)

                    else:
                        outs = eqn.primitive.bind(*args, **params)

                if not eqn.primitive.multiple_results:
                    outs = [outs]

                jax_util.safe_map(
                    env.write,
                    eqn.outvars,
                    outs,
                )

            out_values = jax.util.safe_map(
                env.read,
                jaxpr.outvars,
            )
            retval = jtu.tree_unflatten(out_tree(), out_values)
            return retval, None

        return eval_jaxpr_iterate_cps(
            jaxpr.eqns,
            env,
            jaxpr.invars,
            flat_args,
        )

    @staticmethod
    def time_travel(f):
        def _inner(*args):
            closed_jaxpr, (flat_args, _, out_tree) = stage(f)(*args)
            jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
            return TimeTravelCPSInterpreter._eval_jaxpr_hybrid_cps(
                jaxpr,
                consts,
                flat_args,
                out_tree,
            )

        return _inner


def time_travel(f):
    return TimeTravelCPSInterpreter.time_travel(f)


@Pytree.dataclass
class Debugger(Pytree):
    final_retval: Any
    sequence: list[Frame[Any, Any]]
    jump_points: dict[Any, Any] = Pytree.static()
    ptr: int = Pytree.static()

    def frame(self) -> tuple[str | None, Frame[Any, Any]]:
        frame = self.sequence[self.ptr]
        reverse_jump_points = {v: k for (k, v) in self.jump_points.items()}
        jump_tag = reverse_jump_points.get(self.ptr, None)
        return jump_tag, frame

    @Pytree.dataclass
    class Summary(Pytree):
        retval: Any
        jump_tag: str
        frame: Frame[Any, Any]

    def summary(self) -> Summary:
        frame = self.sequence[self.ptr]
        reverse_jump_points = {v: k for (k, v) in self.jump_points.items()}
        jump_tag = reverse_jump_points.get(self.ptr, None)
        return Debugger.Summary(
            self.final_retval,
            jump_tag,
            frame,
        )

    def jump(self, debug_tag: str) -> "Debugger":
        jump_pt = self.jump_points[debug_tag]
        return Debugger(
            self.final_retval,
            self.sequence,
            self.jump_points,
            jump_pt,
        )

    def fwd(self) -> "Debugger":
        new_ptr = self.ptr + 1
        if new_ptr >= len(self.sequence):
            return self
        else:
            return Debugger(
                self.final_retval,
                self.sequence,
                self.jump_points,
                self.ptr + 1,
            )

    def bwd(self) -> "Debugger":
        new_ptr = self.ptr - 1
        if new_ptr >= len(self.sequence) or new_ptr < 0:
            return self
        else:
            return Debugger(
                self.final_retval,
                self.sequence,
                self.jump_points,
                new_ptr,
            )

    def remix(self, *args) -> "Debugger":
        frame = self.sequence[self.ptr]
        contracts, f, cont = frame.contracts, frame.f, frame.cont
        _, debugger = _record(cont)(*args)
        new_frame = Frame(args, contracts, f, cont)
        return Debugger(
            debugger.final_retval,
            [*self.sequence[: self.ptr], new_frame, *debugger.sequence],
            self.jump_points,
            self.ptr,
        )

    def enter(self) -> "Debugger":
        return self.jump("enter")

    def exit(self) -> "Debugger":
        return self.jump("exit")

    def check(self):
        return self.summary().frame.check()

    def __call__(self, *args):
        return self.remix(*args)


def _record(source: Callable[..., Any]):
    def inner(*args) -> tuple[Any, Debugger]:
        retval, next = time_travel(source)(*args)  # pyright: ignore[reportGeneralTypeIssues]
        sequence = []
        jump_points = {}
        while next:
            (debug_tag, frame) = next
            sequence.append(frame)
            if debug_tag:
                jump_points[debug_tag] = len(sequence) - 1
            args, cont = frame.args, frame.cont
            retval, next = time_travel(cont)(*args)  # pyright: ignore[reportGeneralTypeIssues]
        return retval, Debugger(retval, sequence, jump_points, 0)

    return inner


def debug(source: Callable[..., Any]):
    def instrumented(*args):
        return tag(brk(source, "enter")(*args), "exit")

    def inner(*args) -> Debugger:
        _, debugger = _record(instrumented)(*args)
        return debugger

    return inner
