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
from genjax._src.core.pytree import Closure, Pytree
from genjax._src.core.typing import (
    Any,
    ArrayLike,
    Callable,
    Generic,
    TypeVar,
)

R = TypeVar("R")
S = TypeVar("S")

record_p = InitialStylePrimitive("record_p")


@Pytree.dataclass
class FrameRecording(Generic[R, S], Pytree):
    f: Callable[..., R]
    args: tuple[Any, ...]
    local_retval: R
    cont: Callable[..., S]


@Pytree.dataclass
class RecordPoint(Generic[R, S], Pytree):
    callable: Closure[R]
    debug_tag: str | None = Pytree.static()

    def default_call(self, *args) -> R:
        return self.callable(*args)

    def handle(self, cont: Callable[[R], tuple[S, Any]], *args):
        @Pytree.partial()
        def _cont(*args) -> S:
            final_ret, _ = cont(self.callable(*args))
            return final_ret

        # Normal execution.
        ret = self.callable(*args)
        final_ret = _cont(*args)
        return final_ret, (
            self.debug_tag,
            FrameRecording(self.callable, args, ret, _cont),
        )

    def __call__(self, *args):
        def _cont_prim_call(brk_pt, *args):
            return brk_pt.default_call(*args)

        return initial_style_bind(record_p)(_cont_prim_call)(self, *args)


def rec(
    callable: Callable[..., R],
    debug_tag: str | None = None,
):
    if not isinstance(callable, Closure):
        callable = Closure[R]((), callable)

    def inner(*args):
        return RecordPoint(callable, debug_tag)(*args)

    return inner


def tag(v, name=None):
    return rec(lambda v: v, name)(v)


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

                        @Pytree.partial()
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
                        cps_prim, *args = jtu.tree_unflatten(in_tree, args[num_consts:])
                        if rebind:
                            return _kont(cps_prim(*args))

                        else:
                            return cps_prim.handle(_kont, *args)

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
class TimeTravelingDebugger(Pytree):
    final_retval: Any
    sequence: list[FrameRecording[Any, Any]]
    jump_points: dict[Any, Any] = Pytree.static()
    ptr: int = Pytree.static()

    def frame(self) -> tuple[str | None, FrameRecording[Any, Any]]:
        frame = self.sequence[self.ptr]
        reverse_jump_points = {v: k for (k, v) in self.jump_points.items()}
        jump_tag = reverse_jump_points.get(self.ptr, None)
        return jump_tag, frame

    def summary(self) -> tuple[Any, tuple[str | None, FrameRecording[Any, Any]]]:
        frame = self.sequence[self.ptr]
        reverse_jump_points = {v: k for (k, v) in self.jump_points.items()}
        jump_tag = reverse_jump_points.get(self.ptr, None)
        return self.final_retval, (jump_tag, frame)

    def jump(self, debug_tag: str) -> "TimeTravelingDebugger":
        jump_pt = self.jump_points[debug_tag]
        return TimeTravelingDebugger(
            self.final_retval,
            self.sequence,
            self.jump_points,
            jump_pt,
        )

    def fwd(self) -> "TimeTravelingDebugger":
        new_ptr = self.ptr + 1
        if new_ptr >= len(self.sequence):
            return self
        else:
            return TimeTravelingDebugger(
                self.final_retval,
                self.sequence,
                self.jump_points,
                self.ptr + 1,
            )

    def bwd(self) -> "TimeTravelingDebugger":
        new_ptr = self.ptr - 1
        if new_ptr >= len(self.sequence) or new_ptr < 0:
            return self
        else:
            return TimeTravelingDebugger(
                self.final_retval,
                self.sequence,
                self.jump_points,
                new_ptr,
            )

    def remix(self, *args) -> "TimeTravelingDebugger":
        frame = self.sequence[self.ptr]
        f, cont = frame.f, frame.cont
        local_retval = f(*args)
        _, debugger = _record(cont)(*args)
        new_frame = FrameRecording(f, args, local_retval, cont)
        return TimeTravelingDebugger(
            debugger.final_retval,
            [*self.sequence[: self.ptr], new_frame, *debugger.sequence],
            self.jump_points,
            self.ptr,
        )

    def __call__(self, *args):
        return self.remix(*args)


def _record(source: Callable[..., Any]):
    def inner(*args) -> tuple[Any, TimeTravelingDebugger]:
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
        return retval, TimeTravelingDebugger(retval, sequence, jump_points, 0)

    return inner


def time_machine(source: Callable[..., Any]):
    def instrumented(*args):
        return tag(rec(source, "_enter")(*args), "exit")

    def inner(*args) -> TimeTravelingDebugger:
        _, debugger = _record(instrumented)(*args)
        return debugger

    return inner
