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


from genjax._src.core.generative import (
    Argdiffs,
    EmptyTrace,
    GenerativeFunction,
    GenericProblem,
    Retdiff,
    Sample,
    Score,
    Trace,
    UpdateProblem,
    Weight,
)
from genjax._src.core.interpreters.incremental import Diff, incremental
from genjax._src.core.pytree import Pytree
from genjax._src.core.traceback_util import register_exclusion
from genjax._src.core.typing import (
    Any,
    Callable,
    Optional,
    PRNGKey,
    String,
    Tuple,
    typecheck,
)

register_exclusion(__file__)


@Pytree.dataclass
class DimapTrace(Trace):
    dimap_combinator: "DimapCombinator"
    inner: Trace
    args: Tuple
    retval: Any

    def get_args(self):
        return self.args

    def get_gen_fn(self):
        return self.dimap_combinator

    def get_sample(self):
        return self.inner.get_sample()

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.inner.get_score()


@Pytree.dataclass
class DimapCombinator(GenerativeFunction):
    inner: GenerativeFunction
    argument_mapping: Callable = Pytree.static()
    retval_mapping: Callable = Pytree.static()
    info: Optional[String] = Pytree.static(default=None)

    @GenerativeFunction.gfi_boundary
    @typecheck
    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> DimapTrace:
        inner_args = self.argument_mapping(*args)
        tr = self.inner.simulate(key, inner_args)
        inner_retval = tr.get_retval()
        retval = self.retval_mapping(inner_args, inner_retval)
        return DimapTrace(self, tr, args, retval)

    @typecheck
    def update_change_target(
        self,
        key: PRNGKey,
        trace: Trace,
        update_problem: UpdateProblem,
        argdiffs: Argdiffs,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        assert isinstance(trace, EmptyTrace | DimapTrace)
        primals = Diff.tree_primal(argdiffs)
        tangents = Diff.tree_tangent(argdiffs)
        inner_argdiffs = incremental(self.argument_mapping)(
            None,
            primals,
            tangents,
        )
        match trace:
            case DimapTrace():
                inner_trace = trace.inner
            case EmptyTrace():
                inner_trace = EmptyTrace(self.inner)
        tr, w, inner_retdiff, bwd_problem = self.inner.update(
            key, inner_trace, GenericProblem(inner_argdiffs, update_problem)
        )
        inner_retval_primals = Diff.tree_primal((inner_retdiff,))
        inner_retval_tangents = Diff.tree_tangent((inner_retdiff,))

        def closed_mapping(args, retval):
            return self.retval_mapping(args, retval)

        retval_diff = incremental(closed_mapping)(
            None,
            (primals, inner_retval_primals),
            (tangents, inner_retval_tangents),
        )
        retval_primal = Diff.tree_primal(retval_diff)
        return (
            DimapTrace(self, tr, primals, retval_primal),
            w,
            retval_diff,
            bwd_problem,
        )

    @typecheck
    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        update_problem: UpdateProblem,
    ) -> Tuple[Trace, Weight, Retdiff, UpdateProblem]:
        match update_problem:
            case GenericProblem(argdiffs, subproblem):
                return self.update_change_target(key, trace, subproblem, argdiffs)
            case _:
                return self.update_change_target(
                    key, trace, update_problem, Diff.no_change(trace.get_args())
                )

    @typecheck
    def assess(
        self,
        sample: Sample,
        args: Tuple,
    ) -> Tuple[Score, Any]:
        inner_args = self.argument_mapping(*args)
        w, inner_retval = self.inner.assess(sample, inner_args)
        retval = self.retval_mapping(args, inner_retval)
        return w, retval


#############
# Decorator #
#############


def dimap(
    *,
    pre: Callable = lambda *args: args,
    post: Callable = lambda _args, retval: retval,
    info: Optional[String] = None,
) -> Callable[[GenerativeFunction], DimapCombinator]:
    def decorator(f) -> DimapCombinator:
        return DimapCombinator(f, pre, post, info)

    return decorator


def map(
    f: Callable,
    *,
    info: Optional[String] = None,
) -> Callable[[GenerativeFunction], DimapCombinator]:
    return dimap(pre=lambda *args: args, post=lambda _, ret: f(ret), info=info)


def contramap(
    f: Callable,
    *,
    info: Optional[String] = None,
) -> Callable[[GenerativeFunction], DimapCombinator]:
    return dimap(pre=f, post=lambda _, ret: ret, info=info)
