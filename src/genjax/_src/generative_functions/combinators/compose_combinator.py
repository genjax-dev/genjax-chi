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

from equinox import module_update_wrapper

from genjax._src.core.generative import (
    ChoiceMap,
    GenerativeFunction,
    Selection,
    Trace,
)
from genjax._src.core.interpreters.incremental import Diff, incremental
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    Callable,
    FloatArray,
    PRNGKey,
    Tuple,
    typecheck,
)


class ComposeTrace(Trace):
    compose_combinator: "ComposeCombinator"
    inner: Trace
    args: Tuple
    retval: Any

    def get_gen_fn(self):
        return self.compose_combinator

    def get_choices(self):
        return self.inner.get_choices()

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.inner.get_score()

    def get_args(self):
        return self.args

    def project(
        self,
        key: PRNGKey,
        selection: Selection,
    ) -> FloatArray:
        return self.inner.project(key, selection)


class ComposeCombinator(GenerativeFunction):
    inner: GenerativeFunction
    argument_pushforward: Callable = Pytree.static()
    retval_pushforward: Callable = Pytree.static()

    @typecheck
    def simulate(
        self,
        key: PRNGKey,
        args: Tuple,
    ) -> ComposeTrace:
        inner_args = self.argument_pushforward(*args)
        tr = self.inner.simulate(key, inner_args)
        inner_retval = tr.get_retval()
        retval = self.retval_pushforward(args, inner_retval)
        return ComposeTrace(self, tr, args, retval)

    @typecheck
    def importance(
        self,
        key: PRNGKey,
        constraints: ChoiceMap,
        args: Tuple,
    ) -> Tuple[ComposeTrace, FloatArray]:
        inner_args = self.argument_pushforward(*args)
        tr, w = self.inner.importance(key, constraints, inner_args)
        inner_retval = tr.get_retval()
        retval = self.retval_pushforward(inner_retval)
        return ComposeTrace(self, tr, args, retval), w

    @typecheck
    def update(
        self,
        key: PRNGKey,
        trace: ComposeTrace,
        constraints: ChoiceMap,
        argdiffs: Tuple,
    ) -> Tuple[ComposeTrace, FloatArray, Any, Any]:
        diff_primals = Diff.tree_primal(argdiffs)
        diff_tangents = Diff.tree_tangent(argdiffs)
        inner_argdiffs = incremental(self.argument_pushforward)(
            None, diff_primals, diff_tangents
        )
        tr, w, inner_retval_diff, d = self.inner.update(
            key, trace.inner, constraints, inner_argdiffs
        )
        inner_retval_primals = Diff.tree_primal(inner_retval_diff)
        inner_retval_tangents = Diff.tree_tangent(inner_retval_diff)
        retval_diff = incremental(self.retval_pushforward)(
            None,
            (diff_primals, inner_retval_primals),
            (diff_tangents, inner_retval_tangents),
        )
        retval_primal = Diff.tree_primal(retval_diff)
        return (ComposeTrace(self, tr, diff_primals, retval_primal), w, retval_diff, d)

    @typecheck
    def assess(
        self,
        constraints: ChoiceMap,
        args: Tuple,
    ) -> Tuple[FloatArray, Any]:
        inner_args = self.argument_pushforward(*args)
        w, inner_retval = self.inner.assess(constraints, inner_args)
        retval = self.retval_pushforward(args, inner_retval)
        return w, retval

    @property
    def __wrapped__(self):
        return self.inner


#############
# Decorator #
#############


def compose_combinator(f, precompose, postcompose) -> ComposeCombinator:
    return module_update_wrapper(
        ComposeCombinator(f, precompose, postcompose),
    )
