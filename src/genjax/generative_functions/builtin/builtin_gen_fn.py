# Copyright 2022 MIT Probabilistic Computing Project
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

from dataclasses import dataclass
from typing import Callable
from typing import Tuple

import jax

from genjax.core.datatypes import ChoiceMap
from genjax.core.datatypes import GenerativeFunction
from genjax.core.datatypes import Selection
from genjax.core.datatypes import Trace
from genjax.core.masks import BooleanMask
from genjax.generative_functions.builtin.builtin_datatypes import (
    BuiltinChoiceMap,
)
from genjax.generative_functions.builtin.builtin_datatypes import BuiltinTrace
from genjax.generative_functions.builtin.builtin_tracetype import (
    get_trace_type,
)
from genjax.generative_functions.builtin.handlers import handler_arg_grad
from genjax.generative_functions.builtin.handlers import handler_choice_grad
from genjax.generative_functions.builtin.handlers import handler_importance
from genjax.generative_functions.builtin.handlers import handler_simulate
from genjax.generative_functions.builtin.handlers import handler_update


@dataclass
class BuiltinGenerativeFunction(GenerativeFunction):
    source: Callable

    def flatten(self):
        return (), (self.source,)

    def __call__(self, key, *args):
        return self.source(key, *args)

    def get_trace_type(self, key, args, **kwargs):
        assert isinstance(args, Tuple)
        jaxpr = jax.make_jaxpr(self.__call__)(key, *args)
        return get_trace_type(jaxpr)

    def simulate(self, key, args, **kwargs):
        assert isinstance(args, Tuple)
        key, (f, args, r, chm, score) = handler_simulate(
            self.source, **kwargs
        )(key, args)
        return key, BuiltinTrace(self, args, r, chm, score)

    @BooleanMask.collapse_boundary
    def importance(self, key, chm, args, **kwargs):
        assert isinstance(chm, ChoiceMap) or isinstance(chm, Trace)
        assert isinstance(args, Tuple)
        key, (w, (f, args, r, chm, score)) = handler_importance(
            self.source, **kwargs
        )(key, chm, args)
        return key, (w, BuiltinTrace(self, args, r, chm, score))

    @BooleanMask.collapse_boundary
    def update(self, key, prev, new, args, **kwargs):
        assert isinstance(new, ChoiceMap)
        assert isinstance(args, Tuple)
        key, (w, (f, args, r, chm, score), discard) = handler_update(
            self.source, **kwargs
        )(key, prev, new, args)
        return key, (
            w,
            BuiltinTrace(self, args, r, chm, score),
            BuiltinChoiceMap(discard),
        )

    def arg_grad(self, argnums, **kwargs):
        def _inner(key, tr, args):
            assert isinstance(tr, Trace)
            assert isinstance(args, Tuple)
            return handler_arg_grad(self.source, argnums, **kwargs)(
                key, tr, args
            )

        return _inner

    def choice_grad(self, key, tr, selected, **kwargs):
        assert isinstance(tr, Trace)
        assert isinstance(selected, Selection)
        selected, _ = selected.filter(tr)
        selected = selected.strip_metadata()
        grad_fn = handler_choice_grad(self.source, **kwargs)
        grad, key = jax.grad(
            grad_fn,
            argnums=2,
            has_aux=True,
        )(key, tr, selected)
        return key, grad
