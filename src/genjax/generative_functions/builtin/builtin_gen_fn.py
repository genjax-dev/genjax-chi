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
from genjax.generative_functions.builtin.propagating import (
    choice_grad_transform,
)
from genjax.generative_functions.builtin.propagating import (
    importance_transform,
)
from genjax.generative_functions.builtin.propagating import (
    retval_grad_transform,
)
from genjax.generative_functions.builtin.propagating import simulate_transform
from genjax.generative_functions.builtin.propagating import update_transform


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
        key, (f, args, r, chm, score) = simulate_transform(
            self.source, **kwargs
        )(key, args)
        return key, BuiltinTrace(self, args, r, chm, score)

    def importance(self, key, chm, args, **kwargs):
        chm = BooleanMask.collapse(chm)
        assert isinstance(chm, ChoiceMap) or isinstance(chm, Trace)
        assert isinstance(args, Tuple)
        key, (w, (f, args, r, chm, score)) = importance_transform(
            self.source, **kwargs
        )(key, chm, args)
        return key, (w, BuiltinTrace(self, args, r, chm, score))

    def update(self, key, prev, new, args, **kwargs):
        prev = BooleanMask.collapse(prev)
        new = BooleanMask.collapse(new)
        assert isinstance(new, ChoiceMap)
        assert isinstance(args, Tuple)
        (
            key,
            (
                w,
                retval_diff,
                (f, args, r, chm, score),
                discard,
            ),
        ) = update_transform(self.source, **kwargs)(key, prev, new, args)
        return key, (
            retval_diff,
            w,
            BuiltinTrace(self, args, r, chm, score),
            BuiltinChoiceMap(discard),
        )

    def choice_vjp(self, key, tr, selection, **kwargs):
        assert isinstance(tr, Trace)
        assert isinstance(selection, Selection)
        args = tr.get_args()
        fn = choice_grad_transform(self.source, key, selection, **kwargs)
        _, f_vjp, key = jax.vjp(fn, tr, args, has_aux=True)
        return key, lambda retval_grad: f_vjp((1.0, retval_grad))

    def retval_vjp(self, key, tr, selection, **kwargs):
        assert isinstance(tr, Trace)
        assert isinstance(selection, Selection)
        args = tr.get_args()
        fn = retval_grad_transform(self.source, key, selection, **kwargs)
        _, f_vjp, key = jax.vjp(fn, tr, args, has_aux=True)
        return key, lambda retval_grad: f_vjp(retval_grad)
