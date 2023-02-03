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

import jax

from genjax._src.core.datatypes import ChoiceMap
from genjax._src.core.datatypes import GenerativeFunction
from genjax._src.core.datatypes import Trace
from genjax._src.core.diff_rules import check_is_diff
from genjax._src.core.pytree import Pytree
from genjax._src.core.tracetypes import TraceType
from genjax._src.core.typing import Any
from genjax._src.core.typing import Callable
from genjax._src.core.typing import Dict
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import List
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import Union
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.builtin.builtin_datatypes import BuiltinChoiceMap
from genjax._src.generative_functions.builtin.builtin_datatypes import BuiltinTrace
from genjax._src.generative_functions.builtin.builtin_tracetype import (
    trace_type_transform,
)
from genjax._src.generative_functions.builtin.intrinsics import cache
from genjax._src.generative_functions.builtin.intrinsics import trace
from genjax._src.generative_functions.builtin.transforms import assess_transform
from genjax._src.generative_functions.builtin.transforms import importance_transform
from genjax._src.generative_functions.builtin.transforms import simulate_transform
from genjax._src.generative_functions.builtin.transforms import update_transform


# This class is used to allow syntactic sugar (e.g. the `@` operator)
# in the builtin language for functions via the `cache` intrinsics.
@dataclass
class DeferredFunctionCall(Pytree):
    fn: Callable
    kwargs: Dict
    args: Union[None, Tuple]

    def flatten(self):
        return (self.args,), (self.fn, self.kwargs)

    @classmethod
    def new(cls, fn, **kwargs):
        assert not isinstance(fn, GenerativeFunction)
        return DeferredFunctionCall(fn, kwargs, None)

    def __call__(self, *args):
        return DeferredFunctionCall(self.fn, self.kwargs, args)

    def __matmul__(self, addr):
        return cache(addr, self.fn, **self.kwargs)(*self.args)


def save(fn, **kwargs):
    return DeferredFunctionCall.new(fn, **kwargs)


# This class is used to allow syntactic sugar (e.g. the `@` operator)
# in the builtin language for generative functions via the `trace` intrinsic.
@dataclass
class DeferredGenerativeFunctionCall(Pytree):
    gen_fn: GenerativeFunction
    kwargs: Dict
    args: Tuple

    def flatten(self):
        return (self.args,), (self.gen_fn, self.kwargs)

    @classmethod
    def new(cls, gen_fn, args, kwargs):
        return DeferredGenerativeFunctionCall(gen_fn, kwargs, args)

    def __matmul__(self, addr):
        return trace(addr, self.gen_fn, **self.kwargs)(*self.args)


@dataclass
class BuiltinGenerativeFunction(GenerativeFunction):
    source: Callable
    aux_args: Union[None, List]

    def flatten(self):
        return (self.aux_args,), (self.source,)

    @classmethod
    def new(cls, source, aux_args=None):
        return BuiltinGenerativeFunction(source, aux_args)

    # This overloads the call functionality for this generative function
    # and allows usage of shorthand notation in the builtin DSL.
    def __call__(self, *args, **kwargs) -> DeferredGenerativeFunctionCall:
        return DeferredGenerativeFunctionCall.new(self, args, kwargs)

    @typecheck
    def get_trace_type(self, *args, **kwargs) -> TraceType:
        return trace_type_transform(self.source, **kwargs)(*args)

    @typecheck
    def simulate(
        self, key: PRNGKey, args: Tuple, **kwargs
    ) -> Tuple[PRNGKey, BuiltinTrace]:
        converted_fn, aux_args = jax.closure_convert(self.source, *args)
        key, (f, cc_args, r, chm, score), cache = simulate_transform(
            converted_fn, **kwargs
        )(key, (*args, *aux_args))
        cc_gen_fn = BuiltinGenerativeFunction.new(f, aux_args)
        return key, BuiltinTrace(cc_gen_fn, cc_args, r, chm, cache, score)

    @typecheck
    def importance(
        self, key: PRNGKey, chm: ChoiceMap, args: Tuple, **kwargs
    ) -> Tuple[PRNGKey, Tuple[FloatArray, BuiltinTrace]]:
        key, (w, (f, args, r, chm, score)), cache = importance_transform(
            self.source, **kwargs
        )(key, chm, args)
        return key, (w, BuiltinTrace(self, args, r, chm, cache, score))

    @typecheck
    def update(
        self,
        key: PRNGKey,
        prev: Trace,
        constraints: ChoiceMap,
        argdiffs: Tuple,
        **kwargs
    ) -> Tuple[PRNGKey, Tuple[Any, FloatArray, Trace, ChoiceMap]]:
        assert all(map(check_is_diff, argdiffs))
        (
            key,
            (
                retval_diffs,
                w,
                (f, args, r, chm, score),
                discard,
            ),
            cache,
        ) = update_transform(self.source, **kwargs)(key, prev, constraints, argdiffs)
        return key, (
            retval_diffs,
            w,
            BuiltinTrace(self, args, r, chm, cache, score),
            BuiltinChoiceMap(discard),
        )

    @typecheck
    def assess(
        self, key: PRNGKey, chm: ChoiceMap, args: Tuple, **kwargs
    ) -> Tuple[PRNGKey, Tuple[Any, FloatArray]]:
        key, (retval, score) = assess_transform(self.source, **kwargs)(key, chm, args)
        return key, (retval, score)
