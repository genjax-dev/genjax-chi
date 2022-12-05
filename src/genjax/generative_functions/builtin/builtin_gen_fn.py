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

from genjax.core.datatypes import ChoiceMap
from genjax.core.datatypes import GenerativeFunction
from genjax.core.datatypes import Trace
from genjax.core.staging import stage
from genjax.generative_functions.builtin.builtin_datatypes import BuiltinTrace
from genjax.generative_functions.builtin.builtin_tracetype import (
    get_trace_type,
)
from genjax.generative_functions.builtin.propagating import assess_transform
from genjax.generative_functions.builtin.propagating import (
    importance_transform,
)
from genjax.generative_functions.builtin.propagating import simulate_transform
from genjax.generative_functions.builtin.propagating import update_transform


@dataclass
class BuiltinGenerativeFunction(GenerativeFunction):
    source: Callable

    def flatten(self):
        return (), (self.source,)

    def get_trace_type(self, key, args, **kwargs):
        assert isinstance(args, Tuple)
        closed_jaxpr, _ = stage(self.__call__)(key, *args)
        return get_trace_type(closed_jaxpr)

    def simulate(self, key, args, **kwargs):
        assert isinstance(args, Tuple)
        key, (f, args, r, chm, score), cache = simulate_transform(
            self.source, **kwargs
        )(key, args)
        return key, BuiltinTrace(self, args, r, chm, cache, score)

    def importance(self, key, chm, args, **kwargs):
        assert isinstance(chm, ChoiceMap) or isinstance(chm, Trace)
        assert isinstance(args, Tuple)
        key, (w, (f, args, r, chm, score)), cache = importance_transform(
            self.source, **kwargs
        )(key, chm, args)
        return key, (w, BuiltinTrace(self, args, r, chm, cache, score))

    def assess(self, key, chm, args, **kwargs):
        assert isinstance(chm, ChoiceMap)
        key, (retval, score) = assess_transform(self.source, **kwargs)(
            key, chm, args
        )
        return key, (retval, score)

    def update(self, key, prev, new, args, **kwargs):
        assert isinstance(new, ChoiceMap)
        assert isinstance(args, Tuple)
        (
            key,
            (
                retval_diffs,
                w,
                (f, args, r, chm, score),
                discard,
            ),
            cache,
        ) = update_transform(self.source, **kwargs)(key, prev, new, args)
        return key, (
            retval_diffs,
            w,
            BuiltinTrace(self, args, r, chm, cache, score),
            discard,
        )
