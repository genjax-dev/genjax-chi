# Copyright 2023 MIT Probabilistic Computing Project
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

from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import Any, Dict, PRNGKey, Tuple, dispatch


# This class is used to allow syntactic sugar (e.g. the `@` operator)
# in languages which support callees for generative functions via a `trace` intrinsic.
@dataclass
class SugaredGenerativeFunctionCall(Pytree):
    gen_fn: GenerativeFunction
    kwargs: Dict
    args: Tuple

    def flatten(self):
        return (self.args,), (self.gen_fn, self.kwargs)

    def __matmul__(self, addr):
        return handle_off_trace_stack(addr, self.gen_fn, self.args)


# NOTE: Setup a global handler stack for the `trace` sugar.
# C.f. above.
# This stack will not interact with JAX tracers at all
# so it's safe, and will be resolved at JAX tracing time.
GLOBAL_TRACE_HANDLER_STACK = []


def handle_off_trace_stack(addr, gen_fn, args):
    handler = GLOBAL_TRACE_HANDLER_STACK[-1]
    return handler(addr, gen_fn, args)


def push_trace_overload_stack(handler, fn):
    def wrapped(*args):
        GLOBAL_TRACE_HANDLER_STACK.append(handler)
        ret = fn(*args)
        GLOBAL_TRACE_HANDLER_STACK.pop()
        return ret

    return wrapped


# This mixin overloads the call functionality for this generative function
# and allows usage of shorthand notation in the static DSL.
class SupportsCalleeSugar:
    @dispatch
    def __call__(self, *args: Any, **kwargs) -> SugaredGenerativeFunctionCall:
        return SugaredGenerativeFunctionCall(self, kwargs, args)

    @dispatch
    def __call__(self, key: PRNGKey, args: Tuple) -> Any:
        tr = self.simulate(key, args)
        retval = tr.get_retval()
        return retval
