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


from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import Any, Callable, Dict, List, PRNGKey, Protocol, Tuple


# This class is used to allow syntactic sugar (e.g. the `@` operator)
# in languages which support callees for generative functions via a `trace` intrinsic.
class SugaredGenerativeFunctionCall(Pytree):
    gen_fn: Callable
    args: Tuple
    kwargs: Dict = Pytree.static()

    def __matmul__(self, addr):
        return handle_off_trace_stack(addr, self.gen_fn, self.args)


# NOTE: Setup a global handler stack for the `trace` sugar.
# C.f. above.
# This stack will not interact with JAX tracers at all
# so it's safe, and will be resolved at JAX tracing time.
GLOBAL_TRACE_HANDLER_STACK: List[Callable] = []


def handle_off_trace_stack(addr, gen_fn: Callable, args):
    handler = GLOBAL_TRACE_HANDLER_STACK[-1]
    return handler(addr, gen_fn, args)


def push_trace_overload_stack(handler, fn):
    def wrapped(*args):
        GLOBAL_TRACE_HANDLER_STACK.append(handler)
        ret = fn(*args)
        GLOBAL_TRACE_HANDLER_STACK.pop()
        return ret

    return wrapped


class CanSimulate(Protocol):
    def simulate(self, key: PRNGKey, args: Tuple) -> Any:
        ...

    def __call__(self, *args, **kwargs) -> Any:
        ...


# This mixin overloads the call functionality for this generative function
# and allows usage of shorthand notation in the static DSL.
class SupportsCalleeSugar:
    def __call__(
        self: CanSimulate, *args: Any, **kwargs
    ) -> SugaredGenerativeFunctionCall:
        return SugaredGenerativeFunctionCall(self, args, kwargs)

    def apply(self: CanSimulate, key: PRNGKey, args: Tuple) -> Any:
        return self.simulate(key, args).get_retval()
