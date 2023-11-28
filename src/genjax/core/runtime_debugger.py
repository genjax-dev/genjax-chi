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

from genjax._src.core.datatypes.generative import JAXGenerativeFunction
from genjax._src.core.runtime_debugger import pull
from genjax._src.core.runtime_debugger import push
from genjax._src.core.runtime_debugger import record_call
from genjax._src.core.runtime_debugger import record_value
from genjax._src.core.runtime_debugger import tag
from genjax._src.core.typing import Any
from genjax._src.core.typing import dispatch
from genjax._src.generative_functions.static.static_gen_fn import SupportsStaticSugar


####################
# Debug combinator #
####################


@dataclass
class DebugCombinator(JAXGenerativeFunction, SupportsStaticSugar):
    gen_fn: JAXGenerativeFunction

    def flatten(self):
        return (self.gen_fn,), ()

    def simulate(self, *args):
        return record_call(self.gen_fn.simulate)(*args)

    def propose(self, *args):
        return record_call(self.gen_fn.propose)(*args)

    def importance(self, *args):
        return record_call(self.gen_fn.importance)(*args)

    def update(self, *args):
        return record_call(self.gen_fn.update)(*args)

    def assess(self, *args):
        return record_call(self.gen_fn.assess)(*args)


@dispatch
def record(gen_fn: JAXGenerativeFunction) -> DebugCombinator:
    """A multimethod which dispatches to `record_value`, or wraps a
    `JAXGenerativeFunction` in `DebugCombinator` to allow recording information
    about generative function interface invocations."""
    return DebugCombinator.new(gen_fn)


@dispatch
def record(v: Any) -> Any:
    """A multimethod which dispatches to `record_value`, or wraps a
    `JAXGenerativeFunction` in `DebugCombinator` to allow recording information
    about generative function interface invocations."""
    return record_value(v)


__all__ = [
    "tag",
    "pull",
    "push",
    "record_value",
    "record_call",
    "record",
]
