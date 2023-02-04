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

import dataclasses

from genjax._src.core.pytree import Pytree
from genjax._src.core.staging import get_shaped_aval
from genjax._src.core.typing import Callable
from genjax._src.core.typing import List


@dataclasses.dataclass
class PytreeClosure(Pytree):
    callable: Callable
    traced: List

    def flatten(self):
        return (self.traced,), (self.callable,)

    def __call__(self, *args):
        for (cell, v) in zip(self.callable.__closure__, self.traced):
            cell.cell_contents = v
        ret = self.callable(*args)
        for cell in self.callable.__closure__:
            cell.cell_contents = None
        return ret

    def __hash__(self):
        avals = list(map(get_shaped_aval, self.traced))
        return hash((self.callable, *avals))


def closure_convert(callable):
    captured = []
    for cell in callable.__closure__:
        captured.append(cell.cell_contents)
        cell.cell_contents = None
    return PytreeClosure(callable, captured)
