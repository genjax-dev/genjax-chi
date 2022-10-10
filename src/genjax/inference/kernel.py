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

from rich.tree import Tree

from genjax.core.pytree import Pytree


@dataclass
class MCMCKernel(Pytree):
    kernel: Callable
    reversal: None = None

    def __call__(self, *args, **kwargs):
        return self.kernel(*args, **kwargs)

    def flatten(self):
        return (), (self.kernel, self.reversal)

    def get_reversal(self):
        return self.reversal

    def set_reversal(self, reversal):
        self.reversal = reversal

    def tree_console_overload(self):
        tree = Tree("[blue][b]MCMCKernel[/b]")
        tree.add(str(self.kernel))
        return tree


def pkern(**kwargs):
    return lambda fn: MCMCKernel(fn, **kwargs)
