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

"""
This module holds an abstract class which forms the core functionality
for "tree-like" classes (like :code:`ChoiceMap` and :code:`Selection`).
"""

import abc
from dataclasses import dataclass
from genjax.core.pytree import Pytree
import jax._src.pretty_printer as pp
import genjax.core.pretty_printer as gpp


@dataclass
class ChoiceTree(Pytree):
    def overload_pprint(self, **kwargs):
        entries = []
        indent = kwargs["indent"]
        for (k, v) in self.get_subtrees_shallow():
            entry = gpp._dict_entry(k, v, **kwargs)
            entries.append(entry)
        return pp.concat(
            [
                pp.text(f"{type(self).__name__}"),
                gpp._nest(indent, pp.join(gpp._comma_sep, entries)),
            ]
        )

    @abc.abstractmethod
    def get_subtrees_shallow(self):
        pass

    @abc.abstractmethod
    def is_leaf(self):
        pass

    @abc.abstractmethod
    def get_leaf_value(self):
        pass

    @abc.abstractmethod
    def has_subtree(self, addr):
        pass

    @abc.abstractmethod
    def get_subtree(self, addr):
        pass

    @abc.abstractmethod
    def merge(self, other):
        pass
