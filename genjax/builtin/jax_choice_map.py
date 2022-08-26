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

import jax.numpy as jnp
from dataclasses import dataclass
import genjax.core.pretty_printer as pp
from genjax.core.datatypes import ChoiceMap
from genjax.distributions.distribution import ValueChoiceMap

#####
# JAXChoiceMap
#####


@dataclass
class JAXChoiceMap(ChoiceMap):
    tree: dict

    def __init__(self, constraints):
        self.tree = {}
        if isinstance(constraints, dict):
            for (k, v) in constraints.items():
                assert isinstance(k, tuple)
                if len(k) == 1:
                    self.tree[k[-1]] = ValueChoiceMap(jnp.array(v))
                else:
                    if k[0] in self.tree:
                        inner = self.tree[k[0]]
                        inner.insert(k[1:], v)
                    else:
                        self.tree[k[0]] = JAXChoiceMap({k[1:]: v})
        else:
            for (k, v) in constraints:
                self.tree[k] = v

    # Implement the `Pytree` interfaces.
    def flatten(self):
        return self.tree.values(), self.tree.keys()

    @classmethod
    def unflatten(cls, slices, values):
        return JAXChoiceMap(zip(slices, values))

    def __setitem__(self, k, v):
        self.tree[k] = v

    def __getitem__(self, k):
        if isinstance(k, tuple):
            assert len(k) > 0
            prefix = k[0]
            sub = self.tree[prefix]
            return sub.__getitem__(k[1:])
        return self.tree[k]

    def __str__(self):
        return pp.tree_pformat(self)

    def get_key(self, k):
        return self.tree[k]

    def has_key(self, k):
        return k in self.tree
