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

from pygtrie import StringTrie
from jax.tree_util import register_pytree_node
import jax.numpy as jnp
from typing import Tuple, Any
from genjax.core.pytree import Pytree
from dataclasses import dataclass


class ChoiceMap(Pytree):
    trie: StringTrie

    def __init__(self, constraints):
        self.trie = StringTrie(separator=".")
        if isinstance(constraints, dict):
            for (k, v) in constraints.items():
                full = ".".join(k)
                self.trie[full] = v
        else:
            for (k, v) in constraints:
                self.trie[k] = v

    def __setitem__(self, k, v):
        self.trie[k] = v

    def __getitem__(self, k):
        assert isinstance(k, tuple)
        full = ".".join(k)
        ch = self.trie[full]
        if isinstance(ch, tuple):
            return ch[0]
        else:
            return ch

    def get_value(self, k):
        ch = self.trie[k]
        if isinstance(ch, tuple):
            return ch[0]
        else:
            return ch

    def get_score(self, k):
        (_, s) = self.trie[k]
        return s

    def has_choice(self, k):
        return k in self.trie

    def clear(self):
        self.trie.clear()

    def flatten(self):
        return self.trie.values(), self.trie.keys()

    @classmethod
    def unflatten(cls, slices, values):
        return ChoiceMap(zip(slices, values))


@dataclass
class Trace:
    args: Tuple
    retval: Any
    choices: StringTrie
    score: jnp.float32

    def get_choices(self):
        return self.choices

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score


register_pytree_node(
    Trace,
    lambda trace: (
        (trace.args, trace.retval, trace.choices, trace.score),
        None,
    ),
    lambda _, args: Trace(*args),
)
