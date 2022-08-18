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

import jax
import jax.numpy as jnp
from pygtrie import StringTrie
from typing import Tuple, Any
from genjax.core.pytree import Pytree
from dataclasses import dataclass
import genjax.pretty_print as pp


def stringtrie_flatten(self):
    return self.trie.values(), self.trie.keys()


def stringtrie_unflatten(cls, slices, values):
    return StringTrie(zip(slices, values))


# Global registration for `StringTrie`
jax.tree_util.register_pytree_node(
    StringTrie,
    stringtrie_flatten,
    stringtrie_unflatten,
)


#####
# ChoiceMap
#####


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
        if isinstance(k, tuple):
            full = ".".join(k)
            ch = self.trie[full]
        else:
            ch = self.trie[k]
        if isinstance(ch, tuple):
            return ch[0]
        else:
            return ch

    def __str__(self):
        return pp.tree_pformat(self)

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

    def setdiff(self, other):
        discard = ChoiceMap([])
        for (k, v) in self.trie.items():
            if other.has_choice(k):
                discard[k] = v
        return discard

    def clear(self):
        self.trie.clear()

    def flatten(self):
        return self.trie.values(), self.trie.keys()

    @classmethod
    def unflatten(cls, slices, values):
        return ChoiceMap(zip(slices, values))


#####
# Trace
#####


@dataclass
class Trace(Pytree):
    args: Tuple
    retval: Any
    choices: StringTrie
    score: jnp.float32

    def __str__(self):
        return pp.tree_pformat(self)

    def get_choices(self):
        return self.choices

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def get_args(self):
        return self.args

    def get_slice(self, k: int):
        pass

    def __getitem__(self, k):
        return self.choices[k]

    def flatten(self):
        return (self.args, self.retval, self.choices, self.score), None

    @classmethod
    def unflatten(cls, slices, values):
        return Trace(*values)
