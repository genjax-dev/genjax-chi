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
This module provides a hashable dictionary class - allowing the
usage of :code:`dict`-like instances as JAX JIT cache keys
(and allowing their usage with JAX :code:`static_argnums` in :code:`jax.jit`).
"""

import jax.tree_util as jtu
from jax.util import safe_zip


class HashableDict(dict):
    def __key(self):
        return tuple((k, self[k]) for k in sorted(self))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return self.__key() == other.__key()


jtu.register_pytree_node(
    HashableDict,
    lambda x: (list(x.values()), list(x.keys())),
    lambda keys, values: HashableDict(safe_zip(keys, values)),
)


def hashabledict():
    return HashableDict({})
