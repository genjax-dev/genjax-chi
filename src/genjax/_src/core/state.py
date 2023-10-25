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
"""This module contains a `Module` class which supports parameter learning by
exposing primitives which allow users to sow functions with state."""

import dataclasses
import functools

import jax.core as jc
import jax.tree_util as jtu

from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import Callable
from oryx import harvest


NAMESPACE = "state"

collect = functools.partial(harvest.reap, tag=NAMESPACE)
inject = functools.partial(harvest.plant, tag=NAMESPACE)

# "clobber" here means that parameters get shared across sites with
# the same name and tag.
def param(*args, name):
    f = functools.partial(
        harvest.sow,
        tag=NAMESPACE,
        mode="clobber",
        name=name,
    )
    return f(*args)


##############
# State trie #
##############


@dataclasses.dataclass
class StateTrie(harvest.ReapState):
    inner: Trie

    def flatten(self):
        return (self.inner,), ()

    @classmethod
    def new(cls):
        trie = Trie.new()
        return StateTrie(trie)

    def sow(self, values, tree, name, mode):
        if name in self.inner:
            values = jtu.tree_leaves(harvest.tree_unreap(self.inner[name]))
        else:
            avals = jtu.tree_unflatten(
                tree,
                [jc.raise_to_shaped(jc.get_aval(v)) for v in values],
            )
            self.inner[name] = harvest.Reap.new(
                jtu.tree_unflatten(tree, values),
                dict(aval=avals),
            )

        return values


@dataclasses.dataclass
class Module(Pytree):
    apply: Callable
    params: StateTrie

    def flatten(self):
        return (self.params,), (self.apply,)

    @classmethod
    def new(cls, params, apply):
        return Module(apply, params)

    def get_params(self):
        return self.params

    def __call__(self, *args, **kwargs):
        return self.apply(self.params, *args, **kwargs)

    @classmethod
    def init(cls, apply):
        _collect = functools.partial(
            harvest.reap,
            state=StateTrie.new(),
            tag=NAMESPACE,
        )

        def wrapped(*args):
            _, params = _collect(apply)(*args)
            params = harvest.tree_unreap(params)
            jax_partial = jtu.Partial(apply)
            return Module.new(params, inject(jax_partial))

        return wrapped


##############
# Shorthands #
##############

init = Module.init
