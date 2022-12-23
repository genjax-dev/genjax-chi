# Copyright 2022 MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains a utility class for defining new :code:`jax.Pytree`
implementors.

In addition to this functionality, there's a "sum type" :code:`Pytree`
implementation which allows effective decomposition of multiple potential
:code:`Pytree` value inhabitants into a common tree shape.

This allows, among other things, an efficient implementation of :code:`SwitchCombinator`.
"""

import abc
from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

import genjax._src.core.pretty_printing as gpp
from genjax._src.core.hashabledict import HashableDict
from genjax._src.core.hashabledict import hashabledict
from genjax._src.core.specialization import is_concrete


class Pytree(metaclass=abc.ABCMeta):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        jtu.register_pytree_node(
            cls,
            cls.flatten,
            cls.unflatten,
        )

    @abc.abstractmethod
    def flatten(self):
        pass

    @classmethod
    def unflatten(cls, data, xs):
        return cls(*data, *xs)

    @classmethod
    def new(cls, *args):
        return cls(*args)

    def __rich_console__(self, console, options):
        tree = gpp.tree_pformat(self)
        yield tree


#####
# Pytree sum type
#####

# If you have multiple Pytrees, you might want
# to generate a "sum" Pytree with leaves that minimally cover
# the entire set of dtypes and shapes.
#
# The code below is intended to provide this functionality.


def get_call_fallback(d, k, fn, fallback):
    if k in d:
        d[k] = fn(d[k])
    else:
        d[k] = fallback


def minimum_covering_leaves(pytrees: Sequence):
    leaf_schema = hashabledict()
    for tree in pytrees:
        local = hashabledict()
        jtu.tree_map(
            lambda v: get_call_fallback(local, v, lambda v: v + 1, 1),
            tree,
        )
        for (k, v) in local.items():
            get_call_fallback(leaf_schema, k, lambda u: v if v > u else u, v)

    return leaf_schema


def zero_payload(leaf_schema):
    payload = hashabledict()
    for (k, v) in leaf_schema.items():
        dtype = k.dtype
        shape = k.shape
        payload[k] = [jnp.zeros(shape, dtype) for _ in range(0, v)]
    return payload


def shape_dtype_struct(x):
    return jax.ShapeDtypeStruct(x.shape, x.dtype)


def set_payload(leaf_schema, pytree):
    leaves = jtu.tree_leaves(pytree)
    payload = hashabledict()
    for k in leaves:
        aval = shape_dtype_struct(jax.core.get_aval(k))
        if aval in payload:
            shared = payload[aval]
        else:
            shared = []
            payload[aval] = shared
        shared.append(k)

    for (k, limit) in leaf_schema.items():
        dtype = k.dtype
        shape = k.shape
        if k in payload:
            v = payload[k]
            cur_len = len(v)
            v.extend(
                [jnp.zeros(shape, dtype) for _ in range(0, limit - cur_len)]
            )
        else:
            payload[k] = [jnp.zeros(shape, dtype) for _ in range(0, limit)]
    return payload


def get_visitation(pytree):
    return jtu.tree_flatten(pytree)


def build_from_payload(visitation, form, payload):
    counter = hashabledict()

    def _check_counter_get(k):
        index = counter.get(k, 0)
        counter[k] = index + 1
        return payload[k][index]

    payload_copy = [_check_counter_get(k) for k in visitation]
    return jtu.tree_unflatten(form, payload_copy)


@dataclass
class StaticCollection(Pytree):
    seq: Sequence

    def flatten(self):
        return (), (self.seq,)


@dataclass
class SumPytree(Pytree):
    visitations: StaticCollection
    forms: StaticCollection
    payload: HashableDict

    def flatten(self):
        return (self.payload,), (self.visitations, self.forms)

    @classmethod
    def new(cls, source: Pytree, covers: Sequence[Pytree]):
        leaf_schema = minimum_covering_leaves(covers)
        visitations = []
        forms = []
        for cover in covers:
            visitation, form = get_visitation(cover)
            visitations.append(visitation)
            forms.append(form)
        visitations = StaticCollection(visitations)
        forms = StaticCollection(forms)
        payload = set_payload(leaf_schema, source)
        return SumPytree(visitations, forms, payload)

    def materialize_iterator(self):
        static_visitations = self.visitations.seq
        static_forms = self.forms.seq
        return map(
            lambda args: build_from_payload(args[0], args[1], self.payload),
            zip(static_visitations, static_forms),
        )


#####
# Utilities
#####


def tree_stack(trees):
    """Takes a list of trees and stacks every corresponding leaf.

    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).

    Useful for turning a list of objects into something you can feed to a
    vmapped function.

    This function respects concrete vs. traced values. It will leave concrete
    leaves unchanged (it will not lift them to :code:`jax.core.Tracer`).
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jtu.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [
        np.stack(leaf) if all(map(is_concrete, leaf)) else jnp.stack(leaf)
        for leaf in grouped_leaves
    ]
    return treedef_list[0].unflatten(result_leaves)


def tree_unstack(tree):
    """Takes a tree and turns it into a list of trees. Inverse of tree_stack.

    For example, given a tree ((a, b), c), where a, b, and c all have first
    dimension k, will make k trees
    [((a[0], b[0]), c[0]), ..., ((a[k], b[k]), c[k])]

    Useful for turning the output of a vmapped function into normal objects.
    """
    leaves, treedef = jtu.tree_flatten(tree)
    n_trees = leaves[0].shape[0]
    new_leaves = [[] for _ in range(n_trees)]
    for leaf in leaves:
        for i in range(n_trees):
            new_leaves[i].append(leaf[i])
    new_trees = [treedef.unflatten(leaf) for leaf in new_leaves]
    return new_trees


def squeeze(tree):
    def _inner(v):
        if isinstance(v, np.ndarray):
            return np.squeeze(v)
        elif isinstance(v, jnp.ndarray):
            return jnp.squeeze(v)
        else:
            return v

    return jtu.tree_map(_inner, tree)
