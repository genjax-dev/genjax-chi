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

"""This module contains a "sum type" :code:`Pytree`
implementation which allows effective decomposition of multiple potential
:code:`Pytree` value inhabitants into a common tree shape.

This allows, among other things, an efficient implementation of :code:`SwitchCombinator`.
"""

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from genjax._src.core.hashabledict import HashableDict
from genjax._src.core.hashabledict import hashabledict
from genjax._src.core.pytree import Pytree


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
class Sumtree(Pytree):
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
        return Sumtree(visitations, forms, payload)

    def materialize_iterator(self):
        static_visitations = self.visitations.seq
        static_forms = self.forms.seq
        return map(
            lambda args: build_from_payload(args[0], args[1], self.payload),
            zip(static_visitations, static_forms),
        )
