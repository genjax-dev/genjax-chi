# Copyright 2024 MIT Probabilistic Computing Project
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


import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import Sequence


def get_call_fallback(d, k, fn, fallback):
    if k in d:
        d[k] = fn(d[k])
    else:
        d[k] = fallback


def minimum_covering_leaves(pytrees: Sequence):
    leaf_schema = dict()
    for tree in pytrees:
        local = dict()
        jtu.tree_map(
            lambda v: get_call_fallback(local, v, lambda v: v + 1, 1),
            tree,
        )
        for k, v in local.items():
            get_call_fallback(leaf_schema, k, lambda u: v if v > u else u, v)

    return leaf_schema


def shape_dtype_struct(x):
    return jax.ShapeDtypeStruct(x.shape, x.dtype)


def set_payload(leaf_schema, pytree):
    leaves = jtu.tree_leaves(pytree)
    payload = dict()
    for k in leaves:
        aval = shape_dtype_struct(jax.core.get_aval(k))
        if aval in payload:
            shared = payload[aval]
        else:
            shared = []
            payload[aval] = shared
        shared.append(k)

    for k, limit in leaf_schema.items():
        dtype = k.dtype
        shape = k.shape
        if k in payload:
            v = payload[k]
            cur_len = len(v)
            v.extend([jnp.zeros(shape, dtype) for _ in range(0, limit - cur_len)])
        else:
            payload[k] = [jnp.zeros(shape, dtype) for _ in range(0, limit)]
    return payload


def build_from_payload(visitation, form, payload):
    counter = dict()

    def _check_counter_get(k):
        index = counter.get(k, 0)
        counter[k] = index + 1
        return payload[k][index]

    payload_copy = [_check_counter_get(k) for k in visitation]
    return jtu.tree_unflatten(form, payload_copy)


class StaticCollection(Pytree):
    seq: Sequence = Pytree.static()


class DataSharedSumTree(Pytree):
    payload: dict
    visitations: StaticCollection = Pytree.static()
    forms: StaticCollection = Pytree.static()

    @classmethod
    def new(cls, source: Pytree, covers: Sequence[Pytree]):
        leaf_schema = minimum_covering_leaves(covers)
        visitations = []
        forms = []
        for cover in covers:
            visitation, form = jtu.tree_flatten(cover)
            visitations.append(visitation)
            forms.append(form)
        visitations = StaticCollection(visitations)
        forms = StaticCollection(forms)
        payload = set_payload(leaf_schema, source)
        return DataSharedSumTree(payload, visitations, forms)

    def materialize_iterator(self):
        static_visitations = self.visitations.seq
        static_forms = self.forms.seq
        return map(
            lambda args: build_from_payload(args[0], args[1], self.payload),
            zip(static_visitations, static_forms),
        )
