# Copyright 2024 MIT Probabilistic Computing Project
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

import jax  # noqa: I001
import jax.numpy as jnp  # noqa: I001
from genjax._src.core.pytree import Pytree
from genjax._src.generative_functions.static.static_transforms import AddressVisitor
from genjax._src.core.generative import (
    GenerativeFunction,
    Selection,
    ChoiceMap,
    Trace,
)
from genjax._src.core.serialization.pickle import (
    PickleDataFormat,
    PickleSerializationBackend,
    SupportsPickleSerialization,
)
from genjax._src.core.typing import (
    Any,
    FloatArray,
    Tuple,
    dispatch,
    PRNGKey,
    List,
)

#########
# Trace #
#########


class StaticTrace(
    Trace,  # inherits from Pytree
    SupportsPickleSerialization,
):
    gen_fn: GenerativeFunction
    args: Tuple
    retval: Any
    addresses: AddressVisitor
    subtraces: List[Trace]
    score: FloatArray

    def get_gen_fn(self):
        return self.gen_fn

    def get_choices(self):
        addresses = self.addresses.get_visited()
        addresses = Pytree.tree_unwrap_const(addresses)
        chm = ChoiceMap.n
        for addr, subtrace in zip(addresses, self.subtraces):
            chm = chm ^ ChoiceMap.a(addr, subtrace.get_choices())
        return chm

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def get_args(self):
        return self.args

    def get_subtrace(self, addr):
        addresses = self.addresses.get_visited()
        addresses = Pytree.tree_unwrap_const(addresses)
        idx = addresses.index(addr)
        return self.subtraces[idx]

    def project(
        self,
        key: PRNGKey,
        selection: Selection,
    ) -> FloatArray:
        weight = jnp.array(0.0)
        for k, subtrace in self.address_choices.get_submaps_shallow():
            key, sub_key = jax.random.split(key)
            remaining = selection.step(k)
            weight += subtrace.project(sub_key, remaining)
        return weight

    def get_aux(self):
        return (self.address_choices,)

    #################
    # Serialization #
    #################

    @dispatch
    def dumps(
        self,
        backend: PickleSerializationBackend,
    ) -> PickleDataFormat:
        args, retval, score = self.args, self.retval, self.score
        choices_payload = []
        addr_payload = []
        for addr, subtrace in self.address_choices.get_submaps_shallow():
            inner_payload = subtrace.dumps(backend)
            choices_payload.append(inner_payload)
            addr_payload.append(addr)
        payload = [
            backend.dumps(args),
            backend.dumps(retval),
            backend.dumps(score),
            backend.dumps(addr_payload),
            backend.dumps(choices_payload),
        ]
        return PickleDataFormat(payload)
