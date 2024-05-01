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

from genjax._src.core.generative import (
    ChoiceMap,
    GenerativeFunction,
    Trace,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Any,
    FloatArray,
    List,
    Tuple,
)
from genjax._src.generative_functions.static.static_transforms import AddressVisitor

#########
# Trace #
#########


@Pytree.dataclass
class StaticTrace(Trace):
    gen_fn: GenerativeFunction
    args: Tuple
    retval: Any
    addresses: AddressVisitor
    subtraces: List[Trace]
    score: FloatArray

    def get_gen_fn(self):
        return self.gen_fn

    def get_sample(self):
        addresses = self.addresses.get_visited()
        addresses = Pytree.tree_unwrap_const(addresses)
        chm = ChoiceMap.n
        for addr, subtrace in zip(addresses, self.subtraces):
            chm = chm ^ ChoiceMap.a(addr, subtrace.get_sample())

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
