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

from jax import vmap
from jax.random import split

from genjax._src.core.generative import (
    Arguments,
    GenerativeFunction,
    Retval,
    Sample,
    Trace,
    UpdateRequest,
    Weight,
)
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import (
    Int,
    PRNGKey,
    Tuple,
)


@Pytree.dataclass
class SIR(GenerativeFunction):
    G: GenerativeFunction
    N: Int = Pytree.static()

    def simulate(
        self,
        key: PRNGKey,
        args: Arguments,
    ) -> Trace:
        return self.G.simulate(key, args)

    def assess(
        self,
        key: PRNGKey,
        z: Sample,
        args: Arguments,
    ) -> Tuple[Weight, Retval]:
        return self.G.assess(key, z, args)

    def update(
        self,
        key: PRNGKey,
        trace: Trace,
        request: UpdateRequest,
    ):
        sub_keys = split(key, self.N)
        tr, w, retdiff, bwd_req = vmap(self.G.update, in_axes=(0, None, None))(
            sub_keys, trace, request
        )


@Pytree.dataclass
class Marginal(GenerativeFunction):
    pass
