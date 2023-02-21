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

"""This module provides a combinator which transforms a generative function
into a `nn.Module`-like object that holds learnable parameters.

It exposes an extended set of interfaces (new: `param_grad` and `update_state`) which allow programmatic computation of gradients with respect to held parameters, as well as updating parameters.

It enables learning idioms which cohere with other packages in the JAX ecosystem (e.g. supporting `optax` optimizers).
"""

import functools
from dataclasses import dataclass
from typing import Any

from genjax._src.core.datatypes import ChoiceMap
from genjax._src.core.datatypes import GenerativeFunction
from genjax._src.core.datatypes import Trace
from genjax._src.core.interpreters.harvest import harvest
from genjax._src.core.interpreters.harvest import plant
from genjax._src.core.interpreters.harvest import sow
from genjax._src.core.typing import Callable
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import Union
from genjax._src.core.typing import typecheck


TAG = "state"
collect = functools.partial(harvest, tag=TAG)
inject = functools.partial(plant, tag=TAG)
param = functools.partial(sow, tag=TAG)


@dataclass
class StateCombinator(GenerativeFunction):
    inner: Union[Callable, GenerativeFunction]
    params: Any

    def flatten(self):
        return (self.inner, self.state), ()

    @typecheck
    def simulate(self, key: PRNGKey, args: Tuple):
        return self.inner.simulate(key, (self.params, *args))

    @typecheck
    def importance(self, key: PRNGKey, chm: ChoiceMap, args: Tuple):
        return self.inner.importance(key, chm, (self.params, *args))

    @typecheck
    def update(self, key: PRNGKey, prev: Trace, chm: ChoiceMap, args: Tuple):
        return self.inner.update(key, prev, chm, (self.params, *args))

    @typecheck
    def assess(self, key: PRNGKey, chm: ChoiceMap, args: Tuple):
        return self.inner.assess(key, chm, (self.params, *args))


def init(f):
    def wrapped(*args):
        _, params = collect(f)({}, *args)
        return StateCombinator.new(inject(f), params)

    return wrapped


##############
# Shorthands #
##############

State = StateCombinator.new
