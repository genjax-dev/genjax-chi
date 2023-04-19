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

from genjax._src.core.pytree import Pytree
from genjax._src.core.pytree import PytreeClosure
from genjax._src.core.pytree import closure_convert
from genjax._src.core.transforms import harvest
from genjax._src.core.typing import Any
from genjax._src.core.typing import Dict
from genjax._src.core.typing import String


NAMESPACE = "state"

collect = functools.partial(harvest.reap, tag=NAMESPACE)
inject = functools.partial(harvest.plant, tag=NAMESPACE)

# "clobber" here means that parameters get shared across sites with
# the same name and namespace.
param = functools.partial(harvest.sow, tag=NAMESPACE, mode="clobber")


@dataclasses.dataclass
class Module(Pytree):
    params: Dict[String, Any]
    apply: PytreeClosure

    def flatten(self):
        return (self.params, self.apply), ()

    def get_params(self):
        return self.params

    def __call__(self, *args, **kwargs):
        return self.apply(self.params, *args, **kwargs)

    @classmethod
    def init(cls, apply):
        def wrapped(*args):
            _, params = collect(apply)(*args)
            params = harvest.unreap(params)
            pytree_closure = closure_convert(apply)
            return Module(params, inject(pytree_closure))

        return wrapped


##############
# Shorthands #
##############

init = Module.init
