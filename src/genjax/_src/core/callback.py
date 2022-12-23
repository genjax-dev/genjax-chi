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

"""This module implements a utility class which wraps the
`jax.experimental.host_callback.call` functionality, allowing a convenient way
to declare and use Python callbacks in device-intended code."""

from dataclasses import dataclass
from typing import Callable
from typing import Union

import jax
import jax.experimental.host_callback as hcb

from genjax._src.core.pytree import Pytree


@dataclass
class Callback(Pytree):
    callable: Callable
    get_result_shape: Union[None, Callable]

    def flatten(self):
        return (), (self.callable, self.get_result_shape)

    def __call__(self, *args, **kwargs):
        if self.get_result_shape is None:
            jaxpr = jax.make_jaxpr(self.callable)(*args)
            result_shape = jaxpr.out_avals
        else:
            result_shape = self.get_result_shape(*args)
        call_with_device = kwargs.get("call_with_device", False)
        return hcb.call(
            callable,
            *args,
            result_shape=result_shape,
            call_with_device=call_with_device,
        )
