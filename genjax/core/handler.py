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

from jax import core
from typing import (
    Any,
    Callable,
    Dict,
    Sequence,
)


class Handler:
    """
    A handler dispatchs a `jax.core.Primitive` - and must provide
    a `Callable` with signature `def (name_of_primitive)(continuation, *args)`
    where `*args` must match the `core.Primitive` declaration signature.
    """

    handles: Sequence[core.Primitive]
    callable: Callable

    def __init__(self, handles: Sequence[core.Primitive], callable: Callable):
        self.handles = handles
        self.callable = callable
