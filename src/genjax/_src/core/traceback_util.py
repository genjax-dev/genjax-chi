# Copyright 2024 The JAX Authors and The MIT Probabilistic Computing Project.
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

import jax._src.traceback_util as traceback_util

from genjax._src.core.typing import Any, Callable, TypeVar

_C = TypeVar("_C", bound=Callable[..., Any])


def gfi_boundary(c: _C) -> _C:
    return traceback_util.api_boundary(c)
