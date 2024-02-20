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

from deprecated import deprecated  # noqa: I001
from genjax._src.generative_functions.static.static_gen_fn import (
    static_gen_fn,
    StaticGenerativeFunction,
)
from genjax._src.generative_functions.static.static_transforms import (
    cache,
    save,
    trace,
    trace_p,
)


@deprecated(version="0.2.0", reason="now called @static_gen_fn")
def static(f) -> StaticGenerativeFunction:
    return static_gen_fn(f)


__all__ = [
    "trace_p",
    "trace",
    "cache",
    "save",
    "static",
    "static_gen_fn",
]
