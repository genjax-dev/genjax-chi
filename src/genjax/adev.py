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

from genjax._src.adev.core import ADEVPrimitive  # noqa: I001
from genjax._src.adev.core import E
from genjax._src.adev.core import HigherOrderADEVPrimitive
from genjax._src.adev.core import adev
from genjax._src.adev.core import reap_key
from genjax._src.adev.core import sample
from genjax._src.adev.core import sample_with_key
from genjax._src.adev.core import sow_key
from genjax._src.adev.primitives import add_cost
from genjax._src.adev.primitives import average
from genjax._src.adev.primitives import baseline
from genjax._src.adev.primitives import categorical_enum_parallel
from genjax._src.adev.primitives import flip_enum
from genjax._src.adev.primitives import flip_enum_parallel
from genjax._src.adev.primitives import flip_mvd
from genjax._src.adev.primitives import flip_reinforce
from genjax._src.adev.primitives import geometric_reinforce
from genjax._src.adev.primitives import maps
from genjax._src.adev.primitives import mv_normal_diag_reparam
from genjax._src.adev.primitives import mv_normal_reparam
from genjax._src.adev.primitives import normal_reinforce
from genjax._src.adev.primitives import normal_reparam
from genjax._src.adev.primitives import reinforce
from genjax._src.adev.primitives import uniform


__all__ = [
    # Language.
    "adev",
    "sample",
    "sample_with_key",
    "reap_key",
    "sow_key",
    "E",
    "ADEVPrimitive",
    "HigherOrderADEVPrimitive",
    # Primitives.
    "flip_enum",
    "flip_enum_parallel",
    "flip_mvd",
    "flip_reinforce",
    "categorical_enum_parallel",
    "geometric_reinforce",
    "normal_reinforce",
    "normal_reparam",
    "mv_normal_reparam",
    "mv_normal_diag_reparam",
    "uniform",
    "baseline",
    "reinforce",
    "average",
    "add_cost",
    # Higher order primitives.
    "maps",
]
