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
"""This module holds a set of generative function implementations called
generative function combinators.

These combinators accept generative functions as arguments, and return
generative functions with modified choice map shapes and behavior.

They are used to express common patterns of computation, including
if-else (`SwitchCombinator`), mapping across vectorial arguments (`MapCombinator`), and dependent for-loops (`UnfoldCombinator`).

.. attention::

    The implementations of these combinators are similar to those in `Gen.jl`, but JAX imposes extra restrictions on their construction and usage.

    In contrast to `Gen.jl`, `UnfoldCombinator` must have the number of
    unfold steps specified ahead of time as a static constant. The length of the unfold chain cannot depend on a variable whose value is known
    only at runtime.

    Similarly, for `MapCombinator` - the shape of the vectorial arguments
    which will be mapped over must be known at JAX tracing time.

    These restrictions are not due to the implementation, but are fundamental to JAX's programming model (as it stands currently).
"""

from genjax._src.generative_functions.combinators.switch.switch_combinator import Switch
from genjax._src.generative_functions.combinators.switch.switch_combinator import (
    SwitchCombinator,
)
from genjax._src.generative_functions.combinators.vector.map_combinator import Map
from genjax._src.generative_functions.combinators.vector.map_combinator import (
    MapCombinator,
)
from genjax._src.generative_functions.combinators.vector.unfold_combinator import Unfold
from genjax._src.generative_functions.combinators.vector.unfold_combinator import (
    UnfoldCombinator,
)
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    IndexChoiceMap,
)
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    IndexSelection,
)
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    VectorChoiceMap,
)
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    VectorSelection,
)
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    index_choice_map,
)
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    index_select,
)
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    vector_choice_map,
)
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    vector_select,
)
from genjax._src.generative_functions.combinators.vector.vector_utilities import (
    static_check_broadcast_dim_length,
)


__all__ = [
    "MapCombinator",
    "Map",
    "UnfoldCombinator",
    "Unfold",
    "SwitchCombinator",
    "Switch",
    "VectorChoiceMap",
    "vector_choice_map",
    "VectorSelection",
    "vector_select",
    "IndexChoiceMap",
    "index_choice_map",
    "IndexSelection",
    "index_select",
    "static_check_broadcast_dim_length",
]
