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

from genjax._src.generative_functions.combinators.masked_combinator import Masked
from genjax._src.generative_functions.combinators.masked_combinator import (
    masked_combinator,
)
from genjax._src.generative_functions.combinators.switch.switch_combinator import Switch
from genjax._src.generative_functions.combinators.switch.switch_combinator import (
    switch_combinator,
)
from genjax._src.generative_functions.combinators.vector.map_combinator import Map
from genjax._src.generative_functions.combinators.vector.map_combinator import (
    map_combinator,
)
from genjax._src.generative_functions.combinators.vector.repeat_combinator import Repeat
from genjax._src.generative_functions.combinators.vector.repeat_combinator import (
    repeat_combinator,
)
from genjax._src.generative_functions.combinators.vector.unfold_combinator import Unfold
from genjax._src.generative_functions.combinators.vector.unfold_combinator import (
    unfold_combinator,
)
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    indexed_choice_map,
)
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    indexed_select,
)
from genjax._src.generative_functions.combinators.vector.vector_datatypes import (
    vector_choice_map,
)


__all__ = [
    "Masked",
    "masked_combinator",
    "Map",
    "map_combinator",
    "Repeat",
    "repeat_combinator",
    "Unfold",
    "unfold_combinator",
    "Switch",
    "switch_combinator",
    "vector_choice_map",
    "indexed_select",
    "indexed_choice_map",
]
