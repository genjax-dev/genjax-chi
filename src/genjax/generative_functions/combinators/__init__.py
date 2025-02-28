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

from genjax._src.generative_functions.combinators.dimap import (
    Dimap,
    contramap,
    dimap,
    map,
)
from genjax._src.generative_functions.combinators.mask import (
    MaskCombinator,
    mask,
)
from genjax._src.generative_functions.combinators.mixture import (
    mix,
)
from genjax._src.generative_functions.combinators.or_else import (
    or_else,
)
from genjax._src.generative_functions.combinators.repeat import (
    RepeatCombinator,
    repeat,
)
from genjax._src.generative_functions.combinators.scan import (
    IndexRequest,
    Scan,
    VectorRequest,
    accumulate,
    iterate,
    iterate_final,
    masked_iterate,
    masked_iterate_final,
    reduce,
    scan,
)
from genjax._src.generative_functions.combinators.switch import (
    Switch,
    switch,
)
from genjax._src.generative_functions.combinators.vmap import (
    Vmap,
    vmap,
)

__all__ = [
    "Dimap",
    "IndexRequest",
    "MaskCombinator",
    "RepeatCombinator",
    "Scan",
    "Switch",
    "VectorRequest",
    "Vmap",
    "accumulate",
    "contramap",
    "dimap",
    "iterate",
    "iterate_final",
    "map",
    "mask",
    "masked_iterate",
    "masked_iterate_final",
    "mix",
    "or_else",
    "reduce",
    "repeat",
    "scan",
    "switch",
    "vmap",
]
