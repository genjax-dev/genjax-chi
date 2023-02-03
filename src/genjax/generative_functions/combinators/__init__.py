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
if-else (:code:`SwitchCombinator`), mapping across vectorial arguments (:code:`MapCombinator`), and dependent for-loop (:code:`UnfoldCombinator`), as well as exposing new interfaces - including training learnable parameters (:code:`LearnCombinator`).

.. attention::

    The implementations of these combinators are similar to those in :code:`Gen.jl`, but JAX imposes extra restrictions on their construction and usage.

    In contrast to :code:`Gen.jl`, :code:`UnfoldCombinator` must have the number of
    unfold steps specified ahead of time as a static constant. The length of the unfold chain cannot depend on a variable whose value is known
    only at runtime.

    Similarly, for :code:`MapCombinator` - the shape of the vectorial arguments
    which will be mapped over must be known at JAX tracing time.

    These restrictions are not due to the implementation, but are fundamental to JAX's programming model (as it stands currently).
"""

from genjax._src.generative_functions.combinators import Learn
from genjax._src.generative_functions.combinators import LearnCombinator
from genjax._src.generative_functions.combinators import Map
from genjax._src.generative_functions.combinators import MapCombinator
from genjax._src.generative_functions.combinators import Switch
from genjax._src.generative_functions.combinators import SwitchCombinator
from genjax._src.generative_functions.combinators import Unfold
from genjax._src.generative_functions.combinators import UnfoldCombinator
from genjax._src.generative_functions.combinators import vec_chm
from genjax._src.generative_functions.combinators import vec_sel
from genjax._src.generative_functions.combinators import vector_choice_map
from genjax._src.generative_functions.combinators import vector_select


__all__ = [
    "MapCombinator",
    "Map",
    "UnfoldCombinator",
    "Unfold",
    "SwitchCombinator",
    "Switch",
    "LearnCombinator",
    "Learn",
    "vector_choice_map",
    "vec_chm",
    "vector_select",
    "vec_sel",
]
