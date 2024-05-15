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
"""This module contains estimators of information-theoretic quantities implemented using
the generative function interface.

It supports inference diagnostic tools (like AIDE and SDOS) as well as
entropy estimation tools (like EEVI).

Some of these tools assume usage of `GenSP` to support estimating
density evaluations when the density in question is one produced by an
inference algorithm (see SDOS, for example).
"""

from genjax._src.information.aide import AuxiliaryInferenceDivergenceEstimator
from genjax._src.information.eevi import EntropyEstimatorsViaInference
from genjax._src.information.sdos import SymmetricDivergenceOverDatasets

__all__ = [
    "AuxiliaryInferenceDivergenceEstimator",
    "EntropyEstimatorsViaInference",
    "SymmetricDivergenceOverDatasets",
]
