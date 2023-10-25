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

import abc
from dataclasses import dataclass

from genjax._src.core.datatypes.generative import ValueChoiceMap
from genjax._src.core.typing import FloatArray
from genjax._src.core.typing import PRNGKey
from genjax._src.core.typing import Tuple
from genjax._src.generative_functions.distributions.distribution import Distribution


@dataclass
class SPDistribution(Distribution):
    @abc.abstractmethod
    def random_weighted(key: PRNGKey, *args) -> Tuple[FloatArray, ValueChoiceMap]:
        pass

    @abc.abstractmethod
    def estimate_logpdf(key: PRNGKey, v, *args) -> FloatArray:
        pass
