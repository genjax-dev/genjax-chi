# Copyright 2023 MIT Probabilistic Computing Project
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

from dataclasses import dataclass

from genjax._src.core.datatypes.generative import ChoiceValue
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.typing import PRNGKey, typecheck
from genjax._src.generative_functions.static.static_gen_fn import (
    StaticGenerativeFunction,
)
from genjax._src.gensp.sp_distribution import SPDistribution
from genjax._src.gensp.target import Target

###############
# Enumeration #
###############


@dataclass
class Enumeration(SPDistribution):
    enumeration_strategy: Trie

    def flatten(self):
        return (), (self.enumeration_strategy,)

    @typecheck
    def random_weighted(
        self,
        key: PRNGKey,
        target: Target,
    ):
        gen_fn = target.p

        # This strategy is specialized to work on StaticGenerativeFunction
        # instances only.
        assert isinstance(gen_fn, StaticGenerativeFunction)

    @typecheck
    def estimate_logpdf(
        self,
        key: PRNGKey,
        chm: ChoiceValue,
        target: Target,
    ):
        pass


###########################
# Enumeration interpreter #
###########################
