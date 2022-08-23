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

from dataclasses import dataclass
from typing import Any
from genjax.core.datatypes import ChoiceMap


#####
# ValueChoiceMap
#####


@dataclass
class ValueChoiceMap(ChoiceMap):
    value: Any

    # Implement the `Pytree` interface methods.
    def flatten(self):
        return (self.value,), ()

    @classmethod
    def unflatten(cls, data, xs):
        return ValueChoiceMap(*xs)

    def get_submaps_shallow(self):
        return ()

    def get_values_shallow(self):
        return (self.value,)

    def get_value(self):
        return self.value
