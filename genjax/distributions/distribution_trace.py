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

from genjax.core import Pytree
from dataclasses import dataclass
from typing import Tuple, Callable, Any
from .value_choice_map import ValueChoiceMap


@dataclass
class DistributionTrace(Pytree):
    gen_fn: Callable
    args: Tuple
    choices: ValueChoiceMap
    score: Any

    def get_gen_fn(self):
        return self.gen_fn

    def get_retval(self):
        return (self.choices.get_value(),)

    def get_args(self):
        return self.args

    def get_score(self):
        return self.score

    def get_choices(self):
        return self.choices

    def flatten(self):
        return (self.args, self.choices, self.score), (self.gen_fn,)

    def get_value(self):
        return self.choices.get_value()

    @classmethod
    def unflatten(cls, data, xs):
        return DistributionTrace(*data, *xs)
