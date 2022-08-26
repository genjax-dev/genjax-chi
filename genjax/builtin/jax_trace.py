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

import jax.numpy as jnp
from typing import Callable, Tuple, Any
from dataclasses import dataclass
import genjax.core.pretty_printer as pp
from genjax.core.datatypes import Trace

#####
# Trace
#####


@dataclass
class JAXTrace(Trace):
    gen_fn: Callable
    args: Tuple
    retval: Any
    choices: dict
    score: jnp.float32

    def __str__(self):
        return pp.tree_pformat(self)

    def get_gen_fn(self):
        return self.gen_fn

    def get_choices(self):
        return self.choices

    def get_retval(self):
        return self.retval

    def get_score(self):
        return self.score

    def get_args(self):
        return self.args

    def has_key(self, k):
        return self.choices.has_key(k)

    def get_key(self, k):
        return self.choices[k]

    def __getitem__(self, k):
        if isinstance(k, tuple):
            assert len(k) > 0
            prefix = k[0]
            sub = self.choices[prefix]
            return sub.__getitem__(k[1:])
        return self.choices[k]

    def flatten(self):
        return (self.args, self.retval, self.choices, self.score), (
            self.gen_fn,
        )

    @classmethod
    def unflatten(cls, data, xs):
        return JAXTrace(*data, *xs)
