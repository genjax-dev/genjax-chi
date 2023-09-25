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

import jax

from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.builtin.builtin_gen_fn import (
    BuiltinGenerativeFunction,
)


@dataclass
class TraceKernel(Pytree):
    gen_fn: BuiltinGenerativeFunction

    def flatten(self):
        return (self.gen_fn,), ()

    def assess(self, key, choices, args):
        (r, w) = self.gen_fn.assess(key, choices, args)
        (chm, aux) = r
        return ((chm, aux), w)

    def propose(self, key, args):
        tr = self.gen_fn.simulate(key, args)
        choices = tr.get_choices()
        score = tr.get_score()
        retval = tr.get_retval()
        (chm, aux) = retval
        return (choices, score, (chm, aux))

    def jacfwd_retval(self, key, choices, args):
        def _inner(choices, args):
            ((chm, aux), w) = self.gen_fn.assess(key, choices, args)
            return (chm, aux)

        inner_jacfwd = jax.jacfwd(_inner, argnums=(0, 1))
        J = inner_jacfwd(choices, args)
        return J


#####
# Language decorator
#####


@typecheck
def trace_kernel(gen_fn: BuiltinGenerativeFunction):
    return TraceKernel.new(gen_fn)
