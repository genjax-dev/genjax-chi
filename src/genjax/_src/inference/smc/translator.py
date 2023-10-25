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
from genjax._src.core.pytree.utilities import tree_grad_split
from genjax._src.core.pytree.utilities import tree_zipper
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import typecheck
from genjax._src.generative_functions.static.static_gen_fn import (
    StaticGenerativeFunction,
)


@dataclass
class TraceKernel(Pytree):
    gen_fn: StaticGenerativeFunction

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

    def jacfwd_retval(self, key, choices, trace, proposal_args):
        grad_tree, no_grad_tree = tree_grad_split((choices, trace))

        def _inner(differentiable: Tuple):
            choices, trace = tree_zipper(differentiable, no_grad_tree)
            ((chm, aux), _) = self.gen_fn.assess(key, choices, (trace, proposal_args))
            return (chm, aux)

        inner_jacfwd = jax.jacfwd(_inner)
        J = inner_jacfwd(grad_tree)
        return J


#####
# Language decorator
#####


@typecheck
def trace_kernel(gen_fn: StaticGenerativeFunction):
    return TraceKernel.new(gen_fn)
