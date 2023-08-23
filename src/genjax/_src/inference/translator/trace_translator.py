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

from genjax._src.core.datatypes.generative import ChoiceMap
from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import typecheck
from genjax._src.inference.translator.trace_kernel import TraceKernel


####################
# Trace translator #
####################


@dataclass
class TraceTranslator(Pytree):
    pass


######
# Deterministic trace translator
######


@dataclass
class DeterministicTraceTranslator(TraceTranslator):
    forward_kernel: TraceKernel
    backward_kernel: TraceKernel

    def flatten(self):
        return (self.forward_kernel, self.backward_kernel), ()

    def forward(self, prev_model_trace: Trace, forward_proposal_trace: Trace):
        pass

    def backward(self, prev_model_trace: Trace, prev_obs: ChoiceMap):
        pass

    @typecheck
    def apply(self, trace: Trace, obs: ChoiceMap, args: Tuple):
        pass

    def __call__(self, *args):
        return self.apply(*args)


######
# General trace translator
######


@dataclass
class GeneralTraceTranslator(TraceTranslator):
    forward_kernel: TraceKernel
    backward_kernel: TraceKernel

    def flatten(self):
        return (self.forward_kernel, self.backward_kernel), ()

    def forward(self, prev_model_trace: Trace, forward_proposal_trace: Trace):
        pass

    def backward(self, prev_model_trace: Trace, prev_obs: ChoiceMap):
        pass

    @typecheck
    def apply(self, trace: Trace, obs: ChoiceMap, args: Tuple):
        pass

    def __call__(self, *args):
        return self.apply(*args)


##############
# Shorthands #
##############


@dispatch
def trace_translator(
    forward_kernel: TraceKernel,
    forward_kernel_args: Tuple,
):
    pass


@dispatch
def trace_translator(
    forward_kernel: TraceKernel,
    forward_kernel_args: Tuple,
    backward_kernel: TraceKernel,
    backward_kernel_args: Tuple,
):
    pass
