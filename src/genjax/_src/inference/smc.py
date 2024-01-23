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

from abc import abstractmethod

from genjax._src.core.datatypes.generative import Choice
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import Int, PRNGKey
from genjax._src.inference.core import ChoiceDistribution, InferenceAlgorithm, Target
from genjax._src.inference.translator import TraceTranslator


class SMCAlgorithm(InferenceAlgorithm):
    @abstractmethod
    def get_num_particles(self):
        raise NotImplementedError

    @abstractmethod
    def get_final_target(self):
        raise NotImplementedError

    @abstractmethod
    def run_smc(self, key: PRNGKey):
        raise NotImplementedError

    @abstractmethod
    def run_csmc(self, key: PRNGKey, retained: Choice):
        raise NotImplementedError


class Initialize(SMCAlgorithm):
    q: ChoiceDistribution
    target: Target
    n_particles: Int = Pytree.static()

    def get_num_particles(self):
        return self.n_particles

    def get_final_target(self):
        return self.target

    def run_smc(self, key: PRNGKey):
        pass

    def run_csmc(self, key: PRNGKey, retained: Choice):
        pass


class TranslatorStep(SMCAlgorithm):
    prev: SMCAlgorithm
    translator: TraceTranslator

    def get_num_particles(self):
        return self.prev.get_num_particles()

    def get_final_target(self):
        return self.prev.get_final_target()
