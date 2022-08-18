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

"""
Abstract base class for all `Distribution` inheritors.
"""
import abc
from genjax.core import Pytree


class Distribution(Pytree, metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def flatten(cls):
        pass

    @classmethod
    @abc.abstractmethod
    def unflatten(cls):
        pass

    @classmethod
    @abc.abstractmethod
    def abstract_eval(cls, *args, **kwargs):
        pass

    @classmethod
    @abc.abstractmethod
    def abstract_eval_batched(cls, *args, **kwargs):
        pass

    @classmethod
    @abc.abstractmethod
    def sample(cls, *args, **kwargs):
        pass

    @classmethod
    @abc.abstractmethod
    def score(cls, *args, **kwargs):
        pass
