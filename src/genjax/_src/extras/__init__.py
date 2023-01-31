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

"""This module provides compatibility extension plugins for packages which
provide functionality that is useful for modeling and inference.

Submodules present compatibility layers for usage of these packages with
GenJAX.
"""

import importlib
import importlib.util
import sys
import types


class LazyLoader(types.ModuleType):
    def __init__(self, local_name, parent_module_globals, name):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        super(LazyLoader, self).__init__(name)

    def _load(self):
        try:
            module = importlib.import_module(self.__name__)
            self._parent_module_globals[self._local_name] = module
            self.__dict__.update(module.__dict__)
            return module
        except ModuleNotFoundError as e:
            e.add_note(
                f"(GenJAX) Attempted to load {self._local_name} extension but failed, is it installed in your environment?"
            )
            raise e

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)


# BlackJAX provides HMC samplers.
blackjax = LazyLoader(
    "blackjax",
    globals(),
    "genjax._src.extras.blackjax",
)

# tinygp provides Gaussian process model ingredients.
tinygp = LazyLoader(
    "tinygp",
    globals(),
    "genjax._src.extras.tinygp",
)

# Equinox provides neural networks.
equinox = LazyLoader(
    "equinox",
    globals(),
    "genjax._src.extras.equinox",
)
