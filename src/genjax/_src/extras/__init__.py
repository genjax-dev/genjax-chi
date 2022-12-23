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


def lazy_extras_import(name):
    spec = importlib.util.find_spec(f"genjax.extras.{name}")
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"genjax.extras.{name}"] = module
    loader.exec_module(module)
    return module


# BlackJAX provides HMC samplers.
def blackjax():
    """Load the :code:`BlackJAX` compatibility layer."""
    return lazy_extras_import("blackjax")


# tinygp provides Gaussian process model ingredients.
def tinygp():
    """Load the :code:`tinygp` compatibility layer."""
    return lazy_extras_import("tinygp")
