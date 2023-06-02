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

"""This module contains several standard generative function classes useful for
structuring probabilistic programs.

* The `distributions` module exports standard distributions from several sources, including SciPy (`scipy`), TensorFlow Probability Distributions (`tfd`), and custom distributions.
    * The `distributions` module also contains a small `oryx`-like language called `coryx` which implements the generative function interface for programs with inverse log determinant Jacobian (ildj) compatible return value functions of distribution random choices.
    * The `distributions` module also contains an implementation of `prox`, a research language for probabilistic programming with density estimates.
* The `builtin` module contains a function-like language for defining generative functions from programs.
* The `combinators` module contains combinators which support transforming generative functions into new ones with structured control flow patterns of computation.
"""

from .builtin import *
from .combinators import *
from .distributions import *
