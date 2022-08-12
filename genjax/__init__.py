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
<br>
<p align="center">
<img width="400px" src="./assets/logo.png"/>
</p>
<br>
`genjax` is a probabilistic programming library built by combining the conceptual framework of [Gen](https://www.mct.dev/assets/mct-thesis.pdf) with the high-performance numerical compilation of [JAX](https://github.com/google/jax). In several ways, these two systems are a natural pair - JAX provides flexibility to construct composable function transformations as interpreters, and Gen provides a rich set of modelling and inference ideas grounded in a functional representations of generative processes.

## High-level

- Generative functions (models) are represented as pure functions from `(PRNGKey, *args)` to `(PRNGKey, retval)`.
- Exposes [the generative function interface](https://www.gen.dev/stable/ref/gfi/) as staged effect handlers built on top of `jax`.

  | Interface     | Semantics (informal)                                                                |
  | ------------- | ----------------------------------------------------------------------------------- |
  | `simulate`    | Sample from normalized measure over choice maps                                     |
  | `importance`  | Importance sample from conditioned measure, and compute an importance weight        |
  | `arg_grad`    | Compute gradient of `logpdf` of choice map with respect to arguments                |
  | `choice_grad` | Compute gradient of `logpdf` of choice map with respect to values of random choices |

- Supports usage of any computations acceptable by JAX (tbd) within generative function programs.

<div align="center">
<b>(Early stage)</b> 🔪 expect sharp edges 🔪
</div>
"""

# Public exports.
from .core import *
from .distributions import *
from .intrinsics import *
from .handlers import *
from .generative_function import *
