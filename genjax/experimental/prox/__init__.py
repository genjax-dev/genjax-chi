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
This module is a GenJAX implementation of Alexander K. Lew's
framework for inference-as-generative-functions (RAVI) 
https://arxiv.org/abs/2203.02836 and his Gen implementation GenProx.
"""

from .prox_distribution import *
from .choice_map_distribution import *
from .target import *
from .marginal import *
from .inference import *
