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
"GenJAX is a probabilistic programming system constructed by combining the concepts of Gen with the program transformation and hardware accelerator compilation capabilities of JAX."

# This __init__ file exports GenJAX's public API.
# For the internals, see _src.

# Closed modules.
from importlib import metadata

from .console import *
from .core import *
from .experimental import *
from .extras import *
from .generative_functions import *
from .inference import *
from .information import *

__version__ = metadata.version("genjax")
