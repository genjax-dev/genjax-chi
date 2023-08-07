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
"GenJAX is a probabilistic programming system constructed by combining the concepts of Gen with the program transformation and hardware accelerator compilation capabilities of JAX."

# This __init__ file exports GenJAX's public API.
# For the internals, see _src.

# Closed modules.
from genjax import inference
from genjax import typing
from genjax.core import interpreters
from genjax.generative_functions import gentle
from genjax.generative_functions.distributions import coryx
from genjax.generative_functions.distributions import gensp

from .adev import *
from .console import *
from .core import *
from .debugging import *
from .experimental import *
from .extras import *
from .generative_functions import *
from .global_options import *
from .inference import *
from .information import *
from .interface import *
from .language_decorator import *
from .learning import *
from .utilities import *


__version__ = "0.0.1"

####################################################
#
#   The exports defined above are the public API.
#
#                        /\_/\____,
#              ,___/\_/\ \  ~     /
#              \     ~  \ )   XXX
#                XXX     /    /\_/\___,
#                   \o-o/-o-o/   ~    /
#                    ) /     \    XXX
#                   _|    / \ \_/
#                ,-/   _  \_/   \
#               / (   /____,__|  )
#              (  |_ (    )  \) _|
#             _/ _)   \   \__/   (_
#            (,-(,(,(,/      \,),),)
#
#
#       "Abandon all hope, ye who enter _src."
#
####################################################

try:
    del genjax._src
except NameError:
    pass
