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

"""This module contains a debugger based around inserting/recording state from
pure functions."""

import functools

from genjax._src.core.interpreters import harvest
from genjax._src.core.typing import Dict
from genjax._src.core.typing import String
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import typecheck


NAMESPACE = "debug"


tag = functools.partial(harvest.sow, tag=NAMESPACE)


def grab(f):
    def wrapped(*args, **kwargs):
        return harvest.harvest(f, tag=NAMESPACE)({}, *args)

    return wrapped


def stab(f):
    @typecheck
    def wrapped(plants: Dict, args: Tuple):
        v, state = harvest.harvest(f, tag=NAMESPACE)(plants, *args)
        return v, {**plants, **state}

    return wrapped
