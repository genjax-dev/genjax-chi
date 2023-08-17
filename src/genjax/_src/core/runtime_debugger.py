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

import dataclasses
import functools
import inspect

import jax.core as jc
import jax.tree_util as jtu

from genjax._src.core.transforms import harvest
from genjax._src.core.typing import Dict
from genjax._src.core.typing import List
from genjax._src.core.typing import Tuple
from genjax._src.core.typing import typecheck


NAMESPACE = "debug"

###########
# Tagging #
###########


def tag(*args):
    stack = inspect.stack()
    caller_frame = stack[1]
    caller_filename = caller_frame.filename
    caller_lineno = caller_frame.lineno
    caller_function = caller_frame.function
    f = functools.partial(
        harvest.sow,
        tag=NAMESPACE,
        name=(caller_filename, caller_function, caller_lineno),
    )
    return f(*args)


def tag_with_name(*args, name):
    f = functools.partial(
        harvest.sow,
        tag=NAMESPACE,
        name=name,
    )
    return f(*args)


###########
# Pulling #
###########


@dataclasses.dataclass
class DebuggerRecording(harvest.ReapState):
    call_stack: List
    recorded: List

    def flatten(self):
        return (self.recorded,), (self.call_stack,)

    @classmethod
    def new(cls):
        return DebuggerRecording([], [])

    def sow(self, values, tree, name, tag):
        avals = jtu.tree_unflatten(
            tree,
            [jc.raise_to_shaped(jc.get_aval(v)) for v in values],
        )
        self.recorded.append(
            harvest.Reap.new(
                jtu.tree_unflatten(tree, values),
                dict(aval=avals),
            )
        )
        self.call_stack.append(name)

        return values

    def __getitem__(self, idx):
        return (self.call_stack[idx], self.recorded[idx])


def pull(f):
    _collect = functools.partial(
        harvest.reap,
        state=DebuggerRecording.new(),
        tag=NAMESPACE,
    )

    def wrapped(*args, **kwargs):
        v, state = _collect(f)(*args, **kwargs)
        return v, harvest.tree_unreap(state)

    return wrapped


###########
# Pushing #
###########

plant_and_collect = functools.partial(harvest.harvest, tag=NAMESPACE)


def push(f):
    @typecheck
    def wrapped(plants: Dict, args: Tuple, **kwargs):
        v, state = plant_and_collect(f)(plants, *args, **kwargs)
        return v, {**plants, **state}

    return wrapped
