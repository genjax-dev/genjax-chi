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

import inspect

import jax
import rich
import rich.traceback as traceback
from rich.console import Console

from .core import *
from .experimental import *
from .generative_functions import *
from .inference import *
from .interface import *


#####
# Language decorator
#####


def gen(callable: Callable, **kwargs) -> GenerativeFunction:
    if inspect.isclass(callable):
        return lambda source: callable(
            BuiltinGenerativeFunction(source),
            **kwargs,
        )
    else:
        return BuiltinGenerativeFunction(callable)


#####
# Pretty printing
#####


@dataclass
class GenJAXConsole:
    rich_console: Console

    def print(self, obj):
        self.rich_console.print(obj)

    def inspect(self, obj, **kwargs):
        rich.inspect(obj, console=self.rich_console, **kwargs)

    def help(self, obj):
        rich.inspect(
            obj,
            console=self.rich_console,
            methods=True,
            help=True,
            value=False,
            private=False,
            dunder=False,
        )


def pretty(show_locals=False, max_frames=20, suppress=[jax]):
    rich.pretty.install()
    traceback.install(
        show_locals=show_locals,
        max_frames=max_frames,
        suppress=[jax],
    )

    return GenJAXConsole(Console(soft_wrap=True))
