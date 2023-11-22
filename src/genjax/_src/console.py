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

from dataclasses import dataclass

import jax
import plum
import rich
from rich.console import Console

from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.typing import Dict


###################
# Pretty printing #
###################


@dataclass
class GenJAXConsole(Pytree):
    rich_console: Console
    traceback_kwargs: Dict

    def flatten(self):
        return (), (self.rich_console,)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is not None:
            self.rich_console.print_exception(
                **self.traceback_kwargs,
            )
        return True

    def print(self, obj):
        self.rich_console.print(
            obj,
            soft_wrap=True,
            overflow="ellipsis",
        )

    def render(self, obj):
        console = Console(soft_wrap=True, record=True)
        with console.capture() as _:
            console.print(
                obj,
                soft_wrap=True,
                overflow="ellipsis",
            )
        str_output = console.export_text()
        return f"```raw\n{str_output}```"

    def inspect(self, obj, **kwargs):
        rich.inspect(obj, console=self.rich_console, **kwargs)


def pretty(
    show_locals=False,
    max_frames=30,
    suppress=[jax, plum],
    **kwargs,
):
    traceback_kwargs = {
        "show_locals": show_locals,
        "max_frames": max_frames,
        "suppress": suppress,
        **kwargs,
    }
    return GenJAXConsole(
        Console(soft_wrap=True),
        traceback_kwargs,
    )
