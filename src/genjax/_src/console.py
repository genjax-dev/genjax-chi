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

import graphviz
import jax
import objexplore
import rich
import rich.traceback as traceback
from rich.console import Console

from genjax._src.generative_functions.builtin.graphviz import make_dot


#####
# Pretty printing
#####


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


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

    def explore(self, module):
        if is_notebook():
            raise Exception("Interactive explore only works in terminal.")
        else:
            objexplore.explore(module)

    # EXPERIMENTAL.
    def graph(self, fn, *args):
        g = make_dot(fn)(*args)
        g = graphviz.Source(g)
        if is_notebook():
            return g
        else:
            g.view()


def pretty(show_locals=False, max_frames=20, suppress=[jax], **kwargs):
    rich.pretty.install()
    traceback.install(
        show_locals=show_locals,
        max_frames=max_frames,
        suppress=suppress,
    )

    return GenJAXConsole(Console(soft_wrap=True, **kwargs))
