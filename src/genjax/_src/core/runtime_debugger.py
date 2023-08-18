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
from pygments.token import Comment
from pygments.token import Keyword
from pygments.token import Name
from pygments.token import Number
from pygments.token import Operator
from pygments.token import String
from pygments.token import Text as TextToken
from pygments.token import Token
from rich import pretty
from rich.console import Console
from rich.console import ConsoleOptions
from rich.console import ConsoleRenderable
from rich.console import RenderResult
from rich.console import group
from rich.constrain import Constrain
from rich.highlighter import RegexHighlighter
from rich.panel import Panel
from rich.scope import render_scope
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme

import genjax._src.core.typing as typing
from genjax._src.core.transforms import harvest
from genjax._src.core.typing import typecheck


NAMESPACE = "debug"

###########
# Tagging #
###########


def tag(*args):
    stack = inspect.stack()
    caller_frame_info = stack[1]
    f = functools.partial(
        harvest.sow,
        tag=NAMESPACE,
        name=caller_frame_info,
    )
    return f(*args)


def tag_with_meta(*args, meta):
    f = functools.partial(
        harvest.sow,
        tag=NAMESPACE,
        name=meta,
    )
    return f(*args)


###########
# Pulling #
###########


@dataclasses.dataclass
class Frame:
    filename: typing.String
    lineno: typing.Int
    module: typing.Any
    name: typing.String
    line: typing.String = ""


class PathHighlighter(RegexHighlighter):
    highlights = [r"(?P<dim>.*/)(?P<bold>.+)"]


@dataclasses.dataclass
class RenderSettings:
    theme: Theme
    width: typing.Int
    indent_guides: typing.Bool
    locals_max_length: typing.Optional[typing.Int]
    locals_max_string: typing.Optional[typing.Int]

    @classmethod
    def new(cls):
        theme = Syntax.get_theme("ansi_dark")
        width = 100
        indent_guides = True
        locals_max_length = None
        locals_max_string = None
        return RenderSettings(
            theme,
            width,
            indent_guides,
            locals_max_length,
            locals_max_string,
        )


@dataclasses.dataclass
class DebuggerRecording(harvest.ReapState):
    render_settings: RenderSettings
    frames: typing.List[Frame]
    recorded: typing.List[typing.Any]

    def flatten(self):
        return (self.recorded,), (self.render_settings, self.frames)

    @classmethod
    def new(cls):
        render_settings = RenderSettings.new()
        return DebuggerRecording(render_settings, [], [])

    def sow(self, values, tree, frame, tag):
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
        self.frames.append(frame)

        return values

    def __getitem__(self, idx):
        return DebuggerRecording(
            [self.frames[idx]],
            [self.recorded[idx]],
        )

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        background_style = self.render_settings.theme.get_background_style()
        token_style = self.render_settings.theme.get_style_for_token
        theme = Theme(
            {
                "pretty": token_style(TextToken),
                "pygments.text": token_style(Token),
                "pygments.string": token_style(String),
                "pygments.function": token_style(Name.Function),
                "pygments.number": token_style(Number),
                "repr.indent": token_style(Comment) + Style(dim=True),
                "repr.str": token_style(String),
                "repr.brace": token_style(TextToken) + Style(bold=True),
                "repr.number": token_style(Number),
                "repr.bool_true": token_style(Keyword.Constant),
                "repr.bool_false": token_style(Keyword.Constant),
                "repr.none": token_style(Keyword.Constant),
                "scope.border": token_style(String.Delimiter),
                "scope.equals": token_style(Operator),
                "scope.key": token_style(Name),
                "scope.key.special": token_style(Name.Constant) + Style(dim=True),
            },
            inherit=False,
        )

        rendered: ConsoleRenderable = Panel(
            self._render_frames(self.frames, self.recorded),
            title="Runtime debugger recording (top record last)",
            style=background_style,
            border_style="traceback.border",
            expand=True,
            padding=(0, 1),
        )
        rendered = Constrain(rendered, self.render_settings.width)
        with console.use_theme(theme):
            yield rendered

    @group()
    def _render_frames(
        self,
        stack: typing.List[Frame],
        locals: typing.List[typing.Any],
    ) -> RenderResult:

        path_highlighter = PathHighlighter()
        theme = self.render_settings.theme

        def render_locals(locals: typing.Any) -> typing.Iterable[ConsoleRenderable]:
            locals = {
                key: pretty.traverse(
                    value,
                    max_length=self.render_settings.locals_max_length,
                    max_string=self.render_settings.locals_max_string,
                )
                for key, value in locals.items()
            }
            yield render_scope(
                locals,
                title="ins/outs",
                indent_guides=self.render_settings.indent_guides,
                max_length=self.render_settings.locals_max_length,
                max_string=self.render_settings.locals_max_string,
            )

        for (frame_index, frame), recorded in zip(enumerate(stack), locals):
            first = frame_index == 0

            text = Text.assemble(
                path_highlighter(Text(frame.filename, style="pygments.string")),
                (":", "pygments.text"),
                (str(frame.lineno), "pygments.number"),
                " in ",
                (frame.name, "pygments.function"),
                style="pygments.text",
            )
            if not frame.filename.startswith("<") and not first:
                yield ""
            yield text
            if frame.filename.startswith("<"):
                yield from render_locals(recorded)


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
    def wrapped(plants: typing.Dict, args: typing.Tuple, **kwargs):
        v, state = plant_and_collect(f)(plants, *args, **kwargs)
        return v, {**plants, **state}

    return wrapped


############################
# Record a call as a frame #
############################


@typecheck
def record_call(f: typing.Callable) -> typing.Callable:
    @functools.wraps(f)
    def wrapper(*args):
        retval = f(*args)
        file_name = inspect.getfile(f)
        sourceline_start = inspect.getsourcelines(f)[1]
        module = inspect.getmodule(f)
        name = f.__name__
        frame = Frame(
            file_name,
            sourceline_start,
            module,
            name,
        )
        tag_with_meta(
            {"args": args, "return": retval},
            meta=frame,
        )
        return retval

    return wrapper
