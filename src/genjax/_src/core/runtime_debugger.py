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
"""This module contains a debugger based around inserting/recording state from
pure functions."""

import functools
import inspect
from dataclasses import dataclass

import jax.core as jc
import jax.tree_util as jtu
from oryx import harvest
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
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.typing import typecheck


RECORDING_NAMESPACE = "debug_record"
TAGGING_NAMESPACE = "debug_tag"

###########
# Tagging #
###########


@typecheck
def tag(
    name: typing.Union[typing.String, typing.Tuple[typing.String, ...]],
    val: typing.Any,
) -> typing.Any:
    """Tag a value, allowing the debugger to store and return it as state in
    the `DebuggerTags` produced by `pull`.

    Arguments:
        name: A `String` or `Tuple[String, ...]` providing a reference name (key, into a `Trie`) for the tagged value.
        val: The value. Must be a JAX compatible type.

    Returns:
        val: The value. Must be a JAX compatible type.
    """
    f = functools.partial(
        harvest.sow,
        tag=TAGGING_NAMESPACE,
        meta=name,
    )
    return f(val)


###########
# Pulling #
###########


@dataclass
class Frame:
    filename: typing.String
    lineno: typing.Int
    module: typing.Any
    name: typing.String
    line: typing.String = ""


class PathHighlighter(RegexHighlighter):
    highlights = [r"(?P<dim>.*/)(?P<bold>.+)"]


@dataclass
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


@dataclass
class DebuggerTags(harvest.ReapState):
    tagged: Trie

    def flatten(self):
        return (self.tagged,), ()

    @classmethod
    def new(cls):
        trie = Trie.new()
        return DebuggerTags(trie)

    def sow(self, values, tree, meta, _):
        if meta in self.tagged:
            values = jtu.tree_leaves(harvest.tree_unreap(self.tagged[meta]))
        else:
            avals = jtu.tree_unflatten(
                tree,
                [jc.raise_to_shaped(jc.get_aval(v)) for v in values],
            )
            self.tagged[meta] = harvest.Reap.new(
                jtu.tree_unflatten(tree, values),
                dict(aval=avals),
            )

        return values

    ###########
    # Dunders #
    ###########

    def __getitem__(self, addr):
        v = self.tagged[addr]
        if v is None:
            raise KeyError(addr)
        else:
            return v


@dataclass
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

    def sow(self, values, tree, frame, _):
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
        _: ConsoleOptions,
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
            title="[traceback.title]Runtime debugger recording [dim](follows evaluation order)",
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
                title="recorded values",
                indent_guides=self.render_settings.indent_guides,
                max_length=self.render_settings.locals_max_length,
                max_string=self.render_settings.locals_max_string,
            )

        for frame, recorded in zip(stack, locals):

            text = Text.assemble(
                path_highlighter(Text(frame.filename, style="pygments.string")),
                (":", "pygments.text"),
                (str(frame.lineno), "pygments.number"),
                " in ",
                (frame.name, "pygments.function"),
                style="pygments.text",
            )
            yield text
            yield from render_locals(recorded)


@typecheck
def pull(
    f: typing.Callable,
) -> typing.Callable:
    """Transform a function into one which returns a debugger recording and
    debugger tags.

    Arguments:
        f: A function contain `record` invocations which will be transformed by the debugger transformation.

    Returns:
        callable: A new function which accepts the same arguments as the original function `f`, but returns a tuple, where the first element is the original return value, and the second element is a 2-tuple containing a `DebuggerRecording` instance and a `DebuggerTags` instance.


    Examples:
        Here's an example using pure functions, without generative semantics.

        ```python exec="yes" source="tabbed-left"
        import jax.numpy as jnp
        import genjax
        import genjax.core.runtime_debugger as debug
        console = genjax.console()

        def foo(x):
            v = jnp.ones(10) * x
            debug.record(v)
            z = v / 2
            return z

        v, (recording, tags) = debug.pull(foo)(3.0)
        print(console.render(recording))
        ```

        Here's an example where we mix `tag` and `record`.

        ```python exec="yes" source="tabbed-left"
        import jax.numpy as jnp
        import genjax
        import genjax.core.runtime_debugger as debug
        console = genjax.console()

        def foo(x):
            v = jnp.ones(10) * x
            debug.record(debug.tag("v", v))
            z = v / 2
            return z

        v, (recording, tags) = debug.pull(foo)(3.0)
        print(console.render(recording))
        print(console.render(tags))
        print(console.render(tags["v"]))
        ```

        Here's an example using generative functions. Now, `debug.record` will transform `GenerativeFunction` instances into `debug.DebugCombinator`, wrapping `debug.record_call` around their generative function interface invocations.

        ```python exec="yes" source="tabbed-left"
        import jax
        import jax.numpy as jnp
        import genjax
        import genjax.core.runtime_debugger as debug
        console = genjax.console()
        key = jax.random.PRNGKey(314159)

        @genjax.lang
        def foo(x):
            v = jnp.ones(10) * x
            x = debug.record(genjax.normal)(jnp.sum(v), 2.0) @ "x"
            return x

        v, (recording, tags) = debug.pull(foo.simulate)(key, (3.0, ))
        print(console.render(recording))
        ```
    """

    def _collect(f):
        return harvest.reap(
            harvest.reap(
                f,
                state=DebuggerRecording.new(),
                tag=RECORDING_NAMESPACE,
            ),
            state=DebuggerTags.new(),
            tag=TAGGING_NAMESPACE,
        )

    def wrapped(
        *args: typing.Any,
        **kwargs,
    ) -> typing.Tuple[typing.Any, typing.Tuple[DebuggerRecording, DebuggerTags]]:
        (v, recording_state), tagging_state = _collect(f)(*args, **kwargs)
        return v, (
            harvest.tree_unreap(recording_state),
            harvest.tree_unreap(tagging_state),
        )

    return wrapped


###########
# Pushing #
###########

plant_and_collect = functools.partial(
    harvest.harvest,
    tag=TAGGING_NAMESPACE,
)


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
def tag_with_frame(*args, frame: Frame):
    f = functools.partial(
        harvest.sow,
        tag=RECORDING_NAMESPACE,
        meta=frame,
    )
    return f(*args)


@typecheck
def record_call(f: typing.Callable) -> typing.Callable:
    """> Transform a function into a version which records the arguments to its
    invocation, as well as the return value.

    > The transformed version allows the debugger to store this
    > information in the debug recording, along other debug information,
    > including the definition file, the source line start, the module,
    > and the name of the function.

    The user is not expected to use this function, but to instead use the multimethod `record` below which will dispatch appropriately based on invocation types.
    """

    @functools.wraps(f)
    def wrapper(*args):
        retval = f(*args)
        sourceline_start = inspect.getsourcelines(f)[1]
        module = inspect.getmodule(f)
        name = f.__name__
        frame = Frame(
            repr(module),
            sourceline_start,
            module,
            name,
        )
        tag_with_frame(
            {"args": args, "return": retval},
            frame=frame,
        )
        return retval

    return wrapper


@typecheck
def record_value(value: typing.Any) -> typing.Any:
    """> Record a value, allowing the debugger to store it in the debug
    recording, along with the caller's stack frame information.

    The user is not expected to use this function, but to instead use the multimethod `record` below which will dispatch appropriately based on invocation types.
    """
    caller_frame_info = inspect.stack()[3]
    file_name = caller_frame_info.filename
    source_line = caller_frame_info.lineno
    name = caller_frame_info.function
    module = ""
    frame = Frame(
        file_name,
        source_line,
        module,
        name,
    )
    tag_with_frame(
        {"value": value},
        frame=frame,
    )
    return value
