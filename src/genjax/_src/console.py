# Copyright 2024 MIT Probabilistic Computing Project
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


import jax
import plum
import rich
from penzai import pz
from rich import traceback
from rich.console import Console

from genjax._src.checkify import no_checkify, yes_checkify
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import Any, Bool, Dict

###################
# Pretty printing #
###################


class GenJAXConsole(Pytree):
    rich_console: Console = Pytree.static()
    traceback_kwargs: Dict = Pytree.static()
    enforce_checkify: Bool = Pytree.static()

    def __enter__(self):
        if self.enforce_checkify:
            yes_checkify()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if self.enforce_checkify:
            no_checkify()
        if exc_type is not None:
            show_locals = self.traceback_kwargs["show_locals"]
            trace = traceback.Traceback.extract(
                exc_type,
                exc_value,
                tb,
                show_locals=show_locals,
                locals_hide_sunder=True,
                locals_hide_dunder=True,
            )
            rich_tb = traceback.Traceback(
                trace,
                **self.traceback_kwargs,
            )
            self.rich_console.print(rich_tb)
        return True

    def print(self, obj, roundtrip_mode=False):
        with pz.ts.active_autovisualizer.set_scoped(pz.ts.ArrayAutovisualizer()):
            pz.ts.display(obj, roundtrip_mode=roundtrip_mode)

    def time_print(
        self,
        obj,
        leading_axes=["time"],
        roundtrip_mode=False,
    ):
        def temporal_autovisualizer(
            value: Any,
            path: tuple[Any, ...] | None,
        ):
            if isinstance(value, jax.Array):
                num_provided_axes = len(leading_axes)
                other_axes = [
                    f"axis_{i}" for i in range(len(value.shape[num_provided_axes:]))
                ]
                values_over_time = pz.nx.wrap(value).tag(
                    *leading_axes,
                    *other_axes,
                )
                return pz.ts.IPythonVisualization(
                    pz.ts.render_array(
                        values_over_time,
                        sliders=leading_axes,
                        continuous=True,
                        around_zero=False,
                    )
                )

        with pz.ts.active_autovisualizer.set_scoped(temporal_autovisualizer):
            pz.ts.display(obj, roundtrip_mode=roundtrip_mode)

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


def console(
    enforce_checkify=False,
    **pretty_kwargs,
):
    try:
        # Try to ignore these packages in pretty printing.
        import asyncio

        import ipykernel
        import tornado
        import traitlets

        traceback_kwargs = {
            "word_wrap": True,
            "show_locals": False,
            "max_frames": 30,
            "suppress": [
                jax,
                plum,
                asyncio,
                tornado,
                traitlets,
                ipykernel,
            ],
            **pretty_kwargs,
        }
    except Exception:
        traceback_kwargs = {
            "word_wrap": True,
            "show_locals": False,
            "max_frames": 30,
            "suppress": [jax, plum],
            **pretty_kwargs,
        }

    finally:
        return GenJAXConsole(
            Console(soft_wrap=True),
            traceback_kwargs,
            enforce_checkify,
        )
