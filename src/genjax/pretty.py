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

import inspect
from dataclasses import fields

from penzai import pz
from penzai.treescope import default_renderer
from penzai.treescope.foldable_representation import (
    basic_parts,
    common_structures,
    common_styles,
    foldable_impl,
)
from penzai.treescope.handlers import builtin_structure_handler
from penzai.treescope.handlers.penzai import struct_handler

from genjax._src.core.pytree import Pytree


def pretty():
    def _pytree_handler(node, subtree_renderer):
        constructor_open = struct_handler.render_struct_constructor(node)
        fs = fields(node)

        (
            background_color,
            background_pattern,
        ) = builtin_structure_handler.parse_color_and_pattern(
            node.treescope_color(), type(node).__name__
        )

        if background_pattern is not None:
            if background_color is None:
                raise ValueError(
                    "background_color must be provided if background_pattern is"
                )

            def wrap_block(block):
                return common_styles.WithBlockPattern(
                    block, color=background_color, pattern=background_pattern
                )

            wrap_topline = common_styles.PatternedTopLineSpanGroup
            wrap_bottomline = common_styles.PatternedBottomLineSpanGroup

        elif background_color is not None and background_color != "transparent":

            def wrap_block(block):
                return common_styles.WithBlockColor(block, color=background_color)

            wrap_topline = common_styles.ColoredTopLineSpanGroup
            wrap_bottomline = common_styles.ColoredBottomLineSpanGroup

        else:

            def id(rendering):
                return rendering

            wrap_block = id
            wrap_topline = id
            wrap_bottomline = id

        children = builtin_structure_handler.build_field_children(
            node,
            None,
            subtree_renderer,
            fields_or_attribute_names=fs,
            key_path_fn=node.key_for_field,
            attr_style_fn=struct_handler.struct_attr_style_fn_for_fields(fs),
        )
        children = basic_parts.IndentedChildren(children)

        suffix = ")"

        return wrap_block(
            basic_parts.Siblings(
                children=[
                    wrap_topline(constructor_open),
                    basic_parts.Siblings.build(
                        foldable_impl.HyperlinkTarget(
                            foldable_impl.FoldableTreeNodeImpl(
                                basic_parts.FoldCondition(
                                    collapsed=basic_parts.Text("..."),
                                    expanded=children,
                                )
                            ),
                            keypath=None,
                        ),
                        wrap_bottomline(basic_parts.Text(suffix)),
                    ),
                ],
            )
        )

    def custom_handler(node, path, subtree_renderer):
        if inspect.isfunction(node):
            return common_structures.build_one_line_tree_node(
                line=common_styles.CustomTextColor(
                    basic_parts.Text(f"<fn {node.__name__}>"),
                    color="blue",
                ),
                path=None,
            )
        if isinstance(node, Pytree):
            return _pytree_handler(node, subtree_renderer)
        return NotImplemented

    default_renderer.active_renderer.get().handlers.insert(0, custom_handler)

    pz.ts.register_as_default()
    pz.ts.register_autovisualize_magic()
    pz.enable_interactive_context()

    # Optional: enables automatic array visualization
    pz.ts.active_autovisualizer.set_interactive(pz.ts.ArrayAutovisualizer())


__all__ = [
    "pretty",
]
