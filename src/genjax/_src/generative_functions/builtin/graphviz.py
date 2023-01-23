# Copyright 2020 MIT Probabilistic Computing Project &
# DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EXPERIMENTAL: This module supports an adapted version of the :code:`to_dot` functionality
from DeepMind's :code:`dm-haiku` library. For GenJAX, we've removed all
references.

to :code:`haiku.Module` - and added new hooks that allow customization for
user-defined objects (to support custom :code:`dot` conversion for
generative functions, for example).

Conversion is implemented using an interpreter (defining its own :code:`jax.Trace` and :code:`jax.Tracer` types) to convert callables which are JAX traceable into Graphviz `dot` graphs.
"""

import collections
import functools
import html
from typing import List
from typing import NamedTuple
from typing import Optional

import jax
import jax.core as core
import jax.tree_util as jtu
import tree

from genjax._src.core.pretty_printing import simple_dtype
from genjax._src.core.staging import stage
from genjax._src.generative_functions.builtin.intrinsics import gen_fn_p


safe_map = core.safe_map
safe_zip = core.safe_zip

########################
# Graph representation #
########################

Node = collections.namedtuple("Node", "id,title,outputs")
Edge = collections.namedtuple("Edge", "a,b")


class Graph(NamedTuple):
    """Represents a Graphviz digraph/subgraph."""

    title: str
    nodes: List[Node]
    edges: List[Edge]
    subgraphs: List["Graph"]

    @classmethod
    def create(cls, title: Optional[str] = None):
        return Graph(title=title, nodes=[], edges=[], subgraphs=[])

    def evolve(self, **kwargs) -> "Graph":
        return Graph(**{**self._asdict(), **kwargs})


###############
# Interpreter #
###############


def name_or_str(o):
    return getattr(o, "__name__", str(o))


# This is a custom interpreter, with minimal complexity.
#
# For Graphviz conversion, we just need a traversal pattern,
# which doesn't need to stage out state, etc.


def _handle_trace(*args, addr, gen_fn, tree_in):
    key, *args = jtu.tree_unflatten(tree_in, args)
    g, outvals = make_graph(gen_fn.__call__)(key, *args)
    return g, outvals


def eval_jaxpr_graph(jaxpr, consts, *args):
    env = {}
    node_env = {}
    graph = Graph.create()

    def read(var):
        if type(var) is core.Literal:
            return var.val
        return env[var]

    def write(var, val):
        env[var] = val

    def node_write(var, val):
        node_env[var] = val

    def node_read(var):
        if type(var) is core.Literal:
            return var.val
        return node_env[var]

    safe_map(write, jaxpr.invars, args)
    safe_map(write, jaxpr.constvars, consts)
    safe_map(node_write, jaxpr.invars, args)
    safe_map(node_write, jaxpr.constvars, consts)

    for v in consts:
        node = Node(id=v, title=str(v), outputs=[v])
        graph.nodes.append(node)

    for eqn in jaxpr.eqns:
        if eqn.primitive == gen_fn_p:
            invals = safe_map(read, eqn.invars)
            params = eqn.params
            subgraph, outvals = _handle_trace(*invals, **params)
            graph.subgraphs.append(subgraph)
            id_name = outvals[0]
            safe_map(write, eqn.outvars, outvals)
            safe_map(node_write, eqn.outvars, [id_name for _ in eqn.outvars])
        else:
            outvals = [v.aval for v in eqn.outvars]
            id_name = outvals[0]
            primitive = eqn.primitive
            node = Node(
                id=id_name,
                title=str(primitive),
                outputs=outvals,
            )
            graph.nodes.append(node)
            inodes = safe_map(node_read, eqn.invars)
            graph.edges.extend([(i, id_name) for i in inodes])
            safe_map(node_write, eqn.outvars, [id_name for _ in eqn.outvars])
            safe_map(write, eqn.outvars, outvals)

    outvals = safe_map(read, jaxpr.outvars)
    return graph, outvals


def make_graph(fun):
    @functools.wraps(fun)
    def wrapped_fun(*args):
        closed_jaxpr, (flat_args, _, _) = stage(fun)(*args)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        graph, outvals = eval_jaxpr_graph(jaxpr, consts, *flat_args)
        graph = graph.evolve(title=name_or_str(fun))
        return graph, outvals

    return wrapped_fun


##############
# Transpiler #
##############


def _format_val(val):
    if not hasattr(val, "shape"):
        return repr(val)
    shape = ",".join(map(str, val.shape))
    dtype = simple_dtype(val.dtype)
    return f"{dtype}[{shape}]"


def escape(value):
    return html.escape(str(value))


# Determine maximum nesting depth to appropriately scale subgraph labels.
def _max_depth(g: Graph) -> int:
    if g.subgraphs:
        return 1 + max(0, *[_max_depth(s) for s in g.subgraphs])
    else:
        return 1


def _scaled_font_size(depth: int) -> int:
    return int(1.4**depth * 14)


def graph_to_dot(graph: Graph, args, outputs) -> str:
    """Converts from an internal graph IR to 'dot' format."""

    def format_path(path):
        if isinstance(outputs, tuple):
            out = f"output[{path[0]}]"
            if len(path) > 1:
                out += ": " + "/".join(map(str, path[1:]))
        else:
            out = "output"
            if path:
                out += ": " + "/".join(map(str, path))
        return out

    lines = []
    used_argids = set()
    argid_usecount = collections.Counter()
    op_outids = set()
    captures = []
    argids = {id(v) for v in jax.tree_util.tree_leaves(args)}
    outids = {id(v) for v in jax.tree_util.tree_leaves(outputs)}
    outname = {id(v): format_path(p) for p, v in tree.flatten_with_path(outputs)}

    def render_graph(g: Graph, parent: Optional[Graph] = None, depth: int = 0):
        """Renders a given graph by appending 'dot' format lines."""

        if parent:
            lines.extend(
                [
                    f"subgraph cluster_{id(g)} {{",
                    '  style="rounded,filled";',
                    '  fillcolor="#F0F5F5";',
                    '  color="#14234B;";',
                    "  pad=0.1;",
                    f"  fontsize={_scaled_font_size(depth)};",
                    f"  label = <<b>{escape(g.title)}</b>>;",
                    "  labelloc = t;",
                ]
            )

        for node in g.nodes:
            label = f"<b>{escape(node.title)}</b>"
            for o in node.outputs:
                label += "<br/>" + _format_val(o)
                op_outids.add(id(o))

            node_id = id(node.id)
            if node_id in outids:
                label = f"<b>{escape(outname[node_id])}</b><br/>" + label
                color = "#0053D6"
                fillcolor = "#AABFFF"
                style = "filled,bold"
            else:
                color = "#FFDB13"
                fillcolor = "#FFF26E"
                style = "filled"

            lines.append(
                f"{node_id} [label=<{label}>, "
                f' id="node{node_id}",'
                " shape=rect,"
                f' style="{style}",'
                ' tooltip=" ",'
                ' fontcolor="black",'
                f' color="{color}",'
                f' fillcolor="{fillcolor}"];'
            )

        for s in g.subgraphs:
            render_graph(s, parent=g, depth=depth - 1)

        if parent:
            lines.append(f"}}  // subgraph cluster_{id(g)}")

        for a, b in g.edges:
            if id(a) not in argids and id(a) not in op_outids:
                captures.append(a)

            a, b = map(id, (a, b))
            if a in argids:
                i = argid_usecount[a]
                argid_usecount[a] += 1
                lines.append(f"{a}{i} -> {b};")
            else:
                lines.append(f"{a} -> {b};")
            used_argids.add(a)

    graph_depth = _max_depth(graph)
    render_graph(graph, parent=None, depth=graph_depth)

    # Process inputs and label them in the graph.
    for path, value in tree.flatten_with_path(args):
        if value is None:
            continue

        node_id = id(value)
        if node_id not in used_argids:
            continue

        for i in range(argid_usecount[node_id]):
            label = f"<b>args[{escape(path[0])}]"
            if len(path) > 1:
                label += ": " + "/".join(map(str, path[1:]))
            label += "</b>"
            if hasattr(value, "shape") and hasattr(value, "dtype"):
                label += f"<br/>{escape(_format_val(value))}"
            fillcolor = "#FFDEAF"
            fontcolor = "black"

            if i > 0:
                label = "<b>(reuse)</b><br/>" + label
                fillcolor = "#FFEACC"
                fontcolor = "#565858"

            lines.append(
                f"{node_id}{i} [label=<{label}>"
                f' id="node{node_id}{i}",'
                " shape=rect,"
                ' style="filled",'
                f' fontcolor="{fontcolor}",'
                ' color="#FF8A4F",'
                f' fillcolor="{fillcolor}"];'
            )

    for value in captures:
        node_id = id(value)
        if not hasattr(value, "aval") and hasattr(value, "size") and value.size == 1:
            label = f"<b>{value.item()}</b>"
        else:
            label = f"<b>{escape(_format_val(value))}</b>"

        lines.append(
            f"{node_id} [label=<{label}>"
            " shape=rect,"
            ' style="filled",'
            ' fontcolor="black",'
            ' color="#A261FF",'
            ' fillcolor="#E6D6FF"];'
        )

    head = [
        "digraph G {",
        "rankdir = TD;",
        "compound = true;",
        f"label = <<b>{escape(graph.title)}</b>>;",
        f"fontsize={_scaled_font_size(graph_depth)};",
        "labelloc = t;",
        "stylesheet = <",
        "  data:text/css,",
        "  @import url(https://fonts.googleapis.com/css?family=Roboto:400,700);",
        "  svg text {",
        "    font-family: 'Roboto';",
        "  }",
        "  .node text {",
        "    font-size: 12px;",
        "  }",
    ]
    for node_id, use_count in argid_usecount.items():
        if use_count == 1:
            continue
        # Add hover animation for reused args.
        for a in range(use_count):
            for b in range(use_count):
                if a == b:
                    head.append(
                        f"%23node{node_id}{a}:hover " "{ stroke-width: 0.2em; }"
                    )
                else:
                    head.append(
                        f"%23node{node_id}{a}:hover ~ %23node{node_id}{b} "
                        "{ stroke-width: 0.2em; }"
                    )
    head.append(">")

    lines.append("} // digraph G")
    return "\n".join(head + lines) + "\n"


def make_dot(fun):
    @functools.wraps(fun)
    def wrapped_fun(*args):
        graph, outvals = make_graph(fun)(*args)
        dot = graph_to_dot(graph, args, outvals)
        return dot

    return wrapped_fun
