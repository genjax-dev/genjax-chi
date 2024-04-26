#######################################################################
# %% use Penzai as default pretty-printer, hide all copy buttons.

from penzai import pz
from IPython.display import display, HTML

def init_genjax_penzai_display():
    # hide JAX paths
    display(HTML("<style>span.copybutton{display:none}</style>"))
    pz.ts.register_as_default(fancy_selections=False)

init_genjax_penzai_display()

#######################################################################
# %% A simple Gen function

import genjax
import jax.numpy as jnp

@genjax.static_gen_fn
def model(mu, coins):
    x = genjax.normal(mu, 1.0) @ "x"
    y = genjax.flip(jnp.sum(coins) / coins) @ "y"
    return x + y


key = jax.random.PRNGKey(314159)
tr = model.simulate(key, (-2.1, jnp.array([1, 1, 0])))

# %%
#######################################################################
# JSON handling

import json
import jax.numpy as jnp


def array_to_list(x):
    return x.tolist() if isinstance(x, jnp.ndarray) else x


def to_jsonable(x):
    if isinstance(x, jnp.ndarray):
        return x.tolist()
    elif isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return [to_jsonable(item) for item in x]
    else:
        return x


def to_json(x):
    return json.dumps(x)

#######################################################################
# Treescope handler registration

import penzai.treescope.default_renderer as default_renderer
import types


def prepend_global_handlers(*handlers):
    existing_handlers = default_renderer.active_renderer.get().handlers
    hnames = [handler.__name__ for handler in handlers]
    existing_handlers[:] = [
        *handlers,
        *[
            f
            for f in existing_handlers
            if not isinstance(f, types.FunctionType) or f.__name__ not in hnames
        ],
    ]


def remove_global_handlers(*handlers):
    existing_handlers = default_renderer.active_renderer.get().handlers
    hnames = [handler.__name__ for handler in handlers]
    existing_handlers[:] = [
        f
        for f in handlers
        if not isinstance(f, types.FunctionType) or f.__name__ not in hnames
    ]


#######################################################################
# Treescope UI utils

from jax.tree_util import GetAttrKey
import random
from penzai.treescope.foldable_representation import basic_parts
from penzai.treescope.foldable_representation import common_structures
from penzai.treescope.foldable_representation import common_styles
from penzai.treescope.foldable_representation import embedded_iframe

from pyobsplot import Plot


def to_node(x):
    # automatic coercion of strings to Text nodes
    if isinstance(x, str):
        x = basic_parts.Text(x)
    return x


def to_children(xs):
    # automatic coercion of nodes to lists of children
    if not isinstance(xs, (list, tuple)):
        xs = [xs]
    return [to_node(x) for x in xs]


def collapsed(label, children):
    return common_structures.build_custom_foldable_tree_node(
        contents=basic_parts.FoldCondition(
            collapsed=to_node(label),
            expanded=basic_parts.IndentedChildren.build(children=to_children(children)),
        )
    )

# %%
from penzai.treescope.foldable_representation import embedded_iframe

def iframe(html, fallback="# No text representation"):
    return embedded_iframe.EmbeddedIFrame(
        embedded_html=html,
        fallback_in_text_mode=common_styles.AbbreviationColor(
            basic_parts.Text(fallback)
        ),
    )

#######################################################################
# Observable Plot IFrame implementation

def make_plot_html(plot):
    return (
        f'<script id="plot-spec" type="application/json">{to_json(plot.spec)}</script>'
        + """
        <div id="plot"></div>
        <script type="module">
          import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm";
          import { generate_plot } from "https://cdn.jsdelivr.net/npm/pyobsplot@0.4.2/+esm";
          const spec_json = document.getElementById("plot-spec").innerHTML
          const plot = generate_plot(JSON.parse(spec_json));
          document.getElementById("plot").append(plot);
        </script>
    """
    )

def small_histogram(values):
    return Plot.plot(
        {
            "width": 200,
            "height": 100,
            "marks": [Plot.rectY(array_to_list(values), Plot.binX({"y": "count"}))],
        }
    )

from pyobsplot.widget import ObsplotWidget
def handle_obsplots(node, path, subtree_renderer):
    if isinstance(node, ObsplotWidget):
        return iframe(make_plot_html(node))
    return NotImplemented

from jax.tree_util import GetAttrKey
def handle_score(node, path, subtree_renderer):
    # TODO 
    # make a better decision about whether to generate a histogram of the score.
    # possibly put behind a button.
    if (
        path
        and path[-1] == GetAttrKey(name="score")
        and GetAttrKey(name="subtraces") not in path
    ):
        mock_values = jnp.array([random.randint(0, 99) for _ in range(1000)])
        if node.shape == () and 1 == 2:  # disabled to test the iframe
            return NotImplemented
        return iframe(make_plot_html(small_histogram(mock_values)))
    return NotImplemented


prepend_global_handlers(handle_score)
# remove_global_handlers(handle_obsplots)

tr

# %%
