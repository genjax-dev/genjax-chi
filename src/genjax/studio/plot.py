# %%
import json

import pyobsplot
from ipywidgets import GridBox, Layout
from pyobsplot import Plot, js
import numpy as np 
import jax.numpy as jnp

# This module provides a convenient, composable way to create interactive plots using Observable Plot
# using pyobsplot, https://github.com/juba/pyobsplot and AnyWidget https://github.com/manzt/anywidget)
# See https://observablehq.com/plot/
#
# Key features:
# - Create plot specifications declaratively by combining marks, options and transformations 
# - Compose plot specs using + operator to layer marks and merge options
# - Render specs to interactive plot widgets, with lazy evaluation and caching
# - Easily create grids of small multiples 
# - Includes shortcuts for common options like grid lines, color legends, margins

def get_address(tr, address):
    """
    Retrieve a choice value from a trace using a list of keys.
    The "*" key is for accessing the `.inner` value of a trace.
    """
    result = tr
    for part in address:
        if part == "*":
            result = result.inner
        else:
            result = result[part]
    return result


def ensure_list(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, (jnp.ndarray, np.ndarray)):
        return x.tolist()
    else:
        return [x]


plot_options = {
    "small": {"width": 250, "height": 175, "inset": 10},
    "default": {"width": 500, "height": 350, "inset": 20},
}


class PlotSpec:
    """
    A class for specifying and composing plot options for Observable.Plot
    using pyobsplot. PlotSpecs can be composed using +; marks accumulate and
    plot options are merged. A list of marks or dict of plot options can also be added
    directly to a PlotSpec.

    IPython plot widgets are created lazily when the spec is viewed in a notebook,
    and then cached. PlotSpecs are cheap to create and combine.

    In addition to adding PlotSpecs, you can add a list of marks or a dict of plot options.
    """

    def __init__(self, marks=None, opts_dict=None, **opts):
        self.opts = {'marks': []}
        if marks is not None:
            self._handle_list(self.opts, self.opts['marks'], marks)
        if opts_dict is not None:
            self._handle_dict(self.opts, self.opts['marks'], opts_dict)
        if opts is not None:
            self._handle_dict(self.opts, self.opts['marks'], opts)        
        self._plot = None

    def _handle_list(self, new_opts, new_marks, opts_list):
        for opt in opts_list:
            if isinstance(opt, dict):
                self._handle_dict(new_opts, new_marks, opt)
            elif isinstance(opt, PlotSpec):
                self._handle_dict(new_opts, new_marks, opt.opts)
            elif isinstance(opt, list):
                self._handle_list(new_opts, new_marks, opt)

    def _handle_dict(self, opts, marks, opts_dict):
        if "pyobsplot-type" in opts_dict:
            marks.append(opts_dict)
        else:
            opts.update(opts_dict)
            new_marks = opts_dict.get("marks", None)
            if new_marks:
                self._handle_list(opts, marks, new_marks)

    def __add__(self, opts):
        new_opts = self.opts.copy()
        new_marks = new_opts["marks"].copy()
        if isinstance(opts, list):
            self._handle_list(new_opts, new_marks, opts)
        elif isinstance(opts, dict):
            self._handle_dict(new_opts, new_marks, opts)
        elif isinstance(opts, PlotSpec):
            self._handle_dict(new_opts, new_marks, opts.opts)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for +: 'PlotSpec' and '{type(opts).__name__}'"
            )
        new_opts["marks"] = new_marks    
        return PlotSpec(opts_dict=new_opts)    

    def plot(self):
        if self._plot is None:
            self._plot = Plot.plot(
                {
                    **plot_options["default"],
                    **self.opts,
                }
            )
        return self._plot

    def _repr_mimebundle_(self, include=None, exclude=None):
        return self.plot()._repr_mimebundle_()


# %%

def constantly(x):
    """
    Returns a javascript function which always returns `x` 
    
    Typically used to specify a constant property for all values passed to a mark, 
    eg. plot.dot(values, fill=plot.constantly('My Label')). In this example, the 
    fill color will be assigned (from a color scale) and show up in the color legend.
    """
    x = json.dumps(x)
    return pyobsplot.js(f"()=>{x}")


def scatter(xs, ys, opts={}, **kwargs):
    """
    Wraps Plot.dot to accept lists of xs and ys as values, and plot filled dots by default.
    """

    return PlotSpec(
        [
            Plot.dot(
                {"length": len(xs)},
                {"x": ensure_list(xs), "y": ensure_list(ys), "fill": "currentColor", **opts, **kwargs},
            ),
        ]
    )


# %%
def small_multiples(plotspecs, plot_opts={}, layout_opts={}):
    """
    Create a grid of small multiple plots from the given list of mark sets.

    Args:
        marksets (list): A list of plot mark sets to render as small multiples.
        plot_opts (dict, optional): Options to apply to each individual plot.
            Defaults to the 'small' preset if not provided.
        layout_opts (dict, optional): Options to pass to the Layout of the GridBox.

    Returns:
        ipywidgets.GridBox: A grid box containing the rendered small multiple plots.
    """
    plot_opts = {**plot_options["small"], **plot_opts}
    layout_opts = {
        "grid_template_columns": "repeat(auto-fit, minmax(200px, 1fr))",
        **layout_opts,
    }

    return GridBox(
        [(plotspec + plot_opts).plot() for plotspec in plotspecs],
        layout=Layout(**layout_opts),
    )


# %%
def fetch_mark_names():
    """
    Used in dev to fetch _MARK_NAMES
    """
    import requests

    response = requests.get(
        "https://raw.githubusercontent.com/observablehq/plot/v0.6.14/src/index.js"
    )

    # Find all exported marks
    mark_lines = [
        line
        for line in response.text.split("\n")
        if line.startswith("export {") and "./marks/" in line
    ]

    # Extract the mark names
    marks = []
    for line in mark_lines:
        mark_names = line.split("{")[1].split("}")[0].split(", ")
        lowercase_marks = [name for name in mark_names if name[0].islower()]
        marks.extend(lowercase_marks)

    return marks


# fetch_mark_names()

_MARK_NAMES = [
    "area",
    "areaX",
    "areaY",
    "arrow",
    "auto",
    "autoSpec",
    "axisX",
    "axisY",
    "axisFx",
    "axisFy",
    "gridX",
    "gridY",
    "gridFx",
    "gridFy",
    "barX",
    "barY",
    "bollinger",
    "bollingerX",
    "bollingerY",
    "boxX",
    "boxY",
    "cell",
    "cellX",
    "cellY",
    "contour",
    "crosshair",
    "crosshairX",
    "crosshairY",
    "delaunayLink",
    "delaunayMesh",
    "hull",
    "voronoi",
    "voronoiMesh",
    "density",
    "differenceY",
    "dot",
    "dotX",
    "dotY",
    "circle",
    "hexagon",
    "frame",
    "geo",
    "sphere",
    "graticule",
    "hexgrid",
    "image",
    "line",
    "lineX",
    "lineY",
    "linearRegressionX",
    "linearRegressionY",
    "link",
    "raster",
    "interpolateNone",
    "interpolatorBarycentric",
    "interpolateNearest",
    "interpolatorRandomWalk",
    "rect",
    "rectX",
    "rectY",
    "ruleX",
    "ruleY",
    "text",
    "textX",
    "textY",
    "tickX",
    "tickY",
    "tip",
    "tree",
    "cluster",
    "vector",
    "vectorX",
    "vectorY",
    "spike",
]


def _mark_spec_fn(fn, fn_name):
    """ 
    Returns a wrapping function for an Observable.Plot mark, accepting a positional values argument
    (where applicable) options, which may be a single dict and/or keyword arguments.
    """
    def innerWithValues(values, opts={}, **kwargs):
        mark = fn(ensure_list(values), {**opts, **kwargs})
        return PlotSpec([mark])
    def innerWithoutValues(opts={}, **kwargs):
        mark = fn({**opts, **kwargs})
        return PlotSpec([mark])
    
    if fn_name in ['hexgrid', 'grid', 'gridX', 'gridY', 'gridFx', 'gridFy', 'frame']:
        inner = innerWithoutValues 
    else:
        inner = innerWithValues

    inner.__name__ = fn_name
    return inner


_mark_spec_fns = {
    mark_name: _mark_spec_fn(getattr(Plot, mark_name), mark_name)
    for mark_name in _MARK_NAMES
}

# Re-export the dynamically constructed MarkSpec functions
globals().update(_mark_spec_fns)

# %%


class MarkDefault(PlotSpec):
    """
    A class that wraps a mark function and serves as a default value.
    
    An instance of MarkDefault can be used directly as a PlotSpec or 
    called as a function to customize the behaviour of the mark.

    Args:
        fn_name (str): The name of the mark function to wrap.
        default (dict): The default options for the mark.
    """

    def __init__(self, fn_name, default):
        fn = _mark_spec_fns[fn_name]
        super().__init__([fn(default)])
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

frame = MarkDefault("frame", {"stroke": "#dddddd"})
"""Adds a frame, defaulting to a light gray stroke."""

ruleY = MarkDefault("ruleY", {"y": 0})  
"""Adds a horizontal rule, defaulting to y=0."""

ruleX = MarkDefault("ruleX", {"x": 0})
"""Adds a vertical rule, defaulting to x=0."""

# The following convenience dicts can be added directly to PlotSpecs to declare additional behaviour.

grid_y = {"y": {"grid": True}}
"""Enable grid lines for the y-axis."""

grid_x = {"x": {"grid": True}}
"""Enable grid lines for the x-axis."""

grid = {"grid": True}
"""Enable grid lines for both axes."""

color_legend = {"color": {"legend": True}}
"""Show a color legend."""

# Example usage
# line([[1, 2], [2, 4]]) + grid_x + frame + ruleY + ruleX([1.2])

def margin(*args):
    """
    Set margin values for a plot using CSS-style margin shorthand.

    Supported arities:
        margin(all)
        margin(vertical, horizontal)
        margin(top, horizontal, bottom)
        margin(top, right, bottom, left)

    """
    if len(args) == 1:
        return {"margin": args[0]}
    elif len(args) == 2:
        return {
            "marginTop": args[0],
            "marginBottom": args[0],
            "marginLeft": args[1],
            "marginRight": args[1],
        }
    elif len(args) == 3:
        return {
            "marginTop": args[0],
            "marginLeft": args[1],
            "marginRight": args[1],
            "marginBottom": args[2],
        }
    elif len(args) == 4:
        return {
            "marginTop": args[0],
            "marginRight": args[1],
            "marginBottom": args[2],
            "marginLeft": args[3],
        }
    else:
        raise ValueError(f"Invalid number of arguments: {len(args)}")


# For reference - other options supported by plots
# example_plot_options = {
#     "title": "TITLE",
#     "subtitle": "SUBTITLE",
#     "caption": "CAPTION",
#     "width": "100px",
#     "height": "100px",
#     "grid": True,
#     "inset": 10,
#     "aspectRatio": 1,
#     "style": {"font-size": "100px"},  # css string also works
#     "clip": True,
# }
