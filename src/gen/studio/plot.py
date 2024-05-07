# %%
import json

import pyobsplot
from ipywidgets import GridBox, Layout
from pyobsplot import Plot, js
import numpy as np
import jax.numpy as jnp
import re
from gen.studio.util import benchmark


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


def array_to_list(x):
    if isinstance(x, (jnp.ndarray, np.ndarray)):
        return x.tolist()
    return x


plot_options = {
    "small": {"width": 250, "height": 175, "inset": 10},
    "default": {"width": 500, "height": 350, "inset": 20},
}


def _merge_dicts_recursively(dict1, dict2):
    """
    Recursively merge two dictionaries.
    Values in dict2 overwrite values in dict1. If both values are dictionaries, recursively merge them.
    """
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return dict2
    for k in dict2:
        if k in dict1:
            dict1[k] = _merge_dicts_recursively(dict1[k], dict2[k])
        else:
            dict1[k] = dict2[k]
    return dict1

def _add_list(opts, marks, to_add):
    # mutates opts & marks, returns nothing
    for opt in to_add:
        if isinstance(opt, dict):
            _add_dict(opts, marks, opt)
        elif isinstance(opt, PlotSpec):
            _add_dict(opts, marks, opt.opts)
        elif isinstance(opt, list):
            _add_list(opts, marks, opt)


def _add_dict(opts, marks, to_add):
    # mutates opts & marks, returns nothing
    if "pyobsplot-type" in to_add:
        marks.append(to_add)
    else:
        opts = _merge_dicts_recursively(opts, to_add)
        new_marks = to_add.get("marks", None)
        if new_marks:
            _add_list(opts, marks, new_marks)


def _add(opts, marks, to_add):
    # mutates opts & marks, returns nothing
    if isinstance(to_add, list):
        _add_list(opts, marks, to_add)
    elif isinstance(to_add, dict):
        _add_dict(opts, marks, to_add)
    elif isinstance(to_add, PlotSpec):
        _add_dict(opts, marks, to_add.opts)
    else:
        raise TypeError(
            f"Unsupported operand type(s) for +: 'PlotSpec' and '{type(to_add).__name__}'"
        )


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
        new_marks = []
        self.opts = opts = {"marks": []}

        if marks is not None:
            _add_list(opts, new_marks, marks)
        if opts_dict is not None:
            _add_dict(opts, new_marks, opts_dict)
        if opts is not None:
            _add_dict(opts, new_marks, opts)
        opts["marks"] = new_marks
        self._plot = None

    def __add__(self, to_add):
        opts = self.opts.copy()
        marks = opts["marks"].copy()
        _add(opts, marks, to_add)
        opts["marks"] = marks
        return PlotSpec(opts_dict=opts)

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


def new(*specs):
    return PlotSpec(specs)


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


# %%
def small_multiples(plotspecs, plot_opts={}, layout_opts={}):
    # TODO
    # replace this with a pyobsplot-style js stub which
    # implements all the children in the same js context,
    # each widget has high overhead.
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


def fetch_exports():
    """
    Used in dev to fetch exported names and types from Observable Plot
    """
    import requests

    response = requests.get(
        "https://raw.githubusercontent.com/observablehq/plot/v0.6.14/src/index.js"
    )

    # Find all exported names
    export_lines = [
        line for line in response.text.split("\n") if line.startswith("export {")
    ]

    # Extract the names and types
    exports = {}
    for line in export_lines:
        names = line.split("{")[1].split("}")[0].split(", ")
        for name in names:
            if name[0].islower() and name not in ("plot", "marks"):
                match = re.search(r'from "\./(\w+)(?:/|\.js)?', line)
                if match:
                    type = match.group(1).rstrip("s")
                    name = name.split(" as ")[
                        -1
                    ]  # Handle cases like 'basic as transform'
                    if type not in exports:
                        exports[type] = []
                    exports[type].append(name)

    return exports


# fetch_mark_names()

_PLOT_EXPORTS = {
    "mark": [
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
    ],
    "option": ["valueof", "column", "identity", "indexOf"],
    "transform": [
        "filter",
        "reverse",
        "sort",
        "shuffle",
        "transform",
        "initializer",
        "bin",
        "binX",
        "binY",
        "centroid",
        "geoCentroid",
        "dodgeX",
        "dodgeY",
        "find",
        "group",
        "groupX",
        "groupY",
        "groupZ",
        "hexbin",
        "normalize",
        "normalizeX",
        "normalizeY",
        "map",
        "mapX",
        "mapY",
        "shiftX",
        "window",
        "windowX",
        "windowY",
        "select",
        "selectFirst",
        "selectLast",
        "selectMaxX",
        "selectMaxY",
        "selectMinX",
        "selectMinY",
        "stackX",
        "stackX1",
        "stackX2",
        "stackY",
        "stackY1",
        "stackY2",
        "treeNode",
        "treeLink",
    ],
    "interaction": ["pointer", "pointerX", "pointerY"],
    "format": ["formatIsoDate", "formatWeekday", "formatMonth"],
    "scale": ["scale"],
    "legend": ["legend"],
}


def _wrap_mark_fn(fn, fn_name):
    """
    Returns a wrapping function for an Observable.Plot mark, accepting a positional values argument
    (where applicable) options, which may be a single dict and/or keyword arguments.
    """

    def innerWithValues(values, opts={}, **kwargs):
        mark = fn(array_to_list(values), {**opts, **kwargs})
        return PlotSpec([mark])

    def innerWithoutValues(opts={}, **kwargs):
        mark = fn({**opts, **kwargs})
        return PlotSpec([mark])

    if fn_name in ["hexgrid", "grid", "gridX", "gridY", "gridFx", "gridFy", "frame"]:
        inner = innerWithoutValues
    else:
        inner = innerWithValues

    inner.__name__ = fn_name
    return inner


_plot_fns = {
    name: _wrap_mark_fn(getattr(Plot, name), name)
    if type == "mark"
    else getattr(Plot, name)
    for type, names in _PLOT_EXPORTS.items()
    for name in names
}

# Re-export the dynamically constructed MarkSpec functions
globals().update(_plot_fns)

# %%


def accept_xs_ys(plot_fn, default_opts=None):
    def inner(*args, **kwargs):
        if len(args) == 1:
            values = args[0]
        elif len(args) == 2:
            if isinstance(args[-1], dict):
                values, opts = args[0], args[1]
            else:
                xs, ys = args
        elif len(args) == 3:
            xs, ys, opts = args
        else:
            raise ValueError(f"Invalid number of arguments: {len(args)}")

        kwargs = (
            {**(default_opts or {}), **opts, **kwargs}
            if "opts" in locals()
            else {**(default_opts or {}), **kwargs}
        )

        if "values" in locals():
            return new(plot_fn(array_to_list(values), kwargs))
        else:
            return new(
                plot_fn(
                    {"length": len(xs)},
                    {"x": array_to_list(xs), "y": array_to_list(ys), **kwargs},
                )
            )

    return inner


scatter = accept_xs_ys(Plot.dot, {"fill": "currentColor"})
line = accept_xs_ys(Plot.line)


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
        fn = _plot_fns[fn_name]
        super().__init__([fn(default)])
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


frame = MarkDefault("frame", {"stroke": "#dddddd"})
"""Adds a frame, defaulting to a light gray stroke."""

ruleY = MarkDefault("ruleY", [0])
"""Adds a horizontal rule, defaulting to y=0."""

ruleX = MarkDefault("ruleX", [0])
"""Adds a vertical rule, defaulting to x=0."""


# The following convenience dicts can be added directly to PlotSpecs to declare additional behaviour.

grid_y = {"y": {"grid": True}}
grid_x = {"x": {"grid": True}}
grid = {"grid": True}
color_legend = {"color": {"legend": True}}
clip = {"clip": True}

def aspect_ratio(r):
    return {"aspectRatio": r}

def color_scheme(name):
    # See https://observablehq.com/plot/features/scales#color-scales
    return {"color": {"scheme": name}}

def domainX(d):
    return {"x": {"domain": d}}

def domainY(d):
    return {"y": {"domain": d}}

def domain(xd, yd=None):
    return {"x": {"domain": xd}, "y": {"domain": yd or xd}}

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


# %%
