# %%
import importlib
import json
from functools import partial

import gen.studio.util as util
import gen.studio.js_modules as js_modules
importlib.reload(js_modules)
from gen.studio.js_modules import JSRef, hiccup, js
from gen.studio.util import Widget

# This module provides a composable way to create interactive plots using Observable Plot
# via pyobsplot (https://github.com/juba/pyobsplot) and AnyWidget (https://github.com/manzt/anywidget).
# See https://observablehq.com/plot/ for more on Observable Plot.
#
# Key features:
# - Create plot specifications declaratively by combining marks, options and transformations 
# - Compose plot specs using + operator to layer marks and merge options
# - Render specs to interactive plot widgets, with lazy evaluation and caching
# - Easily create grids of small multiples
# - Includes shortcuts for common options like grid lines, color legends, margins

d3 = JSRef("d3")
Math = JSRef("Math")
View = JSRef("View")

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


def _fn_wrapper(fn_name, meta):
    """
    Returns a wrapping function for an Observable.Plot mark, accepting a positional values argument
    (where applicable) options, which may be a single dict and/or keyword arguments.
    """
    kind = meta["kind"]
    if fn_name in ["hexgrid", "grid", "gridX", "gridY", "gridFx", "gridFy", "frame"]:
        # no values argument
        def inner(fn, spec={}, **kwargs):
            return PlotSpec(fn({**spec, **kwargs}))
    elif kind == "marks":
        # values argument
        def inner(fn, values, spec={}, **kwargs):
            return PlotSpec(fn(values, {**spec, **kwargs}))
    else:

        def inner(fn, *args, **kwargs):
            if kwargs:
                raise ValueError(
                    f"kwargs must not be passed to {fn_name}.{meta['module']} : {kwargs}"
                )
            return fn(*args)

    return inner


_plot_fns = {
    name: JSRef("Plot", name, inner=_fn_wrapper(name, meta), doc=meta["doc"])
    for name, meta in util.OBSERVABLE_PLOT_METADATA.items()
}

# Re-export the dynamically constructed MarkSpec functions
globals().update(_plot_fns)

#%%
plot_options = {
    "small": {"width": 250, "height": 175, "inset": 10},
    "default": {"width": 500, "height": 350, "inset": 20},
}


def _deep_merge(dict1, dict2):
    """
    Recursively merge two dictionaries.
    Values in dict2 overwrite values in dict1. If both values are dictionaries, recursively merge them.
    """
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return dict2
    for k in dict2:
        if k in dict1:
            dict1[k] = _deep_merge(dict1[k], dict2[k])
        else:
            dict1[k] = dict2[k]
    return dict1


def _add_list(spec, marks, to_add):
    # mutates spec & marks, returns nothing
    for new_spec in to_add:
        if isinstance(new_spec, dict):
            _add_dict(spec, marks, new_spec)
        elif isinstance(new_spec, PlotSpec):
            _add_dict(spec, marks, new_spec.spec)
        elif isinstance(new_spec, (list, tuple)):
            _add_list(spec, marks, new_spec)
        else:
            raise ValueError(f"Invalid plot specification: {new_spec}")


def _add_dict(spec, marks, to_add):
    # mutates spec & marks, returns nothing
    if "pyobsplot-type" in to_add:
        marks.append(to_add)
    else:
        spec = _deep_merge(spec, to_add)
        new_marks = to_add.get("marks", None)
        if new_marks:
            _add_list(spec, marks, new_marks)


def _add(spec, marks, to_add):
    # mutates spec & marks, returns nothing
    if isinstance(to_add, (list, tuple)):
        _add_list(spec, marks, to_add)
    elif isinstance(to_add, dict):
        _add_dict(spec, marks, to_add)
    elif isinstance(to_add, PlotSpec):
        _add_dict(spec, marks, to_add.spec)
    else:
        raise TypeError(
            f"Unsupported operand type(s) for +: 'PlotSpec' and '{type(to_add).__name__}'"
        )


class PlotSpec:
    """
    Represents a specification for a plot using pyobsplot.

    PlotSpecs can be composed using the + operator. When combined, marks accumulate
    and plot options are merged. Lists of marks or dicts of plot options can also be 
    added directly to a PlotSpec.

    IPython plot widgets are created lazily when the spec is viewed in a notebook,
    and then cached for efficiency.

    Args:
        *specs: PlotSpecs, lists of marks, or dicts of plot options to initialize with.
        **kwargs: Additional plot options passed as keyword arguments.
    """

    def __init__(self, *specs, **kwargs):
        marks = []
        self.spec = spec = {"marks": []}
        if specs:
            _add_list(spec, marks, specs)
        if kwargs:
            _add_dict(spec, marks, kwargs)
        spec["marks"] = marks
        self._plot = None

    def __add__(self, to_add):
        """
        Combine this PlotSpec with another PlotSpec, list of marks, or dict of options.

        Args:
            to_add: The PlotSpec, list of marks, or dict of options to add.

        Returns:
            A new PlotSpec with the combined marks and options.
        """
        spec = self.spec.copy()
        marks = spec["marks"].copy()
        _add(spec, marks, to_add)
        spec["marks"] = marks
        return PlotSpec(spec)

    def plot(self):
        """
        Generate the pyobsplot widget for this PlotSpec.

        The plot widget is created lazily on first call and cached for subsequent calls.

        Returns:
            The pyobsplot widget representing this plot.
        """
        if self._plot is None:
            self._plot = Widget(
                View.Plot(
                    {
                        **plot_options["default"],
                        **self.spec,
                    }
                )
            )
        return self._plot

    def reset(self, *specs, **kwargs):
        """
        Reset this PlotSpec's options and marks to those from the given specs.

        Reuses the existing plot widget.

        Args:
            *specs: PlotSpecs, lists of marks, or dicts of plot options to reset to.
            **kwargs: Additional options to reset.
        """
        self.spec = PlotSpec(*specs, **kwargs).spec
        self.plot().data = {**plot_options["default"], **self.spec}

    def update(self, *specs, marks=None, **kwargs):
        """
        Update this PlotSpec's options and marks in-place.

        Reuses the existing plot widget.

        Args:
            *specs: PlotSpecs, lists of marks, or dicts of plot options to update with.
            marks (list, optional): List of marks to replace existing marks with.
                If provided, overwrites rather than adds to existing marks.
            **kwargs: Additional options to update.
        """
        if specs:
            self.spec = (self + specs).spec
        if marks is not None:
            self.spec["marks"] = marks
        self.spec.update(kwargs)
        self.plot().data = self.spec

    def _repr_mimebundle_(self, include=None, exclude=None):
        return self.plot()._repr_mimebundle_()

    def to_json(self):
        return View.Plot({**plot_options["default"], **self.spec})


def new(*specs, **kwargs):
    """Create a new PlotSpec from the given specs and options."""
    return PlotSpec(specs, **kwargs)


def constantly(x):
    """
    Returns a javascript function which always returns `x`.

    Typically used to specify a constant property for all values passed to a mark,
    eg. plot.dot(values, fill=plot.constantly('My Label')). In this example, the
    fill color will be assigned (from a color scale) and show up in the color legend.
    """
    x = json.dumps(x)
    return js(f"()=>{x}")


def small_multiples(plotspecs, plot_opts={}, layout_opts={}):
    """
    Create a grid of small multiple plots from the given list of plot specifications.

    Args:
        plotspecs (list): A list of PlotSpecs to render as small multiples.
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

    return hiccup(
        [
            "div.grid.black",
            {
                "style": {
                    "display": "grid",
                    "grid-template-columns": "repeat(auto-fit, minmax(200px, 1fr))",
                }
            },
            *[(plotspec + plot_opts) for plotspec in plotspecs]
        ]
    )


def accept_xs_ys(plot_fn, default_spec=None):
    """
    Wraps a plot function to accept xs and ys arrays in addition to a values array.

    The wrapped function supports the following argument patterns:
    - values
    - values, spec
    - xs, ys
    - xs, ys, spec

    Where spec is a dictionary of plot options.
    """

    def inner(*args, **kwargs):
        if len(args) == 1:
            values = args[0]
        elif len(args) == 2:
            if isinstance(args[-1], dict):
                values, spec = args[0], args[1]
            else:
                xs, ys = args
        elif len(args) == 3:
            xs, ys, spec = args
        else:
            raise ValueError(f"Invalid number of arguments: {len(args)}")

        kwargs = (
            {**(default_spec or {}), **spec, **kwargs}
            if "spec" in locals()
            else {**(default_spec or {}), **kwargs}
        )

        if "values" in locals():
            return PlotSpec(plot_fn(values, kwargs))
        else:
            return PlotSpec(
                plot_fn(
                    {"length": len(xs)},
                    {"x": xs, "y": ys, **kwargs},
                )
            )

    inner.__doc__ = plot_fn.__doc__
    inner.__name__ = plot_fn.__name__
    inner._repr_mimebundle_ = plot_fn._repr_mimebundle_
    return inner


line = accept_xs_ys(_plot_fns["line"])
dot = accept_xs_ys(_plot_fns["dot"], {"fill": "currentColor"})


class PlotSpecWithDefault:
    """
    A class that wraps a mark function with defaults when called with no arguments.

    An instance of MarkDefault can be used directly as a PlotSpec or
    called as a function to customize the behaviour of the mark.

    Args:
        fn_name (str): The name of the mark function to wrap.
        default (dict): The default options for the mark.
    """

    def __init__(self, fn_name, *default_args):
        fn = _plot_fns[fn_name]
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.default_args = default_args
        self.fn = fn

    def __call__(self, *args, **kwargs):
        if not args and not kwargs:
            return new(self.fn(*self.default_args))
        return self.fn(*args, **kwargs)

    def _repr_mimebundle_(self, **kwargs):
        return self.fn._repr_mimebundle_(**kwargs)


frame = PlotSpecWithDefault("frame", {"stroke": "#dddddd"})
ruleY = PlotSpecWithDefault("ruleY", [0])
ruleX = PlotSpecWithDefault("ruleX", [0])

# The following convenience dicts can be added directly to PlotSpecs to declare additional behaviour.

grid_y = {"y": {"grid": True}}
grid_x = {"x": {"grid": True}}
grid = {"grid": True}
color_legend = {"color": {"legend": True}}
clip = {"clip": True}


def title(title):
    return {"title": title}


def subtitle(subtitle):
    return {"subtitle": subtitle}


def caption(caption):
    return {"caption": caption}


def width(width):
    return {"width": width}


def height(height):
    return {"height": height}


def size(size, height=None):
    return {"width": size, "height": height or size}


def aspect_ratio(r):
    return {"aspectRatio": r}


def inset(i):
    return {"inset": i}


def color_scheme(name):
    # See https://observablehq.com/plot/features/scales#color-scales
    return {"color": {"scheme": name}}


def domainX(d):
    return {"x": {"domain": d}}


def domainY(d):
    return {"y": {"domain": d}}


def domain(xd, yd=None):
    return {"x": {"domain": xd}, "y": {"domain": yd or xd}}


def color_map(mappings):
    return {"color": {"domain": mappings.keys(), "range": mappings.values()}}


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


# barX
# For reference - other options supported by plots
example_plot_options = {
    "title": "TITLE",
    "subtitle": "SUBTITLE",
    "caption": "CAPTION",
    "width": "100px",
    "height": "100px",
    "grid": True,
    "inset": 10,
    "aspectRatio": 1,
    "style": {"font-size": "100px"},  # css string also works
    "clip": True,
}
