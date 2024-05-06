# %%
from pyobsplot import Plot, js
import pyobsplot
from ipywidgets import GridBox, Layout
from timeit import default_timer as timer
import json

# Approach:
# - a thin, pythonic layer on top of Observable.Plot, building on pyobsplot.
#   See https://observablehq.com/plot/ and https://github.com/juba/pyobsplot


# %%
class benchmark(object):
    """
    A context manager for simple benchmarking.

    Usage:
        with benchmark("My benchmark"):
            # Code to be benchmarked
            ...

    Args:
        msg (str): The message to display with the benchmark result.
        fmt (str, optional): The format string for the time display. Defaults to "%0.3g".

    http://dabeaz.blogspot.com/2010/02/context-manager-for-timing-benchmarks.html
    """

    def __init__(self, msg, fmt="%0.3g"):
        self.msg = msg
        self.fmt = fmt

    def __enter__(self):
        self.start = timer()
        return self

    def __exit__(self, *args):
        t = timer() - self.start
        print(("%s : " + self.fmt + " seconds") % (self.msg, t))
        self.time = t


# model.simulate(key, data)
# traces = jax.vmap(lambda k: model.simulate(k, (data,)))(jax.random.split(key, 10))


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
    else:
        return x.tolist()


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
        self.opts = opts_dict or opts or {}
        self.opts.setdefault('marks', marks or [])
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

# Tests
def test_plotspec_init():
    ps = PlotSpec()
    assert ps.opts == {'marks': []}

    ps = PlotSpec(marks=[Plot.dot()])
    assert len(ps.opts['marks']) == 1
    assert 'pyobsplot-type' in ps.opts['marks'][0]

    ps = PlotSpec(width=100)
    assert ps.opts == {'marks': [], 'width': 100}

def test_plotspec_add():
    ps1 = PlotSpec(marks=[Plot.dot()], width=100)
    ps2 = PlotSpec(marks=[Plot.line()], height=200)
    
    ps3 = ps1 + ps2
    assert len(ps3.opts['marks']) == 2
    assert ps3.opts['width'] == 100
    assert ps3.opts['height'] == 200

    ps4 = ps1 + [Plot.text()]
    assert len(ps4.opts['marks']) == 2

    ps5 = ps1 + {'color': 'red'}
    assert ps5.opts['color'] == 'red'

    try:
        ps1 + 'invalid'
        assert False, "Expected TypeError"
    except TypeError:
        pass

def test_plotspec_plot():
    ps = PlotSpec(marks=[Plot.dot()], width=100) 
    plot = ps.plot()
    assert isinstance(plot, pyobsplot.Plot)
    assert plot.opts['width'] == 100

    # Check plot is cached
    plot2 = ps.plot()
    assert plot is plot2

def run_tests():
    test_plotspec_init()
    test_plotspec_add()
    test_plotspec_plot()
    print("All tests passed!")

# run_tests()
ps1 = PlotSpec(marks=[Plot.dot()], width=100)
ps2 = PlotSpec(marks=[Plot.line()], height=200)
    
ps3 = ps1 + ps2
ps3.opts
# %%

def constantly(x):
    x = json.dumps(x)
    return pyobsplot.js(f"()=>{x}")


def scatter(xs, ys, opts={}, **kwargs):
    """
    Wraps Plot.dot to accept lists of xs and ys as values
    """

    return PlotSpec(
        [
            Plot.dot(
                {"length": len(xs)},
                {"x": ensure_list(xs), "y": ensure_list(ys), **opts, **kwargs},
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
    import requests

    response = requests.get(
        "https://raw.githubusercontent.com/observablehq/plot/main/src/index.js"
    )
    index_js = response.text

    # Find all exported marks
    mark_lines = [
        line
        for line in index_js.split("\n")
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
    def inner(values, opts={}, **kwargs):
        mark = fn(values, {**opts, **kwargs})
        return PlotSpec([mark])

    inner.__name__ = fn_name
    return inner


_mark_spec_fns = {
    mark_name: _mark_spec_fn(getattr(Plot, mark_name), mark_name)
    for mark_name in _MARK_NAMES
}

# Re-export the dynamically constructed MarkSpec functions
globals().update(_mark_spec_fns)

# %%


class CallableList(list):
    """
    A list subclass that wraps a mark function and provides a default value.

    This class is used to create a default value for a specific mark type,
    while still allowing customization by calling the wrapped function.

    Args:
        fn (callable): The mark function to wrap.
        *args: Arguments to pass to the list constructor for the mark function.
    """

    def __init__(self, fn, *args):
        super().__init__(*args)
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


frame = CallableList(_mark_spec_fns["frame"], [Plot.frame({"stroke": "#dddddd"})])
"""Adds a frame, defaulting to a light gray stroke."""

ruleY = CallableList(_mark_spec_fns["ruleY"], [Plot.ruleY([0])])
"""Adds a horizontal rule, defaulting to y=0."""

ruleX = CallableList(_mark_spec_fns["ruleX"], [Plot.ruleX([0])])
"""Adds a vertical rule, defaulting to x=0."""

grid_y = {"y": {"grid": True}}
"""Enable grid lines for the y-axis."""

grid_x = {"x": {"grid": True}}
"""Enable grid lines for the x-axis."""

grid = {"grid": True}
"""Enable grid lines for both axes."""

color_legend = {"color": {"legend": True}}
"""Show a color legend."""

# line([[1, 2], [2, 4]]) + grid_x + grid_y + frame + ruleY + ruleX

# TODO
# maybe when passing a list [] the list should be able to contain
# MarkSpecs, dicts, or marks.
# opts = {**opts, 'marks': [*marks]}
# for i in other:
#   if MarkSpec, merge_BANG_(old_opts, i.opts)
#   if mark, opts['marks'].append(i)
#   if dict, merge_BANG_(old_opts, i)
# in this way, a + b + c is equivalent to a + [b, c]

frame


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

# %%
scatter([1, 2], [2, 1]).opts["marks"][0]
