# %%
from pyobsplot import Plot, js
import pyobsplot
from ipywidgets import GridBox, Layout
from timeit import default_timer as timer
import json   


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

# def flatten(xs):
#     flattened = []
#     for x in xs:
#         if isinstance(x, list):
#             flattened.extend(flatten(x))
#         else:
#             flattened.append(x)
#     return flattened


class PlotSpec:
    def __init__(self, marks=None, **opts):
        self.opts = opts or {}
        self.opts['marks'] = marks or opts.get('marks', [])
        self._plot = None
        
    def __add__(self, opts):
        if isinstance(opts, list):
            opts = {**self.opts, 'marks': self.opts.get('marks') + opts}
            return PlotSpec(**opts)
        if isinstance(opts, dict):
            opts = {**self.opts, **opts, 'marks': self.opts.get('marks') + opts.get('marks', [])}
            return PlotSpec(**opts)
        if isinstance(opts, PlotSpec):
            return self + opts.opts 
        raise TypeError(
            f"Unsupported operand type(s) for +: 'PlotSpec' and '{type(opts).__name__}'"
        )
        
    def plot(self):
        if self._plot is None:
            self._plot = Plot.plot({**plot_options["default"], **self.opts,})
        return self._plot    
    def _repr_mimebundle_(self, include=None, exclude=None):
        return self.plot()._repr_mimebundle_()

def constantly(x):
    x = json.dumps(x)
    return pyobsplot.js(f"()=>{x}")

def scatter(xs, ys, **opts):
    """
    Wraps Plot.dot to accept lists of xs and ys as values
    """
    opts = opts or {}
    
    return PlotSpec(
        [
            Plot.dot(
                {"length": len(xs)},
                {
                    "x": ensure_list(xs),
                    "y": ensure_list(ys),
                    **opts,
                },
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

def dot(values, **opts):
    return PlotSpec([Plot.dot(values, opts)])
