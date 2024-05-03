# %%
from pyobsplot import Plot, js
import pyobsplot
from ipywidgets import GridBox, Layout
from timeit import default_timer as timer
from functools import partial
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
    The "$ANY" key is for accessing the `.inner` value of a trace.
    """
    result = tr
    for part in address:
        if part == "$ALL":
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


def apply_options(opts, plot):
    """
    Apply plot options to an existing plot.

    Args:
        opts (dict): A dictionary of plot options to apply.
        plot (pyobsplot.widget.ObsplotWidget or Any): The plot to apply options to.
            If it's an ObsplotWidget, the options will be merged with the existing plot spec.
            Otherwise, it will be returned as-is.

    Returns:
        pyobsplot.widget.ObsplotWidget (with options applied) or the passed-in value.
    """
    if isinstance(plot, pyobsplot.widget.ObsplotWidget):
        opts = plot_options[opts] if isinstance(opts, str) else opts
        return Plot.plot({**plot.spec["code"], **opts})
    return plot


def show(marks, **opts):
    return Plot.plot({**plot_options["default"], **opts, "marks": marks})

def scatter(xs, ys, **opts):
    """
    Create a scatter plot using the given x and y data points.

    Args:
        xs (array-like): The x-coordinates of the data points.
        ys (array-like): The y-coordinates of the data points.
        **opts: Additional options to pass to the Plot.dot mark.

    Returns:
        pyobsplot.widget.ObsplotWidget: The scatter plot widget.
    """
    opts = opts or {}
    label = opts.get('label', None)
    fill = js(f"()=>'{label}'") if label else opts.get('fill', None) or "black"
    
    return show(
        [
            Plot.frame({"stroke": "#ddd", "strokeWidth": 1}),
            Plot.dot(
                {"length": len(xs)},
                {
                    "x": ensure_list(xs),
                    "y": ensure_list(ys),
                    "fill": fill,
                    **opts,
                },
            ),
        ],
        color={'legend': True}
    )


def small_multiples(marksets, plot_opts={}, layout_opts={}):
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
        [apply_options(plot_opts, marks) for marks in marksets],
        layout=Layout(**layout_opts),
    )


# %%

from pyobsplot.widget import ObsplotWidget


def __add__(self, other):
    if isinstance(other, ObsplotWidget):
        combined_data = {**self.spec["code"], **other.spec["code"]}

        return ObsplotWidget(
            {
                **combined_data,
                "marks": self.spec["code"]["marks"] + other.spec["code"]["marks"],
            }
        )
    else:
        raise TypeError(
            f"Unsupported operand type(s) for +: 'ObsplotWidget' and '{type(other).__name__}'"
        )


# Monkey patch the __add__ method onto the ObsplotWidget class
# This way we can "add" plots together to overlay their marks.
ObsplotWidget.__add__ = __add__
ObsplotWidget.plot = lambda self, **opts: apply_options(opts, self)

def dot(values, **opts):
    return show([Plot.dot(values, opts)])
