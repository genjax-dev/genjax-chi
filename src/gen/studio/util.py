from timeit import default_timer as timer
import requests
import re
import json
import os

from ipywidgets import HTML

import markdown
import anywidget 
import traitlets
import numpy as np 
import jax.numpy as jnp
import datetime

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



script_dir = os.path.dirname(os.path.abspath(__file__))
metadata_path = os.path.join(script_dir, "scripts/observable_plot_metadata.json")
OBSERVABLE_PLOT_METADATA = json.load(open(metadata_path))

def to_json(data, _widget):
    def default(obj):
        if hasattr(obj, "to_json"):
            return obj.to_json()
        if isinstance(obj, (jnp.ndarray, np.ndarray)):
            return obj.tolist()
        elif isinstance(obj, (datetime.date, datetime.datetime)):
            return {"pyobsplot-type": "datetime", "value": obj.isoformat()}
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    return json.dumps(data, default=default)
class Widget(anywidget.AnyWidget):
    _esm = "widget.js"
    data = traitlets.Any().tag(sync=True, to_json=to_json)

    def __init__(self, data):
        super().__init__(data=data)

    @anywidget.experimental.command
    def ping(self, msg, buffers):
        return "pong", None

def doc_str(functionName):
    return OBSERVABLE_PLOT_METADATA[functionName]['doc']
    
def doc(plot_fn):
    """
    Decorator to display the docstring of a plot function nicely formatted as Markdown.

    Args:
        plot_fn: The plot function whose docstring to display.

    Returns:
        An ipywidgets.HTML widget rendering the docstring as Markdown.
    """

    if plot_fn.__doc__:
        name = plot_fn.__name__
        doc = plot_fn.__doc__
        meta = OBSERVABLE_PLOT_METADATA.get(name, None)
        title = (
            f"<span style='font-size: 20px; padding-right: 10px;'>Plot.{name}</span>"
        )
        url = (
            f"https://observablehq.com/plot/{meta['kind']}/{re.search(r'([a-z]+)', name).group(1)}"
            if meta
            else None
        )
        return HTML(
            f"""
                    <div style="display: block; gap: 10px; border-bottom: 1px solid #ddd; padding: 10px 0;">
                    {title} 
                    <a style='color: #777; text-decoration: none;' href="{url}">Examples &#8599;</a></div>
                    """
            + markdown.markdown(doc)
        )
    else:
        return HTML("No docstring available.")