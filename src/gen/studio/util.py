from timeit import default_timer as timer
import requests
import re
import json
import os

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
def doc(functionName):
    return OBSERVABLE_PLOT_METADATA[functionName]['doc']
    
# %%    
doc('area')    
# %%