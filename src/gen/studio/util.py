from timeit import default_timer as timer
import requests
import re


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

def fetch_exports():
    """
    Used in dev to fetch exported names and types from Observable Plot
    """

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
                    name = name.split(" as ")[-1]  
                    if type not in exports:
                        exports[type] = []
                    exports[type].append(name)

    return exports