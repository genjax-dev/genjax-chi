# %%
import anywidget

import ipywidgets as w
import traitlets

os.environ["ANYWIDGET_HMR"] = "1"  # hot code reloading for AnyWidget js files


# create an AnyWidget class by specifying its javascript source and traitlet assignments:
class ExampleWidget(anywidget.AnyWidget):
    _esm = "anywidget_reactivity.js"
    # the following assignments create "traitlets", little bits of reactive state which
    # are 2-way bound to "model" in the js environment.

    # faceSize will be set from Python using a slider.
    faceSize = traitlets.Int(default_value=100).tag(sync=True)

    # mousePos will be set from js using an `onmousemove` handler.
    mousePos = traitlets.Dict(default_value={"x": 0, "y": 0, "height": 0}).tag(
        sync=True
    )

    # happiness will be computed *in python* using `mousePos` values received from js
    happiness = traitlets.Float(default_value=1).tag(sync=True)


# instantiate our widget class
example = ExampleWidget()


# to set happiness, we write a function which receives a 'change' event and then
# call widget.observe(...)
def set_happiness(mousePosChange):
    example.happiness = (
        mousePosChange.new["height"] - mousePosChange.new["y"]
    ) / mousePosChange.new["height"]


example.observe(handler=set_happiness, names="mousePos")


def linkedWidget(ipywidget, anywidget, attr):
    # returns ipywidget bound to an anywidget instance's trait at name
    w.link((anywidget, attr), (ipywidget, "value"))
    return ipywidget


# after widget has loaded, send messages to it:
example.send({"foo": "bar"})

# for controlling update frequency as widgets change,
# see docs: https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Events.html

w.VBox(
    [
        # linkedWidget(w.FloatSlider(min=0, max=1, description="happiness"), example, "happiness"),
        linkedWidget(
            w.IntSlider(description="face size", max=400, min=30), example, "faceSize"
        ),
        example,
    ]
)
