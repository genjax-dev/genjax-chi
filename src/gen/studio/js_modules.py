# %%
import gen.studio.util as util
from functools import partial


class JSFunctionCall(dict):
    """Represents a JavaScript function call."""
    def __init__(self, module, name, args):
        super().__init__(
            {"pyobsplot-type": "function", "module": module, "name": name, "args": args}
        )

    def _repr_mimebundle_(self, **kwargs):
        """Renders the JavaScript function call result as an interactive widget."""
        return util.Widget(self)._repr_mimebundle_(**kwargs)


def js_call(module, name, *args):
    """Represents a JavaScript function call."""
    return JSFunctionCall(module, name, args)


def js_ref(module, name):
    """Represents a reference to a JavaScript module or name."""
    return {"pyobsplot-type": "ref", "module": module, "name": name}


def js(txt: str) -> dict:
    """Represents raw JavaScript code to be evaluated."""
    return {"pyobsplot-type": "js", "value": txt}


class JSRef(dict):
    """Refers to a JavaScript module or name. When called, returns a function call representation."""
    def __init__(self, module, name=None, inner=lambda fn, *args: fn(*args), doc=None):
        self.__name__ = name
        self.__doc__ = doc
        self.inner = inner
        super().__init__(js_ref(module, name))
    def _repr_mimebundle_(self, **kwargs):
        return util.doc(self)._repr_mimebundle_(**kwargs)

    def __call__(self, *args, **kwargs):
        """Invokes the wrapped JavaScript function in the runtime with the provided arguments."""
        return self.inner(
            partial(js_call, self["module"], self["name"]), *args, **kwargs
        )

    def __getattr__(self, name):
        """Returns a reference to a nested property or method of the JavaScript object."""
        if name[0] == '_':
            return super().__getattribute__(name)
        elif self["name"] is None:
            return JSRef(self["module"], name)
        else:
            raise ValueError("Only module.name paths are currently supported")
            # return JSRef(f"{self['module']}.{self['name']}", name)

class Hiccup(list):
    """Wraps a Hiccup-style list to be rendered as an interactive widget in the JavaScript runtime."""
    def __init__(self, contents):
        super().__init__(contents)

    def _repr_mimebundle_(self, **kwargs):
        """Renders the Hiccup list as an interactive widget in the JavaScript runtime."""
        return util.Widget(self)._repr_mimebundle_(**kwargs)


def hiccup(x):
    """Constructs a Hiccup object from the provided list to be rendered in the JavaScript runtime."""
    return Hiccup(x)


# %%
