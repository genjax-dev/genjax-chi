# Copyright 2022 Equinox maintainers & MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import dataclasses
import functools as ft
import types
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple

import jax
import jax._src.pretty_printer as pp
import jax.numpy as jnp
import numpy as np
from rich.tree import Tree


Dataclass = Any
PrettyPrintable = Any


@dataclasses.dataclass
class CustomPretty:
    @abc.abstractmethod
    def pformat_tree(self, **kwargs):
        pass


def simple_dtype(dtype) -> str:
    if isinstance(dtype, type):
        dtype = dtype(0).dtype
    dtype = dtype.name
    dtype = dtype.replace("complex", "c")
    dtype = dtype.replace("double", "d")
    dtype = dtype.replace("float", "f")
    dtype = dtype.replace("uint", "u")
    dtype = dtype.replace("int", "i")
    dtype = dtype.replace("key<fry>", "fry")
    return dtype


def _pformat_list(obj: List, **kwargs) -> Tree:
    tree = Tree(f"[b]{obj.__class__.__name__}[/b]")
    for v in obj:
        sub_tree = _pformat(v, **kwargs)
        tree.add(sub_tree)
    return tree


def _pformat_tuple(obj: Tuple, **kwargs) -> Tree:
    tree = Tree(f"[b]{obj.__class__.__name__}[/b]")
    for v in obj:
        sub_tree = _pformat(v, **kwargs)
        tree.add(sub_tree)
    return tree


def _dict_entry(key: PrettyPrintable, value: PrettyPrintable, **kwargs) -> Tree:
    tree = Tree(f"[b]{key}[/b]")
    sub_tree = _pformat(value, **kwargs)
    tree.add(sub_tree)
    return tree


def _pformat_dict(obj: Dict, **kwargs) -> Tree:
    tree = Tree(f"[b]{obj.__class__.__name__}[/b]")
    for (k, v) in obj.items():
        sub_tree = _dict_entry(k, v, **kwargs)
        tree.add(sub_tree)
    return tree


def _named_entry(name: str, value: Any, **kwargs) -> Tree:
    tree = Tree(f"{name}")
    sub_tree = _pformat(value, **kwargs)
    tree.add(sub_tree)
    return tree


def _pformat_namedtuple(obj: NamedTuple, **kwargs) -> Tree:
    tree = Tree(f"[b]{obj.__class__.__name__}[/b]")
    entries = [_named_entry(name, getattr(obj, name), **kwargs) for name in obj._fields]
    for entry in entries:
        tree.add(entry)
    return tree


def _pformat_dataclass(obj: Dataclass, **kwargs) -> Tree:
    tree = Tree(f"[b]{obj.__class__.__name__}[/b]")
    entries = [
        _named_entry(f.name, getattr(obj, f.name), **kwargs)
        for f in dataclasses.fields(obj)
        if f.repr
    ]
    for entry in entries:
        tree.add(entry)
    return tree


def _pformat_array(obj, **kwargs) -> Tree:
    short = kwargs["short_arrays"]
    try:
        if short:
            dtype_str = simple_dtype(obj.dtype)
            shape_str = ",".join(map(str, obj.shape))
            backend = "(numpy)" if isinstance(obj, np.ndarray) else ""
            return pp.text(f"{backend} {dtype_str}[{shape_str}]")
        else:
            return pp.text(repr(obj))
    except Exception:
        if hasattr(obj, "aval"):
            wrapped = pp.text(repr(obj))
            return pp.concat(
                [
                    wrapped,
                    pp.text("("),
                    _pformat_array(obj.aval, **kwargs),
                    pp.text(")"),
                ]
            )
        else:
            return pp.text(repr(obj))


def _pformat_function(obj: types.FunctionType, **kwargs) -> Tree:
    if kwargs.get("wrapped", False):
        fn = "wrapped function"
    else:
        fn = "function"
    return pp.text(f"<{fn} {obj.__name__}>")


def _pformat(obj: PrettyPrintable, **kwargs) -> Tree:
    follow_wrapped = kwargs["follow_wrapped"]
    truncate_leaf = kwargs["truncate_leaf"]
    if truncate_leaf(obj):
        tree = Tree(f"[b]{type(obj).__name__}(...)[/b]")
        return tree
    elif isinstance(obj, CustomPretty):
        return obj.pformat_tree(**kwargs)
    elif dataclasses.is_dataclass(obj):
        return _pformat_dataclass(obj, **kwargs)
    elif isinstance(obj, list):
        return _pformat_list(obj, **kwargs)
    elif isinstance(obj, dict):
        return _pformat_dict(obj, **kwargs)
    elif isinstance(obj, tuple):
        if hasattr(obj, "_fields"):
            return _pformat_namedtuple(obj, **kwargs)
        else:
            return _pformat_tuple(obj, **kwargs)
    elif isinstance(obj, np.ndarray) or isinstance(obj, jnp.ndarray):
        doc = _pformat_array(obj, **kwargs)
        return Tree(f"{doc.format()}")
    elif isinstance(obj, (jax.custom_jvp, jax.custom_vjp)):
        return _pformat(obj.__wrapped__, **kwargs)
    elif hasattr(obj, "__wrapped__") and follow_wrapped:
        kwargs["wrapped"] = True
        return _pformat(obj.__wrapped__, **kwargs)
    elif isinstance(obj, ft.partial) and follow_wrapped:
        kwargs["wrapped"] = True
        return _pformat(obj.func, **kwargs)
    elif isinstance(obj, types.FunctionType):
        doc = _pformat_function(obj, **kwargs)
        return Tree(doc.format())
    else:  # int, str, float, complex, bool, etc.
        tree = Tree(f"(const) {repr(obj)}")
        return tree


def _false(_):
    return False


def tree_pformat(
    pytree: PrettyPrintable,
    short_arrays: bool = True,
    follow_wrapped: bool = True,
    truncate_leaf: Callable[[PrettyPrintable], bool] = _false,
) -> str:
    """Pretty-formats a Pytree as a string, whilst abbreviating JAX arrays.

    (This is the function used in `__repr__` of [`equinox.Module`][].)

    All JAX arrays in the Pytree are condensed down to a short string representation
    of their dtype and shape.

    !!! example

        A 32-bit floating-point JAX array of shape `(3, 4)` is printed as `f32[3,4]`.

    **Arguments:**

    - `pytree`: The Pytree to pretty-format.
    - `short_arrays`: Toggles the abbreviation of JAX arrays.
    - `follow_wrapped`: Whether to unwrap `functools.partial` and `functools.wraps`.
    - `truncate_leaf`: A function `Any -> bool`. Applied to all nodes in the Pytree;
        all truthy nodes will be truncated to just `f"{type(node).__name__}(...)"`.

    **Returns:**

    A string.
    """

    return _pformat(
        pytree,
        short_arrays=short_arrays,
        follow_wrapped=follow_wrapped,
        truncate_leaf=truncate_leaf,
    )
