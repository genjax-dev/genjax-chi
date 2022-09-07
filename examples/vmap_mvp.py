import abc
import jax
from typing import NamedTuple
import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu
from dataclasses import dataclass


class Pytree(metaclass=abc.ABCMeta):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        jtu.register_pytree_node(
            cls,
            cls.flatten,
            cls.unflatten,
        )

    @abc.abstractmethod
    def flatten(self):
        pass

    @classmethod
    @abc.abstractmethod
    def unflatten(cls, data, xs):
        pass


# Note: NamedTuple is automatically registered as a PyTree
@dataclass
class Data(Pytree):
    x: np.ndarray
    y: int

    def flatten(self):
        return (self.x, self.y), ()

    @classmethod
    def unflatten(cls, xs, data):
        return Data(*xs, *data)


data = Data(np.arange(5), 2)


def f(x, data):
    print(data.x)
    assert isinstance(data.y, int)  # asserts that `data.y` is static/unchanged
    return data.x + data.y


out = jax.vmap(f, in_axes=[0, None])(jnp.ones(5), data)
print(out)
