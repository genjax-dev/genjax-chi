import numpy as np
from jax import numpy as jnp
from jax.lib import xla_client
from jax import core, dtypes, lax
from jax.interpreters import ad, batching, xla
from jax.abstract_arrays import ShapedArray
from . import extern_gen_fn

for _name, _value in extern_gen_fn.registrations().items():
    xla_client.register_cpu_custom_call_target(_name, _value)
