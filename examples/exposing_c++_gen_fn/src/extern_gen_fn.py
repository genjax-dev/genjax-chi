from jax.lib import xla_client
from . import extern_gen_fn

for _name, _value in extern_gen_fn.registrations().items():
    xla_client.register_cpu_custom_call_target(_name, _value)
