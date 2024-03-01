import msgpack
import jax
import jax.numpy as jnp
import numpy as np
from genjax._src.core.serialization.backend import SerializationBackend
from genjax._src.generative_functions.combinators.staging_utils import get_trace_data_shape

class MsgPackSerializeBackend(SerializationBackend):
  def serialize(self, trace):
    data, treedef = jax.tree_util.tree_flatten(trace)
    arg_len = len(trace.args)
    return msgpack.packb([arg_len, data], default=_msgpack_ext_pack, strict_types = True)

  def deserialize(self, encoded_trace, gen_fn):
    key = jax.random.PRNGKey(0)
    arg_len, payload = msgpack.unpackb(encoded_trace, ext_hook=_msgpack_ext_unpack)
    args = tuple(payload[:arg_len]) # arg numbers
    treedef =  jax.tree_util.tree_structure(get_trace_data_shape(gen_fn, key, args))
    return jax.tree_util.tree_unflatten(treedef, payload)

msgpack_serialize = MsgPackSerializeBackend()

# TODO: CITE FLAX
# https://flax.readthedocs.io/en/latest/_modules/flax/serialization.html#msgpack_serialize

def _ndarray_to_bytes(arr) -> bytes:
  """Save ndarray to simple msgpack encoding."""
  if isinstance(arr, jax.Array):
    arr = np.array(arr)
  if arr.dtype.hasobject or arr.dtype.isalignedstruct:
    raise ValueError(
      'Object and structured dtypes not supported '
      'for serialization of ndarrays.'
    )
  tpl = (arr.shape, arr.dtype.name, arr.tobytes('C'))
  return msgpack.packb(tpl)

def _dtype_from_name(name: str):
  """Handle JAX bfloat16 dtype correctly."""
  if name == b'bfloat16':
    return jax.numpy.bfloat16
  else:
    return np.dtype(name)

def _ndarray_from_bytes(data: bytes) -> np.ndarray:
  """Load ndarray from simple msgpack encoding."""
  shape, dtype_name, buffer = msgpack.unpackb(data)
  return jnp.asarray(np.frombuffer(
    buffer, dtype=_dtype_from_name(dtype_name), count=-1, offset=0
  ).reshape(shape, order='C'))

def _msgpack_ext_pack(obj):
    if isinstance(obj, jax.Array):
        # Set 1 to enum
        return msgpack.ExtType(1, _ndarray_to_bytes(obj))
    print("Other type ", obj, type(obj))
    return obj

def _msgpack_ext_unpack(code, data):
    """Messagepack decoders for custom types."""
    if code == 1:
        return _ndarray_from_bytes(data)
    return data

