# Copyright 2024 MIT Probabilistic Computing Project
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

import jax
import jax.numpy as jnp
import msgpack
import numpy as np

from genjax._src.core.generative import GenerativeFunction, Trace
from genjax._src.core.serialization.backend import SerializationBackend
from genjax._src.core.typing import Tuple


class MsgPackSerializeBackend(SerializationBackend):
    """A class that supports serialization using the MsgPack protocol."""

    def serialize(self, trace: Trace):
        """Serialize an object using MsgPack

        The function strips out the Pytree definition from the generative trace via `tree_flatten` and converts the remaining data into a MsgPack representation.

        Args:
          trace: a Trace object

        Returns:
          msgpack-encoded bytes of the trace
        """
        data, _ = jax.tree_util.tree_flatten(trace)
        return msgpack.packb(data, default=_msgpack_ext_pack, strict_types=True)

    def deserialize(self, encoded_trace, gen_fn: GenerativeFunction, args: Tuple):
        """Deserialize an object using MsgPack

        The function decodes the MsgPack object and restructures the trace using its Pytree definition. The tree definition is retrieved by tracing the generative function using the stored arguments.

        Args:
          encoded_trace: msgpack-encoded bytes of the trace

          gen_fn: the generative function that produced `encoded_trace`

        Returns:
          `Trace` object
        """
        payload = msgpack.unpackb(encoded_trace, ext_hook=_msgpack_ext_unpack)
        trace_data_shape = gen_fn.get_trace_shape(*args)
        treedef = jax.tree_util.tree_structure(trace_data_shape)
        return jax.tree_util.tree_unflatten(treedef, payload)


msgpack_serialize = MsgPackSerializeBackend()

# The below helper functions have been adapted from google/flax [https://flax.readthedocs.io/en/latest/_modules/flax/serialization.html#msgpack_serialize].


def _ndarray_to_bytes(arr) -> bytes:
    """Save ndarray to simple msgpack encoding."""
    if isinstance(arr, jax.Array):
        arr = np.array(arr)
    if arr.dtype.hasobject or arr.dtype.isalignedstruct:
        raise ValueError(
            "Object and structured dtypes not supported "
            "for serialization of ndarrays."
        )
    tpl = (arr.shape, arr.dtype.name, arr.tobytes("C"))
    return msgpack.packb(tpl)


def _dtype_from_name(name: str):
    """Handle JAX bfloat16 dtype correctly."""
    if name == "bfloat16":
        return jax.numpy.bfloat16
    else:
        return np.dtype(name)


def _ndarray_from_bytes(data: bytes) -> np.ndarray:
    """Load ndarray from simple msgpack encoding."""
    shape, dtype_name, buffer = msgpack.unpackb(data)
    return jnp.asarray(
        np.frombuffer(
            buffer, dtype=_dtype_from_name(dtype_name), count=-1, offset=0
        ).reshape(shape, order="C")
    )


def _msgpack_ext_pack(obj):
    if isinstance(obj, (np.ndarray, jax.Array)):
        return msgpack.ExtType(1, _ndarray_to_bytes(obj))
    raise ValueError("Object type ", type(obj), " not supported.")


def _msgpack_ext_unpack(code, data):
    """Messagepack decoders for custom types."""
    if code == 1:
        return _ndarray_from_bytes(data)
    raise ValueError("Failed to deserialize data ", data)
