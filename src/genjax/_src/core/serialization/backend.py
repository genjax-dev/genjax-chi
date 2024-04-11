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

import abc
from dataclasses import dataclass

from genjax._src.core.datatypes.generative import GenerativeFunction, Trace

"""This module contains a trace serialization interface that interacts with different backend implementations. Pickle (`pickle_serialize`) or MsgPack (`msgpack_serialize`) may be used. Note: A serialized trace using a given backend must use the same backend to deserialize.
"""


@dataclass
class SerializationBackend:
    """
    This class supports serialization methods and provides pickle-like functions for convenience.

    Children of this class must override `serialize` which must output a raw byte representation of the trace. Similarly `deserialize` must take in the raw bytes along with the generative function to produce a trace. The *file* argument in `loads` and `dumps` must be a binary file descriptor which supports `read()` and `write()`.
    """

    @abc.abstractmethod
    def serialize(self, tr: Trace):
        pass

    @abc.abstractmethod
    def deserialize(self, bytes, gen_fn: GenerativeFunction):
        pass

    def dumps(self, tr: Trace):
        return self.serialize(tr)

    def loads(self, bytes, gen_fn: GenerativeFunction):
        return self.deserialize(bytes, gen_fn)

    def dump(self, tr: Trace, file):
        file.write(self.dumps(tr))

    def load(self, file, gen_fn: GenerativeFunction):
        bytes = file.read()
        return self.loads(bytes, gen_fn)
