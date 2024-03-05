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

"""This module contains a trace serialization interface that interacts with different backend implementations. Pickle or MsgPack may be used as a backend."""


@dataclass
class SerializationBackend:
    @abc.abstractmethod
    def serialize(self, tr: Trace):
        """Serialize"""
        pass

    @abc.abstractmethod
    def deserialize(self, bytes, gen_fn: GenerativeFunction):
        """Deserialize"""
        pass

    def dumps(self, tr: Trace):
        return self.serialize(tr)

    def loads(self, bytes, gen_fn: GenerativeFunction):
        return self.deserialize(bytes, gen_fn)

    def dump(self, tr: Trace, file: str):
        with open(file, "wb") as f:
            f.write(self.dumps(tr))

    def load(self, file: str, gen_fn: GenerativeFunction):
        with open(file, "rb") as f:
            bytes = f.read()
        return self.loads(bytes, gen_fn)
