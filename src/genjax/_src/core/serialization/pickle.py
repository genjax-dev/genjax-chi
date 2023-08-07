# Copyright 2023 The oryx Authors and the MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

import dill as pickle

from genjax._src.core.pytree import Pytree
from genjax._src.core.serialization.backend import SerializationBackend
from genjax._src.core.typing import dispatch
from genjax._src.core.typing import typecheck
from genjax._src.core.typing import List
from genjax._src.core.typing import Any


@dataclass
class Pickle(SerializationBackend):
    @typecheck
    def _to_tuple(obj: Pytree):
        return (obj.__class__, obj.flatten())

    def dumps(self, obj: Pytree):
        """Serializes an object using pickle."""
        return pickle.dumps(self._to_tuple(obj))

    def loads(self, serialized_obj):
        """Deserializes an object using pickle."""
        cls, (xs, data) = pickle.loads(serialized_obj)
        return cls.unflatten(xs, data)

    def serialize(self, path, obj):
        pickle.dump(path, self._to_tuple(obj))


#####
# Mixin
#####

PickleDataFormat = List[Any]

# This should be implemented for traces.
class SupportsPickleSerialization:
    @dispatch
    def dumps(self, backend: Pickle) -> PickleDataFormat:
        backend.serialize(path, self)


# This should be implemented for generative functions.
class SupportsPickleDeserialization:
    @dispatch
    def loads(self, path, backend: Pickle):
        tr = backend.deserialize(path)
        return tr
