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


import dill as pickle

from genjax._src.core.datatypes.generative import Trace
from genjax._src.core.pytree.pytree import Pytree
from genjax._src.core.serialization.backend import SerializationBackend
from genjax._src.core.typing import Any, dispatch


class PickleDataFormat(Pytree):
    payload: Any = Pytree.static()


class PickleSerializationBackend(SerializationBackend):
    def dumps(self, obj: Any):
        """Serializes an object using pickle."""
        return pickle.dumps(obj)

    def loads(self, serialized_obj) -> PickleDataFormat:
        """Deserializes an object using pickle."""
        return pickle.loads(serialized_obj)

    def serialize(self, path, obj: PickleDataFormat):
        pickle.dump(path, self._to_tuple(obj))


pickle_backend = PickleSerializationBackend()


#####
# Mixins
#####


# This should be implemented for traces.
class SupportsPickleSerialization:
    @dispatch
    def dumps(
        self,
        backend: PickleSerializationBackend,
    ) -> PickleDataFormat:
        raise NotImplementedError


# This should be implemented for generative functions.
class SupportsPickleDeserialization:
    @dispatch
    def loads(
        self,
        data: PickleDataFormat,
        backend: PickleSerializationBackend,
    ) -> Trace:
        raise NotImplementedError
