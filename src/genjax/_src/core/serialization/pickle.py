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
from dataclasses import dataclass
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import typecheck


@typecheck
def to_tuple(obj: Pytree):
    return (obj.__class__, obj.flatten())

@dataclass
class Pickleable(Pytree):
    def pickle_serialize(self):
        """Serializes an object using pickle."""
        return pickle.dumps(to_tuple(self))

    def pickle_dump(self, path):
        pickle.dump(path, to_tuple(self))

    @classmethod
    def pickle_deserialize(cls, serialized_obj):
        """Deserializes an object using pickle."""
        cls, (xs, data) = pickle.loads(serialized_obj)
        return cls.unflatten(xs, data)
