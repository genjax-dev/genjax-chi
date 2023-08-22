# Copyright 2022 MIT Probabilistic Computing Project
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


from genjax._src.core.datatypes.masking import Mask
from genjax._src.core.datatypes.tagged_unions import TaggedUnion
from genjax._src.core.typing import Bool
from genjax._src.core.typing import dispatch


#####
# Lifted type metaclass
#####


class LiftedTypeMeta(type):
    """The `LiftedTypeMeta` metaclass registers a class so that instance checks
    also work for the `Mask` and `TaggedUnion` types.

    This allows us to extend the class type lattice to include `Mask` instances which wrap around instances of the class, and similar for `TaggedUnion`.

    Using a metaclass here prevents us from having to implement versions of `Mask` and `TaggedUnion` for all of our types.
    """

    @dispatch
    def __instancecheck__(cls, inst: Mask) -> Bool:
        return isinstance(inst.value, cls)

    @dispatch
    def __instancecheck__(cls, inst: TaggedUnion):
        return all(map(lambda v: isinstance(v, cls), inst.values))

    @dispatch
    def __instancecheck__(cls, inst):
        return super().__instancecheck__(inst)
