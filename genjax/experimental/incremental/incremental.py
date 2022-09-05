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

from genjax.experimental.interpreters.propagating import (
    get_shaped_aval,
    Cell,
)


class Change(Cell):
    def __init__(self, val, aval):
        super().__init__(aval)
        self.val = val

    def __lt__(self, other):
        return self.bottom() and other.top()

    def top(self):
        return self.val is not None

    def bottom(self):
        return self.val is None

    def join(self, other):
        if other.bottom():
            return self
        else:
            return other

    @classmethod
    def new(cls, val):
        aval = get_shaped_aval(val)
        return Change(val, aval)

    @classmethod
    def unknown(cls, aval):
        return Change(None, aval)

    def flatten(self):
        return (self.val,), (self.aval,)

    @classmethod
    def unflatten(cls, data, xs):
        return Change(xs[0], data[0])
