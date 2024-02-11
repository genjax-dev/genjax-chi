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

import functools

from oryx.core import plant, reap, sow

NAMESPACE = "state"


# "clobber" here means that parameters get shared across sites with
# the same name and tag.
def param(v, name):
    f = functools.partial(
        sow,
        tag=NAMESPACE,
        mode="clobber",
        name=name,
    )
    return f(v)


def pull(f):
    def _wrapped(*args):
        return reap(f, tag=NAMESPACE)(*args)

    return _wrapped


def push(f):
    def _wrapped(params, *args):
        return plant(f, tag=NAMESPACE)(params, *args)

    return _wrapped
