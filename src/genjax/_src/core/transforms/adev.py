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

from genjax._src.core.datatypes.generative import GenerativeFunction
from genjax._src.core.interpreters import primitives
from genjax._src.core.typing import typecheck


# Sample intrinsic.
sample_p = primitives.InitialStylePrimitive("sample")


def _abstract_gen_fn_call(gen_fn, *args):
    return gen_fn.__abstract_call__(*args)


def _sample(gen_fn, *args, **kwargs):
    return primitives.initial_style_bind(sample_p)(_abstract_gen_fn_call)(
        gen_fn, *args, **kwargs
    )


@typecheck
def sample(gen_fn: GenerativeFunction, **kwargs):
    assert isinstance(gen_fn, GenerativeFunction)
    return lambda *args: _sample(gen_fn, *args, **kwargs)
