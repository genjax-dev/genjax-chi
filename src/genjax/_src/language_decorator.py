# Copyright 2023 MIT Probabilistic Computing Project
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
from genjax._src.core.datatypes.generative import LanguageConstructor
from genjax._src.core.typing import Any
from genjax._src.core.typing import Callable
from genjax._src.core.typing import typecheck


######################
# Language decorator #
######################


@typecheck
def lang(lang_constructor: LanguageConstructor, *args, **kwargs) -> Callable:
    @typecheck
    def _inner(inner: Any) -> GenerativeFunction:
        return lang_constructor(inner, *args, **kwargs)

    return _inner
