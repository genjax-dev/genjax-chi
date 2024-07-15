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

from contextlib import contextmanager  # noqa: I001
from genjax._src.core.typing import Callable, typecheck

_GLOBAL_CHECKIFY_HANDLER = []


@contextmanager
def do_checkify():
    _GLOBAL_CHECKIFY_HANDLER.append(True)
    try:
        yield
    finally:
        _GLOBAL_CHECKIFY_HANDLER.pop()


@typecheck
def optional_check(check: Callable[[], None]):
    if _GLOBAL_CHECKIFY_HANDLER:
        check()
