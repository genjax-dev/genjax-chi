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

import functools
import inspect
from dataclasses import dataclass

from genjax._src.core.runtime_debugger import tag_with_name
from genjax._src.core.typing import Bool
from genjax._src.core.typing import Callable


@dataclass
class GenJAXGlobalOptions:
    checkify_flag: Bool = False
    runtime_debugger: Bool = False

    def allow_checkify(self, flag: Bool):
        self.checkify_flag = flag

    def allow_runtime_debugger(self, flag: Bool):
        self.runtime_debugger = flag

    def optional_check(self, check: Callable):
        if self.checkify_flag:
            check()

    def optional_runtime_debugger_capture(self, f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args):
            retval = f(*args)
            if self.runtime_debugger:
                file = inspect.getsourcefile(f)
                module = inspect.getmodule(f)
                tag_with_name(
                    {"args": args, "return": retval},
                    name=(module, file, f),
                )

            return retval

        return wrapper


global_options = GenJAXGlobalOptions()
