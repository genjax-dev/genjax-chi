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
"""This module provides compatibility extension plugins for packages which
provide functionality that is useful for modeling and inference.

Submodules present compatibility layers for usage of these packages with
GenJAX.
"""

import importlib
import importlib.util
import sys
import types


class LazyLoader(types.ModuleType):
    """> A lazy loading system which allows extension modules to optionally
    depend on 3rd party dependencies which may be too heavyweight to include as
    required dependencies for `genjax` proper.

    Examples:

        To utilize the system, the `LazyLoader` expects that you provide a local name for the module, globals, and the source module. Here's example usage for an extension module utilizing `tinygp` - we give the lazy loaded module the name `tinygp`, and tell the loader that the module path is `genjax._src.extras.tinygp`:

        ```python
        # tinygp provides Gaussian process model ingredients.
        tinygp = LazyLoader(
            "tinygp",
            globals(),
            "genjax._src.extras.tinygp",
        )
        ```

        The `tinygp` and `blackjax` extension modules rely on this system to implement functionality, while optionally depending on the presence of `tinygp` and `blackjax` (both 3rd party dependencies) for usage.

        ```python exec="yes" source="tabbed-left"
        import jax
        import genjax
        import tinygp.kernels as kernels
        console = genjax.pretty()

        # Extension module
        tinygp = genjax.extras.tinygp

        kernel_scaled = 4.5 * kernels.ExpSquared(scale=1.5)
        model = tinygp.GaussianProcess(kernel_scaled)

        print(console.render(model))
        ```
    """

    def __init__(self, local_name, parent_module_globals, name):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        super(LazyLoader, self).__init__(name)

    def _load(self):
        try:
            module = importlib.import_module(self.__name__)
            self._parent_module_globals[self._local_name] = module
            self.__dict__.update(module.__dict__)
            return module
        except ModuleNotFoundError as e:
            e.add_note(
                f"(GenJAX) Attempted to load {self._local_name} extension but failed, is it installed in your environment?"
            )
            raise e

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)


# BlackJAX provides HMC samplers.
blackjax = LazyLoader(
    "blackjax",
    globals(),
    "genjax._src.extras.blackjax",
)

# tinygp provides Gaussian process model ingredients.
tinygp = LazyLoader(
    "tinygp",
    globals(),
    "genjax._src.extras.tinygp",
)
