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

from penzai import pz

"""Implementation of an autovisualizer, visualizing arrays."""


def pretty():
    pz.ts.register_as_default()
    pz.ts.register_autovisualize_magic()
    pz.enable_interactive_context()

    # Optional: enables automatic array visualization
    pz.ts.active_autovisualizer.set_interactive(pz.ts.ArrayAutovisualizer())


__all__ = [
    "pretty",
]
