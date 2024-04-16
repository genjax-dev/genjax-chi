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
"""The `genjax.interpreted` language is a generative function language which exposes a
less restrictive set of program constructs, based on normal Python programs.

The intent of this language is pedagogical - one can use it to rapidly construct
models and prototype inference, but it is not intended to be used for performance
critical applications, for several reasons:

* Instances of `genjax.interpreted` generative functions *cannot* be invoked as callees within JAX generative function code, which prevents compositional usage (from above, within `JAXGenerativeFunction` instances).

* It does not feature gradient interfaces - supporting an ad hoc Python AD implementation is out of scope for the intended applications of GenJAX.
"""
