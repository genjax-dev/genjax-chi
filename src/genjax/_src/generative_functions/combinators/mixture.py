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


from genjax._src.core.generative import GenerativeFunction
from genjax._src.core.typing import TypeVar
from genjax._src.generative_functions.combinators.switch import (
    switch,
)
from genjax._src.generative_functions.distributions.tensorflow_probability import (
    categorical,
)
from genjax._src.generative_functions.static import gen

R = TypeVar("R")


def mix(*gen_fns: GenerativeFunction[R]) -> GenerativeFunction[R]:
    """
    Creates a mixture model from a set of generative functions.

    This function takes multiple generative functions as input and returns a new generative function that represents a mixture model.

    The returned generative function takes the following arguments:

    - `mixture_logits`: Logits for the categorical distribution used to select a component.
    - `*args`: Argument tuples for each of the input generative functions

    and samples from one of the input generative functions based on draw from a categorical distribution defined by the provided mixture logits.

    Args:
        *gen_fns: Variable number of [`genjax.GenerativeFunction`][]s to be mixed.

    Returns:
        A new [`genjax.GenerativeFunction`][] representing the mixture model.

    Examples:
        ```python exec="yes" html="true" source="material-block" session="mix"
        import jax
        import genjax


        # Define component generative functions
        @genjax.gen
        def component1(x):
            return genjax.normal(x, 1.0) @ "y"


        @genjax.gen
        def component2(x):
            return genjax.normal(x, 2.0) @ "y"


        # Create mixture model
        mixture = genjax.mix(component1, component2)

        # Use the mixture model
        key = jax.random.PRNGKey(0)
        logits = jax.numpy.array([0.3, 0.7])  # Favors component2
        trace = mixture.simulate(key, (logits, (0.0,), (7.0,)))
        print(trace.render_html())
        ```
    """

    inner_combinator_closure = switch(*gen_fns)

    @gen
    def mixture_model(mixture_logits, *args) -> R:
        mix_idx = categorical(mixture_logits) @ "mixture_component"
        v = inner_combinator_closure(mix_idx, *args) @ "component_sample"
        return v

    return mixture_model
