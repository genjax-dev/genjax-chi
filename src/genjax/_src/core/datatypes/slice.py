import jax.numpy as jnp

import genjax._src.generative_functions.combinators.vector.vector_datatypes as vec
from genjax._src.core.datatypes.choice import Choice, ChoiceValue, HierarchicalChoiceMap
from genjax._src.core.datatypes.selection import TraceSlice
from genjax._src.core.datatypes.trie import Trie
from genjax._src.core.typing import (
    Any,
)


class SliceCompiler:
    def __init__(self):
        pass

    def slice_to_choice(self, s: TraceSlice, value: Any) -> Choice:
        """Convert the TraceSlice to the equivalent Choice which will map the
        segment of the trace implied by slice onto the given value."""

        v = ChoiceValue(value)
        for s0 in reversed(s.s):
            if isinstance(s0, int):
                v = vec.IndexedChoiceMap(jnp.array([s0]), v)
            elif isinstance(s0, jnp.ndarray):
                v = vec.IndexedChoiceMap(s0, v)
            elif isinstance(s0, slice):
                raise NotImplementedError
                # v = IndexedChoiceMap(s0.indices, v)
            elif isinstance(s0, str):
                v = HierarchicalChoiceMap(Trie({s0: v}))
            else:
                raise NotImplementedError

        return v
