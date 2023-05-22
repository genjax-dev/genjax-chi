::: genjax._src.generative_functions.builtin
    options:
      show_root_heading: false

## Usage

The `Builtin` language is a common foundation for constructing models. It exposes a DSL based on JAX primitives and transformations which allows the programmer to construct generative functions out of Python functions. 

Below, we illustrate a simple example:
    
```python
from genjax import beta 
from genjax import bernoulli 
from genjax import uniform 
from genjax import gen

@genjax.gen
def beta_bernoulli_process(u):
    p = beta(0, u) @ "p"
    v = bernoulli(p) @ "v"
    return v

@genjax.gen
def joint():
    u = uniform() @ "u"
    v = beta_bernoulli_process(u) @ "bbp"
    return v
```

## Language primitives

The builtin language exposes custom primitives, which are handled by JAX interpreters to support the semantics of the generative function interface.

### `trace`

The `trace` primitive provides access to the to invoke another generative function as a callee.

### `cache`

The `cache` primitive is designed to expose a space vs. time trade-off for incremental computation in Gen's `update` interface.

## Generative datatypes

The builtin language implements a trie-like trace, choice map, and selection.

::: genjax.generative_functions.builtin.BuiltinTrace
