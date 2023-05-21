::: genjax._src.generative_functions.builtin
    options:
      show_root_heading: false

The `Builtin` language is a common foundation for constructing models. It exposes a DSL based on JAX primitives and transformations which allows the programmer to construct generative functions out of Python functions. Below, we illustrate a simple example:
    
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

## Control flow within the DSL
