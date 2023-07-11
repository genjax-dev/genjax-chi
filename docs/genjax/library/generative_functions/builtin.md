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

The `trace` primitive provides access to the ability to invoke another generative function as a callee. Returning to our example above:


```python exec="yes" source="tabbed-left" session="ex-trace"
import genjax
from genjax import beta 
from genjax import bernoulli 
from genjax import gen

@gen
def beta_bernoulli_process(u):
    # Invoking `trace` can be sweetened, or unsweetened.
    p = genjax.trace("p", beta)(0, u) # not sweet
    v = bernoulli(p) @ "v" # sweet
    return v
```

Now, programs written in the DSL which utilize `trace` have generative function interface method implementations which store callee choice data in the trace:

```python exec="yes" source="tabbed-left" session="ex-trace"
import jax
console = genjax.pretty()

key = jax.random.PRNGKey(314159)
key, tr = beta_bernoulli_process.simulate(key, (2, ))

print(console.render(tr))
```

Notice how the rendered result `Trace` has addresses in its choice trie for `"p"` and `"v"` - corresponding to the invocation of the beta and Bernoulli distribution generative functions.

The `trace` primitive is a critical element of structuring hierarchical generative computation in the builtin language.

### `cache`

The `cache` primitive is designed to expose a space vs. time trade-off for incremental computation in Gen's `update` interface.

## Generative datatypes

The builtin language implements a trie-like trace, choice map, and selection.

::: genjax.generative_functions.builtin.BuiltinTrace

::: genjax.generative_functions.builtin.BuiltinChoiceMap

::: genjax.generative_functions.builtin.BuiltinSelection

::: genjax.generative_functions.builtin.BuiltinComplementSelection
