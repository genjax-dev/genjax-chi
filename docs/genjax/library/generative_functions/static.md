# The `static` generative function language

::: genjax._src.generative_functions.static
    options:
      show_root_heading: true

GenJAX's `static` generative function language is a key workhorse language for expressing complex probabilistic computations. It implements the generative function interface _using `Jaxpr` interpreters_, which allows its modeling language to utilize Python programs as source language programs for generative functions.

**What does it look like?**
Up front, here's a representative program, (with syntactic sugar, using the constructor as a decorator, both of which we'll cover in detail below):

```python exec="yes" source="tabbed-left" session="ex-trace"
import genjax
from genjax import beta 
from genjax import bernoulli 
from genjax import Static

@Static
def beta_bernoulli_process(u):
    p = beta(0.0, u) @ "p"
    v = bernoulli(p) @ "v" # sweet
    return v

print(console.render(beta_bernoulli_process))
```

::: genjax._src.generative_functions.static.static_gen_fn.StaticGenerativeFunction
    options:
      show_root_heading: true

## Usage

The `Static` language is a common foundation for constructing models. It exposes a DSL based on JAX primitives and transformations which allows the programmer to construct generative functions out of Python functions.

Below, we illustrate a simple example:

```python
@Static
def beta_bernoulli_process(u):
    p = beta(0.0, u) @ "p"
    v = bernoulli(p) @ "v"
    return v

@Static
def joint():
    u = uniform() @ "u"
    v = beta_bernoulli_process(u) @ "bbp"
    return v
```

## Language primitives

The static language exposes custom primitives, which are handled by JAX interpreters to support the semantics of the generative function interface.

### `trace`

The `trace` primitive provides access to the ability to invoke another generative function as a callee.

::: genjax.generative_functions.static.trace

Returning to our example above:


```python exec="yes" source="tabbed-left" session="ex-trace"
import genjax
from genjax import beta 
from genjax import bernoulli 
from genjax import Static

@Static
def beta_bernoulli_process(u):
    # Invoking `trace` can be sweetened, or unsweetened.
    p = genjax.trace("p", beta)(0.0, u) # not sweet
    v = bernoulli(p) @ "v" # sweet
    return v
```

Now, programs written in the DSL which utilize `trace` have generative function interface method implementations which store callee choice data in the trace:

```python exec="yes" source="tabbed-left" session="ex-trace"
import jax
console = genjax.console()

key = jax.random.PRNGKey(314159)
tr = beta_bernoulli_process.simulate(key, (2.0, ))

print(console.render(tr))
```

Notice how the rendered result `Trace` has addresses in its choice trie for `"p"` and `"v"` - corresponding to the invocation of the beta and Bernoulli distribution generative functions.

The `trace` primitive is a critical element of structuring hierarchical generative computation in the static language.
