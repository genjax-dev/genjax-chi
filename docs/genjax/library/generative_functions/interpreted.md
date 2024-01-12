::: genjax._src.generative_functions.interpreted.fn
    options:
      show_root_heading: true

---
# The `interpreted` generative function language

::: genjax._src.generative_functions.interpreted
    options:
      show_root_heading: true

GenJAX's `interpreted` generative function language is a non-accelerated variant of GenJAX
suited for rapid prototyping and instructional uses. It cannot access the acceleration
features of JAX, and (therefore) cannot use automatic differentiation (AD) either, but it
can be used somewhat more flexibly.

In particular, you can use ordinary Python control flow in your generative functions.
In `@static` GenJAX, you would use combinators to handle forks in the road rather than
if statements. Further, `@static` requires an up front commitment to the shape of vectors
and tensors so that efficient use of acceleration hardware may be made.

In the interpreted language, there is no such restriction; the length of a vector
could itself be a random variable.

**What does it look like?**
Up front, here's a representative program, (with syntactic sugar, using the constructor as a decorator, both of which we'll cover in detail below):

```python exec="yes" source="tabbed-left" session="ex-trace"
import genjax
from genjax import beta, bernoulli, interpreted
console = genjax.console()

@interpreted
def beta_bernoulli_process(u):
    p = beta(0.0, u) @ "p"
    v = bernoulli(p) @ "v"
    return v

console.print(beta_bernoulli_process)
```

## Usage

The `interpreted` language is a common foundation for constructing models. It exposes a DSL based on JAX primitives and transformations which allows the programmer to construct generative functions out of Python functions.

Below, we illustrate a simple example:

```python
from genjax import beta
from genjax import bernoulli
from genjax import uniform
from genjax import interpreted

@interpreted
def beta_bernoulli_process(u):
    p = beta(0.0, u) @ "p"
    v = bernoulli(p) @ "v"
    return v

@interpreted
def joint():
    u = uniform() @ "u"
    v = beta_bernoulli_process(u) @ "bbp"
    return v
```

## Language primitives

The static language exposes custom primitives, which are handled by JAX interpreters to support the semantics of the generative function interface.

### `trace`

The `trace` primitive provides access to the ability to invoke another generative function as a callee.

::: genjax.generative_functions.interpreted

Returning to our example above:


```python exec="yes" source="tabbed-left" session="ex-trace"
import genjax
from genjax import beta
from genjax import bernoulli
from genjax import interpreted

@interpreted
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

console.render(tr)
```

Notice how the rendered result `Trace` has addresses in its choice trie for `"p"` and `"v"` - corresponding to the invocation of the beta and Bernoulli distribution generative functions.

The `trace` primitive is a critical element of structuring hierarchical generative computation in the interpreted language.

## Usage
