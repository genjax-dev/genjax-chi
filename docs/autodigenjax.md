# Write you a (mini) GenJAX

```python exec="yes" source="material-block" session="ex-dida"
from abc import abstractmethod
from dataclasses import dataclass
import re
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from genjax import console

console = console()
tfd = tfp.distributions
```

This is a "build a miniature GenJAX" tutorial: we construct a small probabilistic programming system (featuring a subset of Gen's generative function interfaces) (1). Along the way, we explore exactly why Gen is designed the way that it is.
{ .annotate }

1. The implementation of the JAX interpreter language here _is a small scale version of the implementation for GenJAX's `static` language_. Therefore, it is useful to work through this tutorial to understand the implementation of that language.

## Distributions

Let's start by assuming we have access to distributions. Distributions are objects which support a `sample` and `logprob` interface. We'll write a `Distribution` class which features JAX compatible interfaces (e.g. `sample` will accept a `jax.random.PRNGKey`):

```python exec="yes" source="material-block" session="ex-dida"
@dataclass
class Distribution:
    @abstractmethod
    def sample(key: PRNGKey, *args):
        pass

    @abstractmethod
    def logprob(v, *args):
        pass
```

Distributions are nice and atomic, but to express inference problems, we need likelihoods - which will expose ways of linking distributions into more complex ones. We encounter a design question: how are we going to express likelihoods in our language?

Let's start by enforcing a certain perspective on distributions - we are going to think of distributions as denoting a type of the kind $\textbf{G} \ \tau \ R$ (where $\tau$ and $R$ are type parameters).

The $\textbf{G}$ here stands for "generative" - it's an indicator that a value which inhabits this type is a computation, with some sort of probabilistic semantics (we'll make this precise later). The type parameters $\tau$ and $R$ denote, respectively, the type of the support space of the value and the _type of the return value_.

The generative types of kind $\textbf{G} \ \tau \ R$ will be defined by two ingredients:

- A measure $\mathbb{P}$ over the space $\tau$
- A deterministic function $f$ (which we call _the return value function_) with signature $f : \tau \rightarrow R$

The notion of a return value function as part of the generative computation is new. Most characterizations of distributions just focus on the support type. Indeed, for distributions, $R \equiv \tau$ - when you sample from a distribution, the return value function is just the identity.

On the other hand, we'll see that by adding this new ingredient, we've given ourselves a degree of flexibility with respect to _composition_ of our types with one another.

---

_A key design insight in Gen: the return value type need not be the same as the support type for all type instances of the kind $\textbf{G} \ \tau \ R$_.

---

Let's define a normal distribution for use later:

```python exec="yes" source="material-block" session="ex-dida"
@dataclass
class _Normal(Distribution):
    def sample(key, mu, sigma):
        return tfd.Normal(loc = mu, scale=sigma).sample(seed = key)

    def logpdf(key, v, mu, sigma):
        return tfd.Normal(loc = mu, scale = sigma).logprob(v)

Normal = _Normal()
```

## The interface on types of kind $\textbf{G} \ \tau \ R$

Let's a step forward: we're going to build up new types of kind $\textbf{G} \ \tau \ R$ _by specifying the ingredients $\mathbb{P}$ and $f$_.

- $\mathbb{P}$ is the measure over the space denoted by $\tau$.
- $f$ is the "return value function" - a function from $\tau \rightarrow R$.

We're going to call our new objects "generative functions". I'm going to provide their interface directly now, and we'll see over a series of examples _why the interface methods have the signatures that they do_.

```python exec="yes" source="material-block" session="ex-dida"
@dataclass
class GenerativeFunction:
    def simulate(key: PRNGKey, args: Tuple) -> Tuple[Any, Any]:
        pass

    def assess(tau_value: Any, args: Tuple) -> Tuple[Any, Any]:
        pass
```

### Introducing products

What if we write a "product combinator" using these concepts?

```python exec="yes" source="material-block" session="ex-dida"
@dataclass
class ProductCombinator(GenerativeFunction):
    product_components: List[Distribution]

    def simulate(key, mixture_args):
        sub_keys = jax.random.split(key, len(self.product_components))
        samples = []
        for (component, args) in zip(self.product_components, mixture_args):
            s = component.sample(key, *args)
            samples.append(s)
        return tuple(samples)

    def assess(key, v, mixture_args):
        pass
```

The idea behind this combinator is that we want a type of object which accepts a sequence of $(\textbf{G} \ \tau_1 \ R_1, ..., \textbf{G} \ \tau_n \ R_n)$, and creates a new instance of type $\textbf{G} \ (\tau_1 ... \otimes \tau_2) \ (R_1 ... \otimes R_n)$.

### Introducing sequencing

While `ProductCombinator` is neat - we still haven't really used that return value degree of freedom. Let's try something interesting: we'll call it the `DoCombinator`.

```python exec="yes" source="material-block" session="ex-dida"
@dataclass
class DoCombinator(GenerativeFunction):
    first: GenerativeFunction
    and_then: Callable[[Any], GenerativeFunction]

    def simulate(key, args):
        (first_args, final_args) = args
        key, sub_key = jax.random.split(key)
        v = first.simulate(sub_key, first_args)
        next_gen_fn = self.and_then(v)
        final = next_gen_fn.simulate(key, final_args)
        return final

    def assess(choices, args):
        (first_args, final_args) = args
        (first_choices, final_choices) = choices
        first_score, first_retval = first.assess(first_choices, first_args)
        next_gen_fn = self.and_then(v)
        final_score, final_retval = next_gen_fn.assess(final_choices, final_args)
        return (final_score + first_score), final_retval
```

## Programmatic composition

Now we're going to introduce a powerful layer of abstraction for composing generative functions together. We're going to rely on JAX to support this abstraction layer - JAX will allow us to use _metaprogramming_ to implement the semantics that we've discussed above for _programs_ which denote values of type $\textbf{G} \ \tau \ R$.

### The `Jaxpr` representation

```python exec="yes" source="tabbed-left" result="ansi" session="ex-dida"
# A little numerical example.
def f(x, y):
    z = x * y
    z = z * z
    q = z * y
    p = q * q
    return p

jaxpr = jax.make_jaxpr(f)(3.0, 5.0)
print(jaxpr)
```

### `Jaxpr` interpreters

To gain access to `Jaxpr` instances, we're going to use a utility function called `stage`.

```python exec="yes" source="tabbed-left" result="ansi" session="ex-dida"
from genjax.core import stage

# Gives us access to a `Jaxpr`, as well as input and output
# Python dataclass information.
jaxpr, (args, in_tree, out_tree) = stage(f)(3.0, 5.0)
print(jaxpr)
```

The `stage` function is very similar to `jax.make_jaxpr` - just a bit smarter and useful for "initial style" interpreters.

```python exec="yes" source="tabbed-left" result="ansi" session="ex-dida"

```
