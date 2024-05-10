# The cookbook
<p align="center">
<b>Recipes for when you've got modelling and inference to do, fast!</b>
</p>

This is a recipe book for _common patterns_. It also acts as a Rosetta stone of sorts for translating between programming constructs (like `if`, `for`, etc.) into the world of GenJAX (and JAX, more broadly).

## A self-contained tour through combinators

The central programmable object of GenJAX is [_the generative function_](library/core.md#genjax.core.GenerativeFunction): these are the infamous _probabilistic programs_ which you write with Gen. Here's one of them now:

```python exec="yes" html="true" source="material-block" session="genfn"
import genjax
from genjax import flip, beta, gen
import jax
import jax.numpy as jnp

print(flip.render_html())
```

Admittedly, not too fabulous - _almost every language has a distributions library_, and most PPLs have a way to sample from distributions. But in GenJAX, when we call _something_ a generative function - we mean something stronger: it supports a compositional interface, implying that it can be used to build up _even larger_ probabilistic computations.

```python exec="yes" html="true" source="material-block" session="genfn"
@gen
def biased_coin_flipper():
    p = beta(1.0, 1.0) @ "p"
    f = flip(p) @ "f"

print(biased_coin_flipper.render_html())
```

Of course, why? Why would anyone _actually bother to write this_. Good point, my divisive and (yet) perceptive reader, what about 10 independent coin flips?

```python exec="yes" html="true" source="material-block" session="genfn"
# We can easily build up this probabilistic model in pieces.
print(biased_coin_flipper.repeat(num_repeats=10).render_html())
```

What about a mixture of two models where one of the models has a fixed `p` value for the coin flips, ..._and then we repeat that_? The mixture component is telling us _whether we're in a fair or biased coin flipping setting_...

```python exec="yes" html="true" source="material-block" session="genfn"
@gen
def fixed_coin_flipper(p):
    f = flip(p) @ "f"

print(biased_coin_flipper.mix(fixed_coin_flipper).repeat(num_repeats=10).render_html())
```

Hmm - what if we had a model where, each time the coin was flipped and it was heads, the probability _for the next flip_ went down a little bit?

```python exec="yes" html="true" source="material-block" session="genfn"
from genjax import scan_combinator

# scan(gen_fn) : ((a, b) -> G (a, c)) -> (a, [b]) -> G (a, [c])
@scan_combinator(max_length = 10) # (a, [b]) -> G (a, [c])
@gen # (a, b) -> G (a, c)
def v0_dependent_coin_flipper(carry, _):
    last_flip, p = carry
    p = p * (1 - 0.1 * last_flip)
    f = flip(p) @ "f"
    return (f, p), None

print(v0_dependent_coin_flipper.render_html())
```

Ah - what if we had a mixture of our first model, and _this model_. I think that would certainly cover all possible coin tossing rodeos we might encounter...

Oh, but first - I want to draw `p` in the dependent flipper from a `beta` too..

```python exec="yes" html="true" source="material-block" session="genfn"
@gen
def dependent_coin_flipper():
    p = beta(1.0, 1.0) @ "p"
    f = v0_dependent_coin_flipper.scan(max_length=10)((False, p), None) @ "f"

print(dependent_coin_flipper.render_html())
```

Okay, now obviously we want to mix everything together ... duh

```python exec="yes" html="true" source="material-block" session="genfn"
@gen
def dependent_coin_flipper():
    p = beta(1.0, 1.0) @ "p"
    f = v0_dependent_coin_flipper.scan(max_length=10)((0, p), None) @ "f"

mix1_logit = jnp.array([0.3])
mix2_logit = jnp.array([0.3])

our_final_model = dependent_coin_flipper.mix(
        biased_coin_flipper.mix(fixed_coin_flipper).repeat(num_repeats=10)
    )

print(our_final_model.render_html())
```

And now, we'll meet our first _generative function interface_: [`simulate`][genjax.core.GenerativeFunction.simulate]:

```python exec="yes" html="true" source="material-block" session="genfn"
from jax import jit
from jax.random import PRNGKey

key = PRNGKey(0)
# Fun, flirty, fast!
tr = jit(our_final_model.simulate)(key, (mix1_logit, (), (mix2_logit, (), p)))
print(tr.render_html())
```

The object `tr` returned by [`GenerativeFunction.simulate`][genjax.core.GenerativeFunction.simulate] is a [`Trace`][genjax.core.Trace] object, a representation of a sampling process provided by generative function. Traces are something that you will become intimately familiar with as you work with GenJAX - they are one of the key types that Gen uses to represent the results of probabilistic computations.
