# Combinators: structured patterns of composition

While the programmatic [`genjax.StaticGenerativeFunction`][] language is powerful, its restrictions can be limiting. Combinators are a way to express common patterns of composition in a more concise way, and to gain access to effects which are common in JAX (like `jax.vmap`) for generative computations.

Each of the combinators below is implemented as a method on [`genjax.GenerativeFunction`][] and as a standalone decorator.

You should strongly prefer the method form. Here's an example of the `vmap` combinator created by the [`genjax.GenerativeFunction.vmap`][] method:

Here is the `vmap` combinator used as a method. `square_many` below accepts an array and returns an array:

```python exec="yes" html="true" source="material-block" session="combinators"
import jax, genjax

@genjax.gen
def square(x):
    return x * x

square_many = square.vmap()
```

Here is `square_many` defined with [`genjax.vmap`][], the decorator version of the `vmap` method:

```python exec="yes" html="true" source="material-block" session="combinators"
@genjax.vmap()
@genjax.gen
def square_many_decorator(x):
    return x * x
```

!!! warning
    We do _not_ recommend this style, since the original building block generative function won't be available by itself. Please prefer using the combinator methods, or the transformation style shown below.

If you insist on using the decorator form, you can preserve the original function like this:

```python exec="yes" html="true" source="material-block" session="combinators"
@genjax.gen
def square(x):
    return x * x

# Use the decorator as a transformation:
square_many_better = genjax.vmap()(square)
```

## `vmap`-like Combinators

::: genjax.vmap
::: genjax.repeat

## `scan`-like Combinators

::: genjax.scan

## Control Flow Combinators

::: genjax.or_else
::: genjax.switch

## Various Transformations

::: genjax.map_addresses
::: genjax.dimap
::: genjax.map
::: genjax.contramap

## The Rest

::: genjax.mask
::: genjax.mix
