# The cookbook
<p align="center">
<b>Recipes for when you've got modelling and inference to do, fast!</b>
</p>

This is a recipe book for _common patterns_. It also acts as a Rosetta stone of sorts for translating between programming constructs (like `if`, `for`, etc.) into the world of GenJAX (and JAX, more broadly).

## Your first model

```python exec="yes" html="true" source="material-block" session="cookbook"
import genjax

@genjax.static_gen_fn
def your_first_model():
  p = genjax.beta(1.0, 1.0) @ "p"
  f = genjax.flip(p) @ "f"
  return f

print(your_first_model().render_html())
```

Here's the first creature -- a `GenerativeFunction`! The powers of these creatures are numerous, and each power has a special name. The first name we will learn is `simulate`.

```python exec="yes" html="true" source="material-block" session="cookbook"
import jax

key = jax.random.PRNGKey(0)
print(your_first_model.simulate(key, ()).render_html())
```
