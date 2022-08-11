# Experimental interfaces for approximate densities

`genjax` assumes a few key language elements:

1. `genjax` models are pure functions from `(PRNGKey, *args) -> (PRNGKey, retval)`.
2. The interfaces `simulate` and `generate` lift pure functions via program transformations to:
   - `(PRNGKey, *args) -> Trace`
   - `(PRNGKey, ChoiceMap, *args) -> (weight, Trace)`

These interfaces (+ `update`, to be implemented), along with the gradient interfaces, are enough to support all of Gen's standard inference library.

The effect/primitive handling implementation introduces a new set of problems for composability when we seek to include approximate density components in models.

Let's say we seek to include:

```
fn = pseudomarginal(f, g, importance)
```

(roughly: let's pseudo-marginalize out a subset of addresses in `f`, using importance sampling, with `g` as a proposal)

What is the type signature of this function? To stay within the `genjax` DSL -- it must be `(PRNGKey, *args) -> (PRNGKey, retval)` (remember: inference information is carried by the `trace` primitives) -- however, `pseudomarginal(...)` will eliminate `trace` ... and with `trace` eliminated, the score/weight computation from calling `fn` will be lost in the `simulate` and `generate` code generation for the toplevel model.

To alleviate this problem, `pseudomarginal` returns a new distribution-like primitive (an `ApproximateDensity`)-- **it is not a JAX-like composable transformation on functions**. In other words, it takes `(Callable, Callable, Callable) -> genjax.ApproximateDensity`.

# `simulate` and `generate`

If we `simulate` from a `genjax.ApproximateDensity` object inside a model -- the semantics of `simulate` require that we splice in the choice map (as if we had traced through a program representing the approximate density) with marginalized addresses hidden. In addition, we must return the score of the sample under the approximate density -- no longer an exact value, but now an estimate based on an inference strategy which furnishes log likelihood estimates.

Let's say the model is `P(z, y, x)` and we wish to pseudo-marginalize `z`. We provide a proposal `Q(z; y, x)` and choose importance sampling as our pseudo-marginalization strategy.

Then simulating from `pseudomarginal(P, Q, importance)` will sample a trace from `P`, then use importance sampling with proposal `Q` to furnish an estimate of `P(z)` (the estimate provided by the importance weight).

Now, the score should be an estimate of `P(y, x)` (because we're "hiding" `z`):

$$
P(y, x) = \int P(z, y, x) dz = \int \frac{P(z, y, x)}{Q(z; y, x)}Q(z; y, x) dz
$$
