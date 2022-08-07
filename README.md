# gax

> **G**en ⊗ [J**AX**](https://github.com/google/jax)

## Implementation strategy

The implementation strategy is based on: [Handling effects with JAX](https://colab.research.google.com/drive/1HGs59anVC2AOsmt7C4v8yD6v8gZSJGm6#scrollTo=OHUTBFIiHJu3) extend to support dynamically specified handlers.

Support a new `Core.Primitive` - `trace`, whose semantics are given by three interpreters: `simulate`, `generate` and `update`.

- When a `gax` function is interpreted with `simulate`, the semantics of `trace` is to sample from the distribution or (or recursively interpret a `gax` function) and accumulate the log probability density for the sampled [`ChoiceMap`](https://www.gen.dev/dev/ref/choice_maps/#Choice-Maps-1)
