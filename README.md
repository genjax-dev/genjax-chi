# genjax

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> [**Gen**](https://www.mct.dev/assets/mct-thesis.pdf) ⊗ [**JAX**](https://github.com/google/jax)

A concise encoding of Gen using zero-cost effect handling/tracing built on top of `jax`.

- Presents a modeling language based on the space of pure Python functions acceptable by `jax`: models are pure functions from `(PRNGKey, *args)` to `(PRNGKey, retval)`.
- Exposes [the generative function interface](https://www.gen.dev/stable/ref/gfi/) as staged effect handlers built on top of `jax`. (Roughly -- see documentation for exact signatures/return types):
  - `Simulate` (sample from normalized measure)
  - `Generate` (condition the generative function, and importance sample with model as prior)
  - `ArgumentGradients` (compute gradient of `logpdf` with respect to arguments)
  - `ChoiceGradients` (compute gradient of `logpdf` with respect to values of random choices)
- Should support usage of any computations acceptable by JAX (tbd) within generative function programs.

> **Early stage** expect 🔪 sharp edges 🔪

## Tour

[Jump into the tour!](/tour.py)

## Implementation strategy

The implementation strategy is based on: [Handling effects with JAX](https://colab.research.google.com/drive/1HGs59anVC2AOsmt7C4v8yD6v8gZSJGm6#scrollTo=OHUTBFIiHJu3) extended to support dynamically specified handlers. As in the original, handlers are staged out (zero-cost, not dynamically dispatched) by the compositional tracing provided by `jax`.

This implementation also takes inspiration from [Zero-cost Effect Handlers by Staging](http://ps.informatik.uni-tuebingen.de/publications/schuster19zero.pdf) - sans automatic insertion of staging annotations. The inference interfaces exposed handle lifting/splicing/jitting internally (manually).
