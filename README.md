<br>
<p align="center">
<img width="400px" src="./assets/logo.png"/>
</p>
<br>

![Build Status](https://github.com/probcomp/genjax/actions/workflows/ci.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A concise encoding of Gen using zero-cost effect handling/tracing built on top of `jax`.

- Presents a modeling language based on the space of pure Python functions acceptable by `jax`: models are pure functions from `(PRNGKey, *args)` to `(PRNGKey, retval)`.
- Exposes [the generative function interface](https://www.gen.dev/stable/ref/gfi/) as staged effect handlers built on top of `jax`. (Roughly -- see documentation for exact signatures/return types):
  - `simulate` (sample from normalized measure)
  - `generate` (condition the generative function, and importance sample with model as prior)
  - `arg_grad` (compute gradient of `logpdf` with respect to arguments)
  - `choice_grad` (compute gradient of `logpdf` with respect to values of random choices)
- Should support usage of any computations acceptable by JAX (tbd) within generative function programs.

<div align="center">
<b>(Early stage)</b> expect 🔪 sharp 🔪 edges 🔪
</div>

## Tour

[Jump into the tour!](/tour.py)

## Implementation strategy

The implementation strategy is based on: [Handling effects with JAX](https://colab.research.google.com/drive/1HGs59anVC2AOsmt7C4v8yD6v8gZSJGm6#scrollTo=OHUTBFIiHJu3) extended to support dynamically specified handlers. As in the original, handlers are staged out (zero-cost, not dynamically dispatched) by the compositional tracing provided by `jax`.

This implementation also takes inspiration from [Zero-cost Effect Handlers by Staging](http://ps.informatik.uni-tuebingen.de/publications/schuster19zero.pdf) - sans automatic insertion of staging annotations. The inference interfaces exposed handle lifting/splicing/jitting internally (manually).
