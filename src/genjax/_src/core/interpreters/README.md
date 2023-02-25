# GenJAX core interpreters

This directory contains several interpreters. These interpreters form the basis layer for functionality which relies on program transformations to implement semantics.

Several of these interpreters have been collected and modified from notebooks or other JAX-based systems. For each interpreter / transformation implementation, we've tried to keep a reference to the original attribution.

This README contains a brief description of each interpreter.

## Context

A final style forward interpreter which dispatches registered primitives to a dynamic context (which the user can inherit and override).

* (**harvest**) Oryx's `harvest` transformation.
* (**diff_rules**) Forward metadata propagation transformation.

## CPS

[Original adapted from notebook on static effect dispatch][effects_notebook]

A final style continuation passing interpreter which dispatches registered primitives to a dynamic context (which the user can inherit and override).

* (**adev**) [ADEV](https://arxiv.org/pdf/2212.06386.pdf)

## Propagate

[Original from Oryx][oryx_propagate]

An initial style interpreter which treats a `Jaxpr` as a graph, primitives as edges, and variables as nodes - and attempts to compute using fixpoint iteration on a lifted type lattice.

* (Supports **`coryx`**) A DSL for exact densities.

[effects_notebook]: https://colab.research.google.com/drive/1HGs59anVC2AOsmt7C4v8yD6v8gZSJGm6#scrollTo=ukjVJ2Ls_6Q3
[oryx_propagate]: https://github.com/jax-ml/oryx/blob/main/oryx/core/interpreters/propagate.py
[oryx_harvest]: https://github.com/jax-ml/oryx/blob/main/oryx/core/interpreters/harvest.py
