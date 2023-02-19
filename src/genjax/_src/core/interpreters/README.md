# GenJAX core interpreters

This directory contains several interpreters, which form the basis of functionality relying on program transformations to implement semantics.

Several of these interpreters have been collected from notebooks or other JAX-based systems. For each interpreter / transformation implementation, we've tried to keep a reference to the original attribution.

This README contains a brief description of each interpreter.

## CPS

[Original adapted from notebook on static effect dispatch][effects_notebook]

An initial style continuation passing interpreter.

## Propagate

[Original from Oryx][oryx_propagate]

An initial style interpreter which treats a `Jaxpr` as a graph, primitives as edges, and variables as nodes - and attempts to compute using fixpoint iteration on a lifted type lattice.

## Harvest

[Original from Oryx][oryx_harvest]

A final style interpreter which supports sowing/reaping state inside of pure functions.

[effects_notebook]: https://colab.research.google.com/drive/1HGs59anVC2AOsmt7C4v8yD6v8gZSJGm6#scrollTo=ukjVJ2Ls_6Q3
[oryx_propagate]: https://github.com/jax-ml/oryx/blob/main/oryx/core/interpreters/propagate.py
[oryx_harvest]: https://github.com/jax-ml/oryx/blob/main/oryx/core/interpreters/harvest.py
