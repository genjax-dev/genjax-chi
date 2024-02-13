# Inference

In probabilistic programming, inference refers to processes which involve computation using normalized conditional distributions. This occurs often, in varied contexts:

* I have a model of some phenomena, which includes several random variables. I have an observation of one random variable, and I'd like to _infer_ the probable distribution of the others. In other words, _how does knowledge of one thing inform my beliefs about the others_?

In GenJAX, the inference stack is based on an implementation of the language described in [Probabilistic Programming with Stochastic Probabilities](https://dl.acm.org/doi/abs/10.1145/3591290) (GenSP). The core concepts include data types which represent posterior inference problems (`Target`), inference algorithms (`InferenceAlgorithm`), and distribution objects which expose sampling and log density _estimation_. 

The concepts from GenSP allow GenJAX to support new modeling constructs like `Marginal`, which denotes marginalization over random variables. This object exposes sampling and density _estimation_ interfaces which cohere with Gen's existing estimator semantics.

## Code example

Up front, here's an example of the power of (a subset of) the inference automation.

## `genjax.inference.core` module reference

::: genjax._src.inference.core
    options:
      members:
        - Target
        - ChoiceDistribution
        - InferenceAlgorithm
        - Marginal
        - ValueMarginal
      show_root_heading: true

## Inference modules

GenJAX currently supports the following inference modules:

* [Sequential Monte Carlo](./smc.md), a class of inference algorithms which utilize collections of weighted samples to approximate posterior distributions, as well as estimates of their densities.

* [Variational inference](./vi.md), a class of inference algorithms which search over a family of approximating distributions using optimization of an objective function which measures closeness to the posterior.
