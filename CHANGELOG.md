# Changelog

## [v0.2.0]

This release introduces a large set of new functionality (GenSP: https://github.com/probcomp/GenSP.jl)  into GenJAX, with a focus on inference infrastructure.

* A new inference core `genjax.inference.core` and two modules which build upon it (`genjax.vi` and `genjax.smc`) to expose first drafts of (composable) variational inference and sequential Monte Carlo inference in GenJAX.
* New types (`Target`, `Marginal`) which can be utilized to construct distributions _with stochastic probabilities_ - objects which are equipped with density _estimators_, instead of exact densities.
* A new implementation (`genjax.adev`) of the ADEV (https://arxiv.org/pdf/2212.06386.pdf) differentiable probabilistic language, for automating the construction of unbiased gradient estimators for expected value loss functions encoded as JAX compatible Python programs.

This PR includes a new set of basic usage tests for all this functionality. See the associated documentation pages on the new modules (`genjax.smc` and `genjax.vi` -- for more information on limitations). If something seems confusing, file an issue or send me a message!

**Important breaking changes:**
* Utility functions for working with `Pytree`, and `Diff` (relevant for `GenerativeFunction.update` invocations) have now been moved _under the class_ (e.g. `Diff.tree_diff_no_change` or `Diff.tree_diff_unknown_change`).

* (NOTE!) The previous SMC library has been removed in this release commit. Looking forward, we're planning to provide a fully fleshed out SMC library (including trace translation) built on top of the new concepts as part of the `0.4.0` update (this is 0.2.0). If you're currently using the old SMC library, PM me so we can talk about timeline for migration (and what to expect -- one idea we're playing around with is to expose the exact same behavior / API, just built on top of the new concepts).

## [v0.1.0]

- First numbered release! We began maintaining CHANGELOG entries from this point
  forward.
