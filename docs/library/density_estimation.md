# The language of inference

::: genjax.inference.Target
    options:
        show_root_heading: true

Algorithms inherit from a class called [`SampleDistribution`][genjax.inference.SampleDistribution] - these are objects which implement the _stochastic probability interface_ [[Lew23](https://dl.acm.org/doi/abs/10.1145/3591290)], meaning they expose methods to produce samples and samples from _density estimators_ for density computations.

::: genjax.inference.SampleDistribution
    options:
        show_root_heading: true
        members:
          - random_weighted
          - estimate_logpdf

`Algorithm` families implement the stochastic probability interface. Their [`Distribution`][genjax.Distribution] methods accept `Target` instances, and produce samples and density estimates for approximate posteriors.

::: genjax.inference.Algorithm
    options:
        show_root_heading: true
        members:
          - random_weighted
          - estimate_logpdf

By virtue of the _stochastic probability interface_, GenJAX also exposes _marginalization_ as a first class concept.

::: genjax.inference.Marginal
    options:
        show_root_heading: true
        members:
          - random_weighted
          - estimate_logpdf
