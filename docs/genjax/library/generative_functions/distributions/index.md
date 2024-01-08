# Distributions

::: genjax._src.generative_functions.distributions
    options:
      show_root_heading: true

## The `Distribution` abstract base class

::: genjax.generative_functions.distributions.Distribution
    options:
      show_root_heading: true
      members:
        - random_weighted
        - estimate_logpdf

## The `ExactDensity` abstract base class

If you are attempting to create a new `Distribution`, you'll likely want to inherit from `ExactDensity` - which assumes that you have access to an exact logpdf method (a more restrictive assumption than `Distribution`). This is most often the case: all of the standard distributions use `ExactDensity`.

::: genjax.generative_functions.distributions.ExactDensity
    options:
      show_root_heading: true
      members:
        - sample
        - logpdf

### Distributions supported via `tfp.distributions`

To support [TensorFlow Probability distributions (`tfp.distributions`)](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions), `genjax` exposes a `TFPDistribution` wrapper class which relies on interfaces defined for `tfp.distributions` objects to implement the `genjax.ExactDensity` interface.

::: genjax.generative_functions.distributions.TFPDistribution
    options:
      members: false
