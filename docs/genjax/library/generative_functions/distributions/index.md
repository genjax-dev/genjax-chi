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

If you are attempting to create a new `Distribution`, you'll likely want to inherit from `ExactDensity` - which assumes that you have access to an exact logpdf method (a more restrictive assumption than `Distribution`). This is most often the case: all of the standard distributions (`scipy`, `tfd`) use `ExactDensity`.

::: genjax.generative_functions.distributions.ExactDensity
    options:
      show_root_heading: true
      members:
        - sample
        - logpdf

## Supported distributions

Below, we list all supported distributions, and their exported names.

::: genjax.generative_functions.distributions 
    options:
        members:
            - tfp_bates
