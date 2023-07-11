# Distributions

::: genjax._src.generative_functions.distributions
    options:
      show_root_heading: true

!!! info 
    
    On this page, we document the abstract base classes which are used throughout the module. For submodules which implement distributions using the base classes (e.g. `scipy`, or `tfd`) - we list the available distributions.

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
