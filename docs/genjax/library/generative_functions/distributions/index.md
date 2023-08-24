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

Below, we list distribution generative function wrappers, all supported distributions, and their exported names.

### Distributions supported via `tfp.distributions`

To support [TensorFlow Probability distributions (`tfp.distributions`)](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions), `genjax` exposes a `TFPDistribution` wrapper class which relies on interfaces defined for `tfp.distributions` objects to implement the `genjax.ExactDensity` interface.

::: genjax.generative_functions.distributions.TFPDistribution

Below, we list all currently exported `TFPDistribution` instances.

::: genjax.generative_functions.distributions
    options:
      members:
        - tfp_bates
        - tfp_chi
        - tfp_chi2
        - tfp_geometric
        - tfp_gumbel
        - tfp_half_cauchy
        - tfp_half_normal
        - tfp_half_student_t
        - tfp_inverse_gamma
        - tfp_kumaraswamy
        - tfp_logit_normal
        - tfp_moyal
        - tfp_multinomial
        - tfp_negative_binomial
        - tfp_plackett_luce
        - tfp_power_spherical
        - tfp_skellam
        - tfp_student_t
        - tfp_normal
        - tfp_mv_normal_diag
        - tfp_mv_normal
        - tfp_categorical
        - tfp_truncated_cauchy
        - tfp_truncated_normal
        - tfp_uniform
        - tfp_von_mises
        - tfp_von_mises_fisher
        - tfp_weibull
        - tfp_zipf

### Custom distributions implemented in `genjax`
