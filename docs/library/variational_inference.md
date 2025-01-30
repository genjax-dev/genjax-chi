# Variational inference with `genjax.vi`

Variational inference is an approach to inference which involves solving optimization problems over spaces of distributions. For a posterior inference problem, the goal is to find the distribution in some parametrized family of distributions (often called _the guide family_) which is close to the posterior under some notion of distance. Variational inference problems typically involve optimization functions which are defined as _expectations_, and these expectations and their analytic gradients are often intractable to compute. Therefore, unbiased gradient estimators are used to approximate the true gradients.

In GenJAX, our approach to variational inference is based on the approach presented in [this paper](https://dl.acm.org/doi/10.1145/3656463). The `genjax.vi` inference module provides automation for constructing variational losses, and deriving gradient estimators. The architecture is shown below.

<figure markdown="span">
  ![GenJAX VI architecture](../assets/img/genjax-vi.png){ width = "300" }
  <figcaption><b>Fig. 1</b>: How variational inference works in GenJAX.</figcaption>
</figure>

::: genjax.inference.vi.adev_distribution
    options:
        show_root_heading: true

::: genjax.inference.vi.ELBO
    options:
        show_root_heading: true

::: genjax.inference.vi.IWELBO
    options:
        show_root_heading: true

::: genjax.inference.vi.PWake
    options:
        show_root_heading: true

::: genjax.inference.vi.QWake
    options:
        show_root_heading: true
