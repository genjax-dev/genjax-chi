# Inference

## Specifying inference problems from generative functions

::: genjax.inference.Target
    options:
        show_root_heading: true

Approximation solutions to inference problems are represented by `InferenceAlgorithm`:

::: genjax.inference.InferenceAlgorithm
    options:
        show_root_heading: true


## The SMC inference library

::: genjax.inference.smc.SMCAlgorithm
    options:
        show_root_heading: true

::: genjax.inference.smc.Importance
    options:
        show_root_heading: true

::: genjax.inference.smc.ImportanceK
    options:
        show_root_heading: true

## The VI inference library

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

## Encapsulation: attaching inference logic to generative functions

Gen makes it possible to attach inference logic as part of the internal proposal distribution family $Q(\cdot; u, a)$.
