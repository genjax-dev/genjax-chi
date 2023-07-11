# Importance sampling

This module exposes two variants of importance sampling, differing in their return signature.

The first is `ImportanceSampling`.

::: genjax.inference.ImportanceSampling
    options:
      members: 
      - new
      - apply

Sampling importance resampling runs importance sampling, and then resamples a single particle from the particle collection to return.

::: genjax.inference.SamplingImportanceResampling
    options:
      members: 
      - new
      - apply
