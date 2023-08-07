# Importance sampling

This module exposes two variants of importance sampling, differing in their return signature.

::: genjax.inference.importance_sampling.BootstrapImportanceSampling
    options:
      members: 
      - new
      - apply

::: genjax.inference.importance_sampling.CustomProposalImportanceSampling
    options:
      members: 
      - new
      - apply

Sampling importance resampling runs importance sampling, and then resamples a single particle from the particle collection to return.

::: genjax.inference.importance_sampling.BootstrapSamplingImportanceResampling
    options:
      members: 
      - new
      - apply

::: genjax.inference.importance_sampling.CustomProposalSamplingImportanceResampling
    options:
      members: 
      - new
      - apply
