# Importance sampling

This module exposes two variants of importance sampling: 

* bootstrap sampling, which uses the generative function's builtin proposal distribution family as a proposal
* custom proposals, which allows the programmer to provide other generative functions as proposals.

## Bootstrap sampling (using the prior)

::: genjax.inference.importance_sampling.BootstrapIS
    options:
      members: 
      - new
      - apply

::: genjax.inference.importance_sampling.BootstrapSIR
    options:
      members: 
      - new
      - apply

## Custom proposals

::: genjax.inference.importance_sampling.CustomProposalIS
    options:
      members: 
      - new
      - apply

Sampling importance resampling runs importance sampling, and then resamples a single particle from the particle collection to return.


::: genjax.inference.importance_sampling.CustomProposalSIR
    options:
      members: 
      - new
      - apply
