# Extension modules

Because the set of generative function types is extensible, languages which extend Gen's interfaces to new types of generative computation are a natural part of Gen's ecosystem.

## Lazy loading

Users who wish to implement extension modules whose functionality depends on external 3rd party packages have a few patterns at their disposal. A user can simply build off `genjax` in their own repo, implementing their own generative functions, and inference, etc - while relying on 3rd party dependencies as well. This is likely the most common pattern, and should be a happy path for extension.

Extension modules which are considered useful and worthy of trunk support (to be bundled and tested with the `genjax` system) can utilize a lazy loading system:

::: genjax.extras.LazyLoader
    options:
        members:
            - _load
            - __getattr__
            - __dir__

## Currently supported modules

Here, we document two current modules:

* An extension module which extends the generative function interface to Gaussian processes using functionality from [`tinygp`](https://tinygp.readthedocs.io/en/stable/).
* An extension module for inference which provides a compatibility layer between generative functions and [`blackjax`](https://github.com/blackjax-devs/blackjax) for state-of-the-art Hamiltonian Monte Carlo (HMC) and No-U-Turn sampling (NUTS) inference.
