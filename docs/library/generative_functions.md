# The menagerie of `GenerativeFunction`

Generative functions are probabilistic building blocks. They allow you to express complex probability distributions, and automate several operations on them. GenJAX exports a standard library of generative functions, and this page catalogues them and their usage.
## The venerable & reliable `Distribution`

To start, distributions are generative functions.

::: genjax.Distribution
    options:
        show_root_heading: true
        members:
          - random_weighted
          - estimate_logpdf

Distributions intentionally expose a permissive interface ([`random_weighted`](generative_functions.md#genjax.Distribution.random_weighted) and [`estimate_logpdf`](generative_functions.md#genjax.Distribution.estimate_logpdf) which doesn't assume _exact_ density evaluation. [`genjax.ExactDensity`](generative_functions.md#genjax.ExactDensity) is a more restrictive interface, which assumes exact density evaluation.

::: genjax.ExactDensity
    options:
        show_root_heading: true
        members:
          - random_weighted
          - estimate_logpdf

GenJAX exports a long list of exact density distributions, which uses the functionality of [`tfp.distributions`](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions). A list of these is shown below.

::: genjax.generative_functions.distributions
    options:
        show_root_heading: true
        summary:
          attributes: true

## `StaticGenerativeFunction`: a programmatic language

For any serious work, you'll want a way to combine generative functions together, mixing deterministic functions with sampling. `StaticGenerativeFunction` is a way to do that: it supports the use of a JAX compatible subset of Python to author generative functions. It also supports the ability _to invoke_ other generative functions: instances of this type (and any other type of generative function) can then be used in larger generative programs.

::: genjax.StaticGenerativeFunction
    options:
        show_root_heading: true
        members:
        - source
        - simulate
        - assess
        - update

## Combinators: structured patterns of composition

While the programmatic `StaticGenerativeFunction` language is powerful, its restrictions can be limiting. Combinators are a way to express common patterns of composition in a more concise way, and to gain access to effects which are common in JAX (like `jax.vmap`) for generative computations.

Each of the combinators below is implemented as a decorator. `GenerativeFunction` instances make each combinator available as a method with the same name.

::: genjax.vmap
::: genjax.repeat
::: genjax.scan
::: genjax.mask
::: genjax.or_else
::: genjax.switch
::: genjax.mix
::: genjax.dimap
::: genjax.map
::: genjax.contramap
