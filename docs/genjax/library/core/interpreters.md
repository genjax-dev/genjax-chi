JAX supports transformations of pure, numerical Python programs by staging out interpreters which evaluate [`Jaxpr`](https://jax.readthedocs.io/en/latest/jaxpr.html) representations of programs.

The `Core` module features interpreter infrastructure, and common transforms designed to facilitate certain types of transformations.

## Contextual interpreter

A common interpreter idiom in JAX involves overloading desired primitives with context-specific behavior by inheriting from `Trace` and define the correct methods to process the primitives.

`ContextualInterpreter` provides this idiom in GenJAX: this interprter mixes initial style (e.g. the Python program is immediately staged, and then an interpreter walks the `Jaxpr` representation) with custom `Trace` and `Tracer` overloads. 

This pattern supports a wide range of program transformations, and allows parametrization over the evaluation order of the inner interpreter (e.g. forward evaluation, or CPS).

!!! note "User interaction with `ContextualInterpreter`"

    Users are not expected to interact with this functionality, but we document its implementation for advanced users or those interested in implementing program transformations with JAX.

::: genjax._src.core.interpreters.context
    options:
      members: 
        - ContextualTracer
        - ContextualTrace

## Incremental computation

::: genjax._src.core.transforms.incremental
    options:
      show_root_heading: true
      members:
        - Diff
        - tree_diff_primal
        - tree_diff_tangent
        - tree_diff_no_change
        - tree_diff_unknown_change

## Stateful computation

!!! note "`harvest` from [oryx](https://github.com/jax-ml/oryx)"

    [The `harvest` transformation is from Oryx](https://www.tensorflow.org/probability/oryx/api_docs/python/oryx/core/interpreters/harvest). GenJAX supports an implementation which essentially matches the `Oryx` version, customized for a few higher-level tools.

::: genjax._src.core.transforms.harvest
    options:
      show_root_heading: true
      members:
        - sow
        - harvest


### Sharp edges

* `harvest` has undefined semantics under automatic differentiation. If a function
  you're taking the gradient of has a `sow`, it might produce unintuitive
  results when harvested. To better control gradient semantics, you can use
  `jax.custom_jvp` or `jax.custom_vjp`. The current implementation sows primals
  and tangents in the JVP but ignore cotangents in the VJP. These particular
  semantics are subject to change.
* Planting values into a `pmap` is partially working. Harvest tries to plant all
  the values, assuming they have a leading map dimension.

### Examples

#### Using `sow` and `harvest`

```python
def f(x):
  y = sow(x + 1., tag='intermediate', name='y')
  return y + 1.

# Injecting, or "planting" a value for `y`.
harvest(f, tag='intermediate')({'y': 0.}, 1.)  # ==> (1., {})
harvest(f, tag='intermediate')({'y': 0.}, 5.)  # ==> (1., {})

# Collecting , or "reaping" the value of `y`.
harvest(f, tag='intermediate')({}, 1.)  # ==> (3., {'y': 2.})
harvest(f, tag='intermediate')({}, 5.)  # ==> (7., {'y': 6.})
```

#### Using `reap` and `plant`.

`reap` and `plant` are simple wrappers around `harvest`. `reap` only pulls
intermediate values without injecting, and `plant` only injects values without
collecting intermediate values.

```python
def f(x):
  y = sow(x + 1., tag='intermediate', name='y')
  return y + 1.

# Injecting, or "planting" a value for `y`.
plant(f, tag='intermediate')({'y': 0.}, 1.)  # ==> 1.
plant(f, tag='intermediate')({'y': 0.}, 5.)  # ==> 1.

# Collecting , or "reaping" the value of `y`.
reap(f, tag='intermediate')(1.)  # ==> {'y': 2.}
reap(f, tag='intermediate')(5.)  # ==> {'y': 6.}
```
