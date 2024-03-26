# Motivation for having a masking system

## Introduction

Consider a generative function
```python
@genjax.static_gen_fn
def k(x):
    a = normal(x, 1) @ "a"
    b = normal(a, 2) @ "b"
    return b
```

Gen's semantics require that we can call `k.importance(key, cmp, x)`
for any of the following choicemaps:
- `cmp = genjax.choice_map({"a": 0.0, "b": 0.0})`
- `cmp = genjax.choice_map({"a": 0.0})`
- `cmp = genjax.choice_map({"b": 0.0})`
- `cmp = genjax.choice_map({})`

This all works.

GenJAX also supports calls to
```python
vmap(k.importance, in_axes=(0, 0, 0))(
    keys, cmps, xs
)
```
for any of the following choicemaps:
- `cmps = genjax.choice_map({"a": jnp.array([0., 0.]), "b": jnp.array([0., 0.])})`
- `cmps = genjax.choice_map({"a": jnp.array([0., 0.])})`
- `cmps = genjax.choice_map({"b": jnp.array([0., 0.])})`
- `cmps = genjax.choice_map({})`

Each of these calls will produce a batched trace effectively containing
2 traces to `k`.

There is no way in GenJAX to construct `cmps` so that the vmapped
importance call produces a batched trace contains one trace
generated with constraints `choice_map({a: 0.0})` and one trace
with constraints  `choice_map({b: 0.0})`.
Any "batched choicemap" must have the same address set for all
"sub choicemaps" in the batch (e.g. either both constrain a, or both constrain b).
I am not entirely sure how critical a limitation this is to resolve.
In any case, I think the masking proposal below will provide a solution
to this problem (for which future extensions could potentially
add syntactic sugar).

## The problem
### Problem 1: Constraining vector traces
Now consider the mapped generative function
```python
mapped_v1 = genjax.map_combinator(
    in_axes=(0,)
)(k)
```

Say I call `tr = mapped_v1.simulate(key, (xs,))` where `key` is a
PRNG key and `xs` is a 3-vector of inputs.  Then
`tr` will be a trace with a choicemap with addresses set
```
S = {("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 1), ("b", 2)}
```
Therefore, per Gen's semantics, it should be possible to call
`mapped_v1.importance(keyk, chm, xs)` with `key` and `xs` as above,
and `chm` containing any subset of the the addresses in `S`.
For instance, it should work if chms is a choicemap which constrains
different addresses sub-addresses at address prefixes 0 and 1, like
- `chms = genjax.choice_map({(0, "a"): 0.0, (1, "b"): 0.0})`

GenJAX currently does not support this, and thereby does not
provide a full implementation of the GFI for the `map_combinator`.

The challenge is that the address prefixes 0, 1, etc., in
`map_combinator.importance` correspond to batch indices in
an under-the-hood
call to `vmap(k.importance)`.  As described
at the end of the introduction,
such vmapped calls cannot constrain different address
sets at different points in the batch.

### Problem 2: dynamic updates to model structure

Let `tr = mapped_v1.simulate(key, (xs,))`
where `xs` is a length 3 vector.
Then `tr.update(key, (new_xs,))` only works
if `new_xs` is also a length 3 vector.
(It does not work, say, if `new_xs` is a length 4 vector.)

I believe this is also an instance in which genjax does not
support the full GFI for the `map_combinator` (which should
support updates to any valid argument to a generative function).

One "solution" would be to make the vector length
an argument to the `map_combinator` constructor.
But this rules out important use cases, like the following.

### Motivating use case necessitating a solution to problem 2: inference in open-universe models

Consider the model
```python
@genjax.static_gen_fn
def generate_n_times():
    n = poisson(5) @ "n"
    vals = mapped_v1(jnp.arange(n)) @ "vals"
    obs = normal(jnp.sum(vals), 1.0) @ "obs"
    return obs
```

Such models are sometimes called "open-universe"
models because the "universe" of addresses
which will appear in the choicemap
is not known a priori, but vary in different
traces.

Inference algorithms in a model like this need to make updates
which change the value of `n`.
This is not currently possible in genjax due to the inability
to change update the length of the vector passed
as argument to the `map_combinator`.

# Proposed masking system
The core of the problems above is that batched JAX calls require
statically-known structure shared across the batch.
This means that at the level of a jax batched call,
we cannot change the size of an array, or do the computation
associated with constraining address "a" at parts of the batch
but not others.

## Illustrations

### Solution to Problem 1: constraining vector traces
The masking system will allow us to call
```python
chm = genjax.choice_map({
    (0, "a"): 0.,
    (1, "b"): 0.
})
mapped_v1.importance(key, chm, xs)
```

Internally, this call will be processed by calling
```python
static_structure_cm = _statically_restructure_cm(chm)
mapped_v1._importance(key, static_structure_cm, xs)
```
Here, `MapCombinator._importance` is an internal
method which is not exposed to the user, and which
takes a choicemap with a statically-known structure
(like the current implementation of `MapCombinator.importance`).

`_statically_restructure_cm` will be a helper function
which uses the masking system to produce a choicemap
where each index contains the same set of sub-addresses, and
the masking system is used to "turn off" sub-addresses
at certain indices.
In the above example, it will produce
```python
from genjax import MaskedChoice
static_structure_cm = genjax.indexed_choicemap(
    [0, 1],
    genjax.choicemap({
        "a": MaskedChoice(jnp.array([True, False]), jnp.array([0., NaN])),
        "b": MaskedChoice(jnp.array([False, True]), jnp.array([NaN, 0.]))
    })
)
```

I will describe how this masked choicemap will be processed
in [the next section](#details-of-proposed-masking-system).

### Solution to Problem 2: dynamic updates to model structure

Recall the problem of calling `tr.update` to change the
length of the vector passed to the `map_combinator`.
I propose that our solution will be to
1. Make the vector length an argument to the `map_combinator` constructor.
    (Thus, `update` cannot even be called in such a way that would
    look like it should change the map length.  This will
    remove the issue that the current map_combinator
    GFI implementation will have full coverage
    with respect to the vector argument.)
2. When users want to instantiate a map with variable length,
    they can fix a maximum length, and manually use the masking system
    to "turn off" the extra elements in the vector.
    (This is a form of "low-level GPU memory management"
    that genjax users will need to do.  Eventually, we
    should add higher-level constructs that automate this.
    But first, I propose that we build a solid low-level interface.)

The following pattern can be used to simulate having a variable-length
map.
```python
masked_kernel = genjax.mask_combinator(k)
# masked_kernel takes argument `(flag, (x,))`

mapped_v2 = genjax.map_combinator(
    in_axes=(0, (0,)),
    max_length=10
)(masked_kernel)
# mapped_v2 takes argument `(flag_vector, (x_vector,))`

# Generate a trace with 3 elements
tr = mapped_v2.simulate(
    key,
    (jnp.arange(10) < 3, (jnp.zeros(10),))
)

print(tr.get_choices())
# ^this^ prints a choicemap with addresses
# {(0, "a"), (0, "b"), (1, "a"), (1, "b"), (2, "a"), (2, "b")}

retval = tr.get_retval()
print(retval)
# ^this^ prints an `Mask` object where `retval.flag`
# is [1, 1, 1, 0, 0, 0, 0, 0, 0, 0].
# `retval.value` will be a 10-vector where only the first 3
# elements are semantically meaningful.

# Update the trace to have 5 elements
tr2 = tr.update(
    key,
    (jnp.arange(10) < 5, (jnp.zeros(10),))
)

tr2.get_choices()
# ^this^ prints a choicemap with addresses
# {(0, "a"), (0, "b"), (1, "a"), (1, "b"), (2, "a"), (2, "b"),
# (3, "a"), (3, "b"), (4, "a"), (4, "b")}.
# The values at addresses 0, 1, 2 are the same as in the initial trace.
```

See [this tutorial](https://github.com/probcomp/lis_bayes3d/blob/main/01_tutorial-model.ipynb)
for a more in-depth exposition of this pattern, and its use
in an open-universe model.

## Details of proposed masking system

### Mask type
```python
class Mask:
    flag: BoolArray
    value: Any
```
This represents either
- nothing (`flag = False`), or
- a value (`flag = True`).

### MaskedChoice
This is a subclass of `Mask` which represents a `genjax.Choice`
which may be masked.

It is implemented as follows.
```python
class MaskedChoice:
    flag: genjax.Choice
    value: genjax.Choice
```
The `flag` is a choicemap with the same shape
as `value`, and where all leaves are boolean arrays.

The difference between `MaskedChoice` and `Mask` is that
`MaskedChoice` is also a subclass of `genjax.Choice`, and
it implements the choicemap interface, as follows.

- `MaskedChoice.isempty()` returns `True` if all values in
    the `flag` choicemap are `False`,
    or if `value` is a `genjax.Choice` which is empty.
- `MaskedChoice.get_value()` returns `value.get_value()`
    if `flag.get_value()` is `True`, and raises an error otherwise.
- `MaskedChoice.get_submap(addr)` returns `MaskedChoice(flag.get_submap(addr), value.get_submap(addr))`.

We should have syntactic sugar
```python
MaskedChoice(flag, value) =
    MaskedChoice(genjax.ValueChoice(flag), genjax.ValueChoice(value))
```

My aim is that if we call
```python
static_structure_cm = genjax.indexed_choicemap(
    [0, 1],
    genjax.choicemap({
        "a": MaskedChoice(jnp.array([True, False]), jnp.array([0., NaN])),
        "b": MaskedChoice(jnp.array([False, True]), jnp.array([NaN, 0.]))
    })
)
mapped_v1._importance(key, static_structure_cm, xs)
```
where `MapCombinator._importance` will execute the same code
code that `MapCombinator.importance` currently executes,
then the JAX tracing process will automatically compile an update
which checks the flag values in the MaskedChoice objects
at each leaf node (ie. each distribution), and decides
whether to constrain or generate the value at each one.

I think (and hope!) one of the following is true:
1. This will immediately work (modulo minor debugging)
    given our current implementation.
2. To get this to work, the only thing we will need to do is
    rewrite the implementation of all the genjax `Distribution`
    types, so that their `importance`, etc., methods all support
    dynamic dispatching when given a `MaskedChoice`.
    (It may be that the current implementation assumes 
    that the update structure is statically known.)
(The reason everything else would happen automatically
would be because we implement `isempty`, `get_submap`, etc.,
on `MaskedChoice`.)

### `mask_combinator`

Say `k` is a generative function which takes args (a_0, ..., a_n).
Then `genjax.mask_combinator(k)` will be a generative function which
takes args `(flag, (a_0, ..., a_n))` where `flag` is a bool.
It represents the following family of distributions over choicemaps:
- if `flag` is `False`, it represents the dirac delta distribution
    at the empty choicemap `genjax.choice_map({})`
- if `flag` is `True`, it represents the same
    distribution over choicemaps as `k(a_0, ..., a_n)`.

The retval of `genjax.mask_combinator(k)` will be a `Mask(flag, val)`
object where `val` is the retval of `k(a_0, ..., a_n)`,
and `flag` is the same as the value of the `flag` argument passed to
`mask_combinator(k)`.

### Accessing values in `Mask` objects

MVP implementation: users are responsible for checking the `flag` array
and only accessing values where `flag` is `True`.
I think it is okay to ship a release with this.

The current WIP masking system has a `Mask.match` method
for safer unwrapping of values.
Bugs currently arise when trying to call `mask.match` when
`mask` is a batched `Mask` object where the value is a
pytree.
An MVP solution would be to only allow `mask.match` to be called
when `mask` is a non-batched `Mask` object,
and have users call `vmap(lambda m: m.match())(batched_masks)`
when they need to do this in a batched context.
I support also including this.

In future extensions, we can try to improve syntactic sugar
for this, like adding a `mask.vmatch` method which automatically
applies `vmap` to the `match` method.

# Questions
1. Should `MaskedChoice` and `Mask` be the same class, or different classes?
    (I propose that they should be different, but I don't feel strongly.)
2. Is there a way we can build more confidence that this design will have
    longevity and behave well under all the forms of composition we wish
    to support?
    (I have tried to integrate it closely with the GFI which we know
    has an effective form of compositionality.)
3. This proposal states that we should get the vector combinators
    to have full GFI coverage, in the sense that any gen choicemap
    should be able to be passed to `importance` and `update`.
    Internally, this will be implemented by lifting the input choicemap
    type to a more structured choicemap type that a lower-level
    `combinator._importance` method can handle.
    Should we (now or later) develop a better compositional type system
    for reasoning about the more restrictive form of choicemaps
    accepted by `_importance`?
4. [Feel free to add questions...]

# TODOs to implement this proposal

TODOs for the core masking system:
1. Add test cases for each problem described in [the problem section](#the-problem).
2. Implement and debug `Mask` and `MaskedChoice` (starting from the current `Mask`
    implementation).
3. Test and debug calling `model.importance` with a `MaskedChoice` object.
    Verify that we can get rich dynamically-structured updates out of the
    current interpreter.
    Do this by adding dynamic support to one or two distribution types
    and testing that the interpreter can handle this.
4. If 3 worked, update the implementation of all the distributions.
5. Update and debug `mask_combinator`.

Other genjax improvements we can make with the masking system:
1. Improve the vector combinator implementations so that they have
    full coverage of the GFI.
    1. Add a `max_length` argument to the `map_combinator` constructor.
    2. When `importance` or `update` is called with different addresses at
        different indices, lift the constraint choicemap to a uniformly
        shaped choicemap with `MaskedChoice` instances to represent missing addresses.
2. Improve the Switch combinator implementation using the masking system.
    (I, George, am not currently familiar with the switch combinator in genjax,
    so I will defer the specifics.)
