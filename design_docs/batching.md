# Proposed plan for genjax batching semantics

### Core semantics

Semantics for batched traces:
- GFI methods for `JAXGenerativeFunctions` can all be vmapped.
- The return types of vmapped `simulate`, `importance`,
    `update`, and `regenerate` will be no different from the
    typical semantics when vmapping a function that outputs a pytree.
    In these cases, one of the returned pytrees will happen to be a
    `genjax.Trace` object.
    We will call such traces "batched traces"
    and all other trace objects "unbatched traces".
- In the initial implementation of this proposal, genjax
    will not symbolically represent batched traces differently
    from regular traces; users are responsible for tracking
    which traces are batched and which are not (just as
    JAX users are typically responsible for tracking the shapes
    of the arrays produced by their code).
- Genjax trace methods (including `tr.get_args`, `tr.get_choices`,
    `tr.get_retval`, `tr.get_score`, `tr.update`, and `tr.project`)
    are only well-defined for unbatched traces.
    If one wishes to call `get_args` across a batched trace,
    they should write `vmap(lambda tr: tr.get_args())(batched_traces)`.

Semantics for the vector combinators (map, repeat, and unfold):
- Let `c` be an instance of a map, repeat, or unfold combinator,
    wrapping an underlying generative function "kernel" `k`.
- `c.simulate` and `c.importance` on  will yield an unbatched
    trace.  This is totally standard as `c` will be an instance
    of a subtype of `GenerativeFunction` (and indeed,
    non-vmapped simulate and importance yielding an
    unbatched trace is required by the GFI).
- Let `T` be the return type of the kernel `k`.  Since
    `k` must be a `JAXGenerativeFunction`, `T` will be a
    pytree.
    The return type of the generative function `c`
    will be a batched instance of `k`, where the batch
    dimensions will be determined by the constructor call for
    `c`.  (These are the `in_axes` in the map combinator.)

### TODOs

Immediate priorities:
1. Update all genjax source code to enure trace methods
    are never called directly on batched traces.
2. Document the return type of vector combinators.
3. Write a quick tutorial on batching and combinator calls in genjax.
    The goal is that ChiSight users who are good with JAX should be able
    to read this in <20 minutes (maybe even just 5 minutes).
    Prioritize getting this out over making it super clean.

Priorities before public genjax release:
1. Improve the tutorial on batching and combinator calls in genjax.

### Possible future extensions
- Add a mechanism to genjax to automatically track whether a trace
is batched or not.  Raise errors if a user tries to call
    a trace method like `project` or `update` on a batched trace.
- Add syntactic sugar for expressions like `vmap(lambda tr: tr.get_args())(tr)`.  (One idea: add `tr.vget_args`, `tr.vupdate`, `tr.vproject`, etc. methods that automatically apply `vmap`.)
