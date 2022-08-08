# gex

> [**Ge**n](https://www.mct.dev/assets/mct-thesis.pdf) ⊗ [JA**X**](https://github.com/google/jax)

## Implementation strategy

The implementation strategy is based on: [Handling effects with JAX](https://colab.research.google.com/drive/1HGs59anVC2AOsmt7C4v8yD6v8gZSJGm6#scrollTo=OHUTBFIiHJu3) extended to support dynamically specified handlers. As in the original, handlers are staged out (zero-cost, not dynamically dispatched). This implementation also takes inspiration from [Zero-cost Effect Handlers by Staging](http://ps.informatik.uni-tuebingen.de/publications/schuster19zero.pdf) - without automatic insertion of staging annotations.

## Tour

`gex` generative functions are pure Python functions from `(PRNGSeed, *args)` to `(PRNGSeed, retval)`.

Let's study an example program in the DSL:

```python
# Here's a program with our primitives.
def f(key, x):
    key, m1 = gex.trace("m1", gex.Bernoulli)(key, x)
    key, m2 = gex.trace("m2", gex.Bernoulli)(key, x)
    return (key, 2 * (m1 + m2))
```

Choices are specified using the `gex.trace(addr, primitive)(*args)` syntax. Here, our program exposes `'m1'` and `'m2'` as Bernoulli choices.

We can examine our initial `Jaxpr` syntax using a `lift` operator:

```python
{ lambda ; a:u32[2] b:f32[]. let
    c:u32[2] d:bool[] = trace[
      addr=m1
      prim=<class 'gex.distributions.Bernoulli'>
    ] a b
    e:u32[2] f:bool[] = trace[
      addr=m2
      prim=<class 'gex.distributions.Bernoulli'>
    ] c b
    g:bool[] = or d f
    h:i32[] = convert_element_type[new_dtype=int32 weak_type=True] g
    i:i32[] = mul h 2
  in (e, i) }
```

Notice here that the primitive distribution objects are inlined into our primitive, as is the address. Later on, our handling interpreter will fetch these objects out and use them to codegen/handle at the site of `trace` -- keep this in mind.

`trace` is a `core.Primitive` (from `jax`) which we've defined -- it has no XLA or native interpretation, instead we have to handle it by providing an interpretation for it, a desugaring to native `jax` constructs.

Here's one way to handle `trace` (which I'll unpack!):

```python
class Simulate(Handler):

  ...

  def trace(self, f, *args, **kwargs):
      prim = kwargs["prim"]
      addr = kwargs["addr"]
      prim = prim()
      key, v = prim.sample(*args)
      self.state[(*self.level, addr)] = v
      score = prim.score(v, args[1])
      self.score += score
      if self.return_or_continue:
          return f(key, v)
      else:
          self.return_or_continue = True
          ret = f(key, v)
          return (ret, self.state, self.score)
```

Inspired by effect handling, our interpreter (which walks the IR and attempts to handle the `trace` primitive) will create a continuation object (for the rest of the computation) at the `trace` site, and provide that object to our handler `trace` here (that continuation is `f` here).

Roughly, what this handler does is use PRNG sampling primitives to transform a `PRNGSeed` into a `prim` sample, score that sample, and then track that sample as state which we'll use later to create a trace. We call the continuation `f(key, v)` to return to the computation, handle more `trace` statements -- our object has a flag `self.return_or_continue` which helps direct codegen (we wait until we've visited all `trace` statements, and return out of the continuation -- and then we return with the data required to create a trace object from the call).

> **Important (JAX Tracer time vs runtime)**
>
> It's important to understand that this handling/interpretation operates on abstract `Tracer` values -- **not runtime values**. The handler above is not something that happens at runtime -- in our interpreter, when we call this handler for `trace` -- it's inlining the traced `jax` definitions into the `Jaxpr` which we'll eventually emit to run on XLA, etc. So when you read a handler definition like `trace` above -- remember that the definition is guiding the trace/codegen process (and is not a runtime process). Note that this also means that tracking `Traced` values in a Python object (like the implementation above does with `self.score` and `self.state`) is perfectly valid, and will be completely eliminated at runtime because it is staged out in the `Jaxpr`.

#### Using the GFI

Using the GFI has a slight twist due to the staging/interpretation process above, as well as our desire to utilize `jax` JIT compilation -- here's a usage pattern:

```python
def f(key, x):
    key, m1 = gex.trace("m1", gex.Bernoulli)(key, x)
    key, m2 = gex.trace("m2", gex.Bernoulli)(key, x)
    return (key, 2 * (m1 + m2))


key = jax.random.PRNGKey(314159)
fn = gex.Simulate().jit(f)(key, 0.3)
tr = fn(key, 0.3)
print(tr.get_choices())
```

We define a generative function which utilizes our primitives, then we can call a handler implementation object like `Simulate()` to stage out / jit our generative function -- _this implements the semantics of `simulate`_.

`fn` here is a JIT-backed callable which returns a `GEXTrace`.

#### The interpreter

It's worth exploring the effect/handling interpreter. It operates on `Jaxpr` objects, and is parametrized by a `Sequence[Handler]` a.k.a. a handler stack. This interpreter is very similar to `eval_jaxpr` in `jax.core` -- with the evaluation loop replaced by recursion, as well as an explicit ability to construct Python-embedded continuation objects.

```python
if eqns:
    eqn = eqns[0]
    kwargs = eqn.params
    in_vals = map(read, eqn.invars)
    in_vals = list(in_vals)
    subfuns, params = eqn.primitive.get_bind_params(eqn.params)
    if hasattr(eqn.primitive, "must_handle"):
        args = subfuns + in_vals
        # This definition "reifies" the remainder of the evaluation
        # loop so it can be explicitly passed to the handler.
        def continuation(*args):
            return eval_jaxpr_loop(eqns[1:], env, eqn.outvars, [*args])

        for handler in reversed(handler_stack):
            if eqn.primitive == handler.handles:
                return handler.callable(continuation, *args, **kwargs)
        raise ValueError("Failed to find handler.")
    else:
        ans = eqn.primitive.bind(*(subfuns + in_vals), **params)
        if not eqn.primitive.multiple_results:
            ans = [ans]
        return eval_jaxpr_loop(eqns[1:], env, eqn.outvars, ans)
else:
    return map(read, jaxpr.outvars)
```

Here, we loop over `eqns` (roughly: SSA lines in the `Jaxpr` representation) and check if the line is a primitive which requires handling. If it is, we capture the continuation (reified syntax, plus environment, etc -- bundled with our interpreter) and pass the continuation (as well as the arguments to the operator which raised the "must be handled" flag) to the handler.
