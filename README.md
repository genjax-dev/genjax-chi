# gax

> [**G**en](https://www.mct.dev/assets/mct-thesis.pdf) ⊗ [J**AX**](https://github.com/google/jax)

## Implementation strategy

The implementation strategy is based on: [Handling effects with JAX](https://colab.research.google.com/drive/1HGs59anVC2AOsmt7C4v8yD6v8gZSJGm6#scrollTo=OHUTBFIiHJu3) extended to support dynamically specified handlers. As in the original, handlers are staged out (and are zero-cost). This implementation also takes inspiration from [Zero-cost Effect Handlers by Staging](http://ps.informatik.uni-tuebingen.de/publications/schuster19zero.pdf) - without automatic insertion of staging annotations.

## Example

Let's study an example program in the DSL:

```python
# Here's a program with our primitives.
def f(x):
    u = trace(0, bernoulli, x)
    z = trace(1, bernoulli, x)
    q = trace(2, bernoulli, x)
    return u + z + q
```

To conveniently support pure JAX values, addresses are integers (for now, this restriction can easily be lifted with a bit of sugar). `trace` is a primitive denoting a random choice - but `trace` desugars into the primitive or call argument (after the address):

```python
{ lambda ; a:f32[]. let
    b:bool[] = bernoulli 0 a
    c:bool[] = bernoulli 1 a
    d:bool[] = bernoulli 2 a
    e:bool[] = or b c
    f:bool[] = or e d
  in (f,) }
```

`bernoulli` is a primitive which must be handled by our interpreter -- it's not something which supports a JAX or XLA-native implementation. To construct a handler, we define a function which accepts the continuation `f`, as well as the arguments to `bernoulli`:

```python
# Declare a handler + wrap in `Handler`.
def _handle_bernoulli(f, addr, p):
    key = seed()
    v = random.bernoulli(key, p)
    score = stats.bernoulli.logpmf(v, p)
    state(addr, v, score)
    return f(v)


handle_bernoulli = Handler(bernoulli_p, _handle_bernoulli)
```

We register `handle_bernoulli` as a `Handler` object. Before proceeding, it's worth understanding a bit about the architecture of the effect/handling interpreter.
The signature of the interpreter:

```python
def eval_jaxpr_handler(
    handler_stack: Sequence[Handler], jaxpr: core.Jaxpr, consts, *args
)
```

It operates on `Jaxpr` objects, and is parametrized by a `HandlerStack`. This interpreter is very similar to `eval_jaxpr` in `jax.core` -- with the evaluation loop replaced by recursion, as well as an explicit ability to construct Python-embedded continuation objects.

```python
if eqns:
    eqn = eqns[0]
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
                return handler.callable(continuation, *args)
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

Luckily, with our implementation -- we can incrementally stage out the effect primitives. Let's examine our handler for `bernoulli` once more:

```python
# Declare a handler + wrap in `Handler`.
def _handle_bernoulli(f, addr, p):
    key = seed()
    v = random.bernoulli(key, p)
    score = stats.bernoulli.logpmf(v, p)
    state(addr, v, score)
    return f(v)
```

Both `seed()` and `state(addr, score)` are also effectful primitives! Handlers can also raise effects!

We can stage out `bernoulli` by providing a handler wrapping `_handle_bernoulli`:

```python
# Here, we use the effect interpreter after desugaring
# `trace`, and then we stage that out to `Jaxpr` syntax.
expr = lift(f, 0.7)
expr = handle([handle_bernoulli], expr)
expr = lift(expr, 0.7)
print(expr)
```

We get a massive `Jaxpr` -- but `bernoulli` is gone (and in it's place, a whole bunch of operations have been inlined) -- but `seed` and `state` are still around!

#### Stateful handlers

To eliminate the `seed` and `state` requests, we need a handler which can run a separate program "on the side" every time we request `seed` and `state`. `seed` essentially asks: "give me a new PRNG key to use in a sample transformation" -- we need to provide a handler which can provide seed, and then update its own copy of seed for the next request.

```python
class PRNGProvider(Handler):
    def __init__(self, seed: int):
        self.handles = seed_p
        self.state = jax.random.PRNGKey(seed)

    def callable(self, f):
        key, sub_key = jax.random.split(self.state)
        self.state = key
        return f(sub_key)
```

We can eliminate `seed` now, and `JAX` will track our updates in the handler. We do something similar for `state`:

```python
class TraceRecorder(Handler):
    def __init__(self):
        self.handles = state_p
        self.state = []
        self.score = 0.0
        self.return_or_continue = False

    def callable(self, f, addr, v, score):
        self.state.append(v)
        self.score += score
        if self.return_or_continue:
            return f(v)
        else:
            self.return_or_continue = True
            ret = f(v)
            return (ret, self.state, self.score)
```

Again, think of `callable` not as something which manipulates runtime values, but a code generator which puts `Tracer` values + eqns into the final `Jaxpr` which we'll compile to XLA.

Here, we accumulate the score -- and we keep track of the choice values. This handler also includes a flag to correctly determine when to throw away the continuation, and just return the entire bundle of state `(ret, choices, score)` at the end.
