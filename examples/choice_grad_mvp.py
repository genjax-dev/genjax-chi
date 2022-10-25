import jax

import genjax


# `go_pretty` provides a console object which supports
# pretty printing of `Pytree` objects.
console = genjax.go_pretty()

# A `genjax` generative function is a pure Python function from
# `(PRNGKey, *args)` to `(PRNGKey, retval)`
#
# The programmer is free to use other JAX primitives, etc -- as desired.
#
# The models below are rather simplistic, but demonstrate the basics.


@genjax.gen
def model(
    key,
):
    key, x = genjax.trace("x", genjax.Normal)(key, (0.0, 1.0))
    key, y = genjax.trace("y", genjax.Normal)(key, (x, 0.2))
    return (key,)


key = jax.random.PRNGKey(314159)
select = genjax.Selection([("x",)])
choices = genjax.ChoiceMap.new({("y",): 0.5})
key, (_, tr) = jax.jit(model.importance)(key, choices, ())

key, trace_grads, arg_grads = model.choice_grad(key, tr, select, ())
console.print(tr["x"])
console.print(trace_grads["x"])
console.print(arg_grads)
