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
def sub(key, m0, x):
    key, m_inner = genjax.trace("m", genjax.Normal)(key, (m0 + x, 1.0))
    return key, m_inner


@genjax.gen
def fn(key, x):
    key, m0 = genjax.trace("m0", genjax.Normal)(key, (x, 1.0))
    key, m_inner = genjax.trace("m", sub)(key, (m0, x))
    return key, m_inner


key = jax.random.PRNGKey(314159)
select = genjax.Selection([("m", "m")])
key, tr = fn.simulate(key, (0.0,))

key, (trace_grads, arg_grads) = fn.retval_grad(key, tr, select, (1.0,))
console.print(arg_grads)
console.print(trace_grads["m", "m"])
