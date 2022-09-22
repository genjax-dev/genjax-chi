import jax


def f(x, _):
    return x


fn = jax.jit(f, abstracted_axes={0: "n"})
