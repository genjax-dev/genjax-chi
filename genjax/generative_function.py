from .core import I, Trace
from .handlers import Simulate, Generate, ArgumentGradients, ChoiceGradients


def simulate(f):
    def _inner(*args):
        jitted = Simulate().jit(f)(*args)
        (r, chm, score) = jitted(*args)
        return Trace(args, r, chm, score)

    return lambda *args: _inner(*args)


def generate(f, chm):
    def _inner(chm, *args):
        jitted = Generate(chm).jit(f)(*args)
        (w, r, chm, score) = jitted(*args)
        return w, Trace(args, r, chm, score)

    return lambda *args: _inner(chm, *args)


def arg_grad(f, tr, argnums):
    def _inner(tr, argnums, *args):
        jitted = ArgumentGradients(tr, argnums).jit(f)(*args)
        arg_grads = jitted(*args)
        return arg_grads

    return lambda *args: _inner(tr, argnums, *args)


def choice_grad(f, tr, *args):
    def _inner(tr, chm, *args):
        jitted = ChoiceGradients(tr).jit(f)(*args)
        choice_grads = jitted(chm)
        return choice_grads

    return lambda chm: _inner(tr, chm, *args)
