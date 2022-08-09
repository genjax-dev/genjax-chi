import jax
from .generative_function import GEXTrace
from .core import handle, lift, Handler, I
from .intrinsics import trace_p, splice_p, unsplice_p

#####
##### GFI handlers
#####

# Note: these handlers _do not manipulate runtime values_ --
# the pointers they hold to objects like `v` and `score` are JAX `Tracer`
# values. When we do computations with these values,
# it adds to the `Jaxpr` trace.
#
# So the trick is write `callable` to coerce the return of the `Jaxpr`
# to send out the accumulated state we want.


class Simulate(Handler):
    def __init__(self):
        self.handles = [trace_p, splice_p, unsplice_p]
        self.state = {}
        self.level = []
        self.score = 0.0
        self.return_or_continue = False

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, *args, **kwargs):
        prim = kwargs["prim"]
        addr = kwargs["addr"]
        prim = prim()
        key, v = prim.sample(*args)
        self.state[(*self.level, addr)] = v
        score = prim.score(v, *args[1:])
        self.score += score
        if self.return_or_continue:
            return f(key, v)
        else:
            self.return_or_continue = True
            ret = f(key, v)
            return (ret, self.state, self.score)

    # Handle hierarchical addressing primitives (push onto level stack).
    def splice(self, f, addr):
        self.level.append(addr)
        return f([])

    # Handle hierarchical addressing primitives (pop off the level stack).
    def unsplice(self, f, addr):
        try:
            self.level.pop()
            return f([])
        except:
            return f([])

    # Return a `Jaxpr` with trace and addressing primitives staged out.
    def _stage(self, f, *args):
        expr = lift(f, *args)
        expr = handle([self], expr)
        expr = lift(expr, *args)
        return expr

    # Match the GFI interface return from Gen.
    def __interface_return(self, gen_fn, jitted, *args):
        (r, chm, score) = jitted(*args)
        return GEXTrace(gen_fn, jitted, args, r, chm, score)

    # JIT compile a function and return a function which implements
    # the semantics of `simulate` from Gen.
    def _jit(self, f, *args):
        expr = lift(f, *args)
        expr = handle([self], expr)
        jitted = jax.jit(expr)
        return lambda *args: self.__interface_return(f, jitted, *args)

    def jit(self, f):
        return lambda *args: self._jit(f, *args)

    def stage(self, f):
        return lambda *args: self._stage(f, *args)


class Generate(Handler):
    def __init__(self, choice_map):
        self.handles = [trace_p, splice_p, unsplice_p]
        self.state = {}
        self.level = []
        self.score = 0.0
        self.weight = 0.0
        self.obs = choice_map
        self.return_or_continue = False

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, *args, **kwargs):
        prim = kwargs["prim"]
        addr = kwargs["addr"]
        key = args[0]
        prim = prim()
        if (*self.level, addr) in self.obs:
            v = self.obs.get((*self.level, addr))
            score = prim.score(v, *args[1:])
            self.state[(*self.level, addr)] = v
            self.weight += score
        else:
            key, v = prim.sample(*args)
            self.state[(*self.level, addr)] = v
            score = prim.score(v, *args[1:])
        self.score += score
        if self.return_or_continue:
            return f(key, v)
        else:
            self.return_or_continue = True
            ret = f(key, v)
            return (self.weight, ret, self.state, self.score)

    # Handle hierarchical addressing primitives (push onto level stack).
    def splice(self, f, addr):
        self.level.append(addr)
        return f([])

    # Handle hierarchical addressing primitives (pop off the level stack).
    def unsplice(self, f, addr):
        try:
            self.level.pop()
            return f([])
        except:
            return f([])

    # Return a `Jaxpr` with trace and addressing primitives staged out.
    def _stage(self, f, *args):
        expr = lift(f, *args)
        expr = handle([self], expr)
        expr = lift(expr, *args)
        return expr

    # Match the GFI interface return from Gen.
    def __interface_return(self, gen_fn, jitted, *args):
        (w, r, chm, score) = jitted(*args)
        return w, GEXTrace(gen_fn, jitted, args, r, chm, score)

    # JIT compile a function and return a function which implements
    # the semantics of `generate` from Gen.
    def _jit(self, f, *args):
        expr = lift(f, *args)
        expr = handle([self], expr)
        jitted = jax.jit(expr)
        return lambda *args: self.__interface_return(f, jitted, *args)

    def jit(self, f):
        return lambda *args: self._jit(f, *args)

    def stage(self, f):
        return lambda *args: self._stage(f, *args)


class ArgumentGradients(Handler):
    def __init__(self, tr, argnums):
        self.handles = [trace_p, splice_p, unsplice_p]
        self.argnums = argnums
        self.level = []
        self.score = 0.0
        self.tr = tr
        self.tree = {}
        self.sources = tr.get_choices()
        self.return_or_continue = False

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, *args, **kwargs):
        prim = kwargs["prim"]
        addr = kwargs["addr"]
        key = args[0]
        prim = prim()
        v = self.sources.get((*self.level, addr))
        score = prim.score(v, *args[1:])
        self.score += score
        if self.return_or_continue:
            return f(key, v)
        else:
            self.return_or_continue = True
            ret = f(key, v)
            return self.score

    # Handle hierarchical addressing primitives (push onto level stack).
    def splice(self, f, addr):
        self.level.append(addr)
        return f([])

    # Handle hierarchical addressing primitives (pop off the level stack).
    def unsplice(self, f, addr):
        try:
            self.level.pop()
            return f([])
        except:
            return f([])

    # Return a `Jaxpr` with trace and addressing primitives staged out.
    def _stage(self, f, *args):
        expr = lift(f, *args)
        expr = handle([self], expr)
        expr = lift(expr, *args)
        return expr

    # Match the GFI interface return from Gen.
    def __interface_return(self, gen_fn, jitted, *args):
        arg_grads = jitted(*args)
        return arg_grads

    def _jit(self, f, *args):
        expr = lift(f, *args)
        expr = handle([self], expr)
        expr = jax.grad(expr, self.argnums)
        jitted = jax.jit(expr)
        return lambda *args: self.__interface_return(f, jitted, *args)

    def jit(self, f):
        return lambda *args: self._jit(f, *args)

    def stage(self, f):
        return lambda *args: self._stage(f, *args)


# This handler is used to stage the choice gradient computation.
# Parametrized by `chm`, this handler will seed `trace`
# with the values in that choice map.
#
# As long as our effect interpreter is composable with other JAX transformations
# we can define functions which accept `chm` and run the interpreter
# to seed values, and then lift these functions and compute
# their gradient.
#
# This allows us to express choice gradient computations.
class Sow(Handler):
    def __init__(self, fallback, chm):
        self.handles = [trace_p, splice_p, unsplice_p]
        self.level = []
        self.fallback = fallback
        self.chm = chm
        self.score = 0.0
        self.stored = {}

    # Handle trace sites -- perform codegen onto the `Jaxpr` trace.
    def trace(self, f, *args, **kwargs):
        prim = kwargs["prim"]
        addr = kwargs["addr"]
        key = args[0]
        prim = prim()
        if (*self.level, addr) in self.chm:
            v = self.chm.get((*self.level, addr))
            self.stored[(*self.level, addr)] = v
        else:
            v = self.fallback.get((*self.level, addr))
        score = prim.score(v, *args[1:])
        self.score += score
        ret = f(key, v)
        return self.score, self.stored

    # Handle hierarchical addressing primitives (push onto level stack).
    def splice(self, f, addr):
        self.level.append(addr)
        return f([])

    # Handle hierarchical addressing primitives (pop off the level stack).
    def unsplice(self, f, addr):
        try:
            self.level.pop()
            return f([])
        except:
            return f([])


class ChoiceGradients(Handler):
    def __init__(self, tr):
        self.tr = tr

    # Return a `Jaxpr` with trace and addressing primitives staged out.
    def _stage(self, f, *args):

        # Here, we define a utility function `sow` which
        # interprets with the `Sow` handler.
        def sow(chm):
            handler = Sow(self.tr.get_choices(), chm)
            expr = lift(f, *args)
            fn = handle([handler], expr)
            return fn(*args)

        # Here, because of composition -- we can lift `sow` itself,
        # and now we have a function parametrized by `chm`
        # which we can compute the gradient of.
        #
        # This function computes the logpdf (score) of the trace,
        # but it accepts the seed `chm` as an argument -- and then stages out
        # the gradient computation using `jax.grad`.
        return lambda chm: lift(jax.grad(sow, has_aux=True), chm)

    # Match the GFI interface return from Gen.
    def __interface_return(self, gen_fn, jitted, chm):
        choice_grads, choices = jitted(chm)
        return choice_grads

    def _jit(self, f, *args):
        def sow(chm):
            handler = Sow(self.tr.get_choices(), chm)
            expr = lift(f, *args)
            fn = handle([handler], expr)
            return fn(*args)

        jitted = jax.jit(jax.grad(sow, has_aux=True))
        return lambda chm: self.__interface_return(f, jitted, chm)

    def jit(self, f):
        return lambda *args: self._jit(f, *args)

    def stage(self, f):
        return lambda *args: self._stage(f, *args)
