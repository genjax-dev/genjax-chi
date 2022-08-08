import jax
from .generative_function import GEXTrace
from .core import handle, lift, Handler
from .intrinsics import trace_p, splice_p, unsplice_p

#####
##### Stateful handlers
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
    def __interface_return(self, fn, *args):
        (r, chm, score) = fn(*args)
        return GEXTrace(fn, args, r, chm, score)

    # JIT compile a function and return a function which implements
    # the semantics of `simulate` from Gen.
    def _jit(self, f, *args):
        expr = lift(f, *args)
        expr = handle([self], expr)
        fn = jax.jit(expr)
        return lambda *args: self.__interface_return(fn, *args)

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
    def __interface_return(self, fn, *args):
        (w, r, chm, score) = fn(*args)
        return w, GEXTrace(fn, args, r, chm, score)

    # JIT compile a function and return a function which implements
    # the semantics of `simulate` from Gen.
    def _jit(self, f, *args):
        expr = lift(f, *args)
        expr = handle([self], expr)
        fn = jax.jit(expr)
        return lambda *args: self.__interface_return(fn, *args)

    def jit(self, f):
        return lambda *args: self._jit(f, *args)

    def stage(self, f):
        return lambda *args: self._stage(f, *args)
