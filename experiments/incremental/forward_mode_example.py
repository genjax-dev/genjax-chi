# Copyright 2022 MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
from jax import lax
import jax.numpy as jnp
import genjax
from genjax.experimental.interpreters.propagating import Cell
from genjax.experimental.interpreters import trace_utils

# Here's an example of defining a simple forward-mode like interface
# using a graph-based propagation interpreter.
#
# The interpreter interface `propagate` is also JIT compilable.


# First, we must define the lattice.
# This will typically be a type of lifted values,
# here, for example -- we'll use dual numbers.
class Dual(Cell):
    def __init__(self, aval, val, dual):
        super().__init__(aval)
        self.val = val
        self.dual = dual

    def __lt__(self, other):
        return self.bottom() and other.top()

    def top(self):
        return self.dual is not None

    def bottom(self):
        return self.val is None and self.dual is None

    def join(self, other):
        if other.bottom():
            return self
        else:
            return other

    @classmethod
    def new(cls, val, dual=None):
        aval = trace_utils.get_shaped_aval(val)
        return Dual(aval, val, dual)

    @classmethod
    def unknown(cls, aval):
        return Dual(aval, None, None)

    def flatten(self):
        return (self.val, self.dual), (self.aval,)

    @classmethod
    def unflatten(cls, data, xs):
        return Dual(*data, *xs)

    def __repr__(self):
        return f"Dual({self.val}, {self.dual})"


# Now, we can register propagation rules for the interpreter.
def exp_rule(invals, outvals):
    (inval,) = invals
    (outval,) = outvals
    if outval.bottom():
        outval = Dual.new(jnp.exp(inval.val), dual=jnp.exp(inval.val))
        outvals = [outval]
    return invals, outvals, None


def add_rule(invals, outvals):
    (a, b) = invals
    (outval,) = outvals
    if not a.bottom() and not b.bottom():
        outval = Dual.new(a.val + b.val, dual=a.dual + b.dual)
        outvals = [outval]
    return invals, outvals, None


def mul_rule(invals, outvals):
    (a, b) = invals
    (outval,) = outvals
    if not a.bottom() and not b.bottom():
        outval = Dual.new(a.val * b.val, dual=a.dual * b.val + a.val * b.dual)
        outvals = [outval]
    return invals, outvals, None


forward_ad_rules = {}
forward_ad_rules[lax.exp_p] = exp_rule
forward_ad_rules[lax.add_p] = add_rule
forward_ad_rules[lax.mul_p] = mul_rule

#####
# Test
#####


def f(x):
    return x * (jnp.exp(x) + x)


def forward_mode(f):
    def _inner(x):
        jaxpr, _ = trace_utils.stage(f)(x)
        jaxpr, consts = jaxpr.jaxpr, jaxpr.literals
        env, _ = genjax.propagate(
            Dual,
            forward_ad_rules,
            jaxpr,
            list(map(Dual.new, consts)),
            [Dual.new(x, dual=1.0) for var in jaxpr.invars],
            [Dual.unknown(var.aval) for var in jaxpr.outvars],
        )
        return env[jaxpr.outvars[0]]

    return _inner


jaxpr = jax.make_jaxpr(f)(2.0)
forward_jaxpr = jax.make_jaxpr(forward_mode(f))(2.0)
dual = jax.jit(forward_mode(f))(2.0)
print(jaxpr)
print(forward_jaxpr)
print(dual)
