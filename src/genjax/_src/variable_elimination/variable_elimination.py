import jax
import jax.core as jc
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import util as jax_util

from genjax._src.core.compiler.interpreters.environment import Environment
from genjax._src.core.compiler.staging import stage
from genjax._src.core.generative.choice_map import StaticAddress
from genjax._src.core.pytree import Pytree
from genjax._src.core.typing import Any, TypeVar
from genjax._src.generative_functions.distributions.distribution import ExactDensity
from genjax._src.generative_functions.distributions.tensorflow_probability import flip
from genjax._src.generative_functions.static import StaticGenerativeFunction, trace, trace_p

R = TypeVar("R", bound=Any)

class VariableEliminationInterpreter():
    def __init__(self, eliminate: StaticAddress, support=None):
        super().__init__()
        self.eliminate = eliminate
        self.enumerated = set()
        self.support = support

    def _eval_jaxpr_forward(
        self,
        jaxpr: jc.Jaxpr,
        consts: list[Any],
        args: list[Any],
        args_enumerated: list[bool],
    ) -> tuple[list[Any], list[bool]]:
        env = Environment()
        env_enum = Environment()
        logprobs = None
        jax_util.safe_map(env.write, jaxpr.constvars, consts)
        jax_util.safe_map(env_enum.write, jaxpr.constvars, [False for _ in consts])
        jax_util.safe_map(env.write, jaxpr.invars, args)
        jax_util.safe_map(env_enum.write, jaxpr.invars, args_enumerated)
        def enum_read(env):
            return lambda var: env.read(var) if isinstance(var, jc.Var) else False
        for eqn in jaxpr.eqns:
            invals = jax_util.safe_map(env.read, eqn.invars)
            invals_enum = jax_util.safe_map(enum_read(env_enum), eqn.invars)
            subfuns, params = eqn.primitive.get_bind_params(eqn.params)
            args = subfuns + invals
            args_enumerated = [False for _ in subfuns] + invals_enum
            if eqn.primitive == trace_p:
                # outvals = trace_p.bind(*args, **params)
                # copied from dispatch:
                in_tree = params["in_tree"]
                num_consts = params.get("num_consts", 0)
                non_const_tracers = args[num_consts:]
                addr, gen_fn, args = jtu.tree_unflatten(in_tree, non_const_tracers)
                addr = Pytree.tree_const_unwrap(addr)
                if addr == self.eliminate:
                    # TODO: support other distributions
                    if gen_fn == flip:
                        self.support = jnp.array([False, True])
                        p = args[0]
                        logprobs = jnp.log(jnp.array([1 - p, p]))
                        v = self.support
                        outvals = [v]
                        outvals_enum = [True]
                    else:
                        raise NotImplementedError(f"Variable elimination: unsupported primitive {gen_fn}")
                else:
                    if any(args_enumerated):
                        in_axes = [0 if e else None for e in args_enumerated]
                        # TODO: this check should also depend on information flow
                        if isinstance(gen_fn, ExactDensity):
                            mixed = gen_fn.mixture(in_axes=in_axes)
                            v = trace(addr, mixed, (logprobs, *args))
                            outval_enum = False
                        else:
                            v = jax.vmap(lambda *args: trace(addr, gen_fn, tuple(args)), in_axes=in_axes)(*args)
                            outval_enum = True
                    else:
                        v = trace(addr, gen_fn, tuple(args))
                        outval_enum = False
                    outvals = jtu.tree_leaves(v)
                    outvals_enum = [outval_enum for _ in outvals]
            else:
                if any(invals_enum):
                    in_axes = [0 if e else None for e in invals_enum]
                    outvals = jax.vmap(lambda *args: eqn.primitive.bind(*args, **params), in_axes=in_axes)(*invals)
                    if eqn.primitive.multiple_results:
                        outvals_enum = [True for _ in outvals]
                    else:
                        outvals = [outvals]
                        outvals_enum = [True]
                else:
                    outvals = eqn.primitive.bind(*args, **params)
                    if eqn.primitive.multiple_results:
                        outvals_enum = [False for _ in outvals]
                    else:
                        outvals = [outvals]
                        outvals_enum = [False]
            jax_util.safe_map(env.write, eqn.outvars, outvals)
            jax_util.safe_map(env_enum.write, eqn.outvars, outvals_enum)

        return jax_util.safe_map(env.read, jaxpr.outvars), jax_util.safe_map(enum_read(env_enum), jaxpr.outvars)

    def run_interpreter(self, fn, *args, **kwargs):
        def _inner(*args):
            return fn(*args, **kwargs)

        closed_jaxpr, (flat_args, _, out_tree) = stage(_inner)(*args)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        flat_out, out_enumerated = self._eval_jaxpr_forward(
            jaxpr,
            consts,
            flat_args,
            [False for _ in flat_args],
        )
        assert not any(out_enumerated), "Variable elimination: enumerated variables in output"
        return jtu.tree_unflatten(out_tree(), flat_out)


def var_elim_transform(source_fn, eliminate: StaticAddress, support=None):
    @Pytree.partial()
    def wrapper(*args):
        interpreter = VariableEliminationInterpreter(eliminate, support)
        return interpreter.run_interpreter(
            source_fn,
            *args,
        )

    return wrapper

def var_elim(fn: StaticGenerativeFunction[R], eliminate: StaticAddress, support=[0,1]) -> StaticGenerativeFunction[R]:
    elim_gf = var_elim_transform(fn.source, eliminate, support)
    return StaticGenerativeFunction(elim_gf)
