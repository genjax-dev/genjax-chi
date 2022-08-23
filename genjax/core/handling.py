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

import jax.core as jc
from jax.util import safe_map, safe_zip
from typing import Any, Dict, Sequence, Callable

#####
# Effect handler
#####


class Handler:
    """
    A handler dispatchs a `jc.Primitive` - and must provide
    a `Callable` with signature `def (name_of_primitive)(continuation, *args)`
    where `*args` must match the `jc.Primitive` declaration signature.
    """

    handles: Sequence[jc.Primitive]
    callable: Callable

    def __init__(self, handles: Sequence[jc.Primitive], callable: Callable):
        self.handles = handles
        self.callable = callable


#####
# Effect-handling interpreter
#####

map = safe_map
zip = safe_zip


def eval_jaxpr_handler(
    handler_stack: Sequence[Handler], jaxpr: jc.Jaxpr, consts, *args
):
    """
    This is an interpreter which is parametrized by a handler stack.
    The handler stack is consulted when a `jc.Primitive` with a `must_handle`
    attribute is encountered.

    This interpreter should always be staged out onto a `Jaxpr`
    - so that handling primitives is a zero runtime cost process.
    """

    env: Dict[jc.Var, Any] = {}

    def write(v, val):
        env[v] = val

    map(write, jaxpr.constvars, consts)

    # This is the recursion that replaces the main loop in the original
    # `eval_jaxpr`.
    def eval_jaxpr_recurse(eqns, env, invars, args):
        # The handler could call the continuation multiple times so we
        # we need this function to be somewhat pure. We copy `env` to
        # ensure it isn't mutated.
        env = env.copy()

        def read(v):
            if isinstance(v, jc.Literal):
                return v.val
            else:
                return env[v]

        def write(v, val):
            env[v] = val

        map(write, invars, args)

        if eqns:
            eqn = eqns[0]

            # Here's where we encode `prim` and `addr` for `trace`.
            kwargs = eqn.params

            in_vals = map(read, eqn.invars)
            in_vals = list(in_vals)
            subfuns, params = eqn.primitive.get_bind_params(eqn.params)
            if hasattr(eqn.primitive, "must_handle"):
                args = subfuns + in_vals
                # This definition "reifies" the remainder of the evaluation
                # loop so it can be explicitly passed to the handler.

                def continuation(*args):
                    return eval_jaxpr_recurse(
                        eqns[1:], env, eqn.outvars, [*args]
                    )

                for handler in reversed(handler_stack):
                    if eqn.primitive in handler.handles:
                        # The handler must provide a method with
                        # the name of the primitive.
                        callable = getattr(handler, repr(eqn.primitive))
                        return callable(continuation, *args, **kwargs)

            ans = eqn.primitive.bind(*(subfuns + in_vals), **params)
            if not eqn.primitive.multiple_results:
                ans = [ans]

            return eval_jaxpr_recurse(eqns[1:], env, eqn.outvars, ans)
        else:
            return map(read, jaxpr.outvars)

    return eval_jaxpr_recurse(jaxpr.eqns, env, jaxpr.invars, args)
