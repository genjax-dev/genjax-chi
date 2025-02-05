# %%
import jax
import jax.numpy as jnp

import genjax
from genjax import ChoiceMapBuilder as C

key = jax.random.key(0)


# %% [markdown]
#
# This notebook shows how to do inference on a discrete Hidden Markov Model.
# We consider our state to be an integer in the range $[0\ldots N)$, and the transitions
# to be modeled by a constant stochastic matrix $X_{ij}$ containing the probability
# of the state transition $i\rightarrow j$. We further suppose that all we can
# observe directly are outputs, where the output from a hidden state is governed
# by another stochastic matrix $Y_{jk}$, which represents the probability that
# output symbol $k$ will follow from state $j$. There may be more or fewer output
# symbols than states, so while $X$ is square, $Y$ may not be.
# %%
class HiddenMarkovModel:
    def __init__(self, key, N, M):
        # Choose transition and output matrices
        self.X = self.stochastic_matrix(key, (N, N))
        self.Y = self.stochastic_matrix(key, (N, M))
        # Convert them to logit, for the convenience of `genjax.categorical`
        self.lX = jnp.log(self.X / (1.0 - self.X))
        self.lY = jnp.log(self.Y / (1.0 - self.Y))

        @genjax.gen
        def step(state, control):
            x1 = genjax.categorical(self.lX[state]) @ "x"
            y1 = genjax.categorical(self.lY[x1]) @ "y"
            return x1, y1

        self.step = step
        self.model = self.step.scan()

    def stochastic_matrix(self, key, shape):
        """Return a matrix of non-negative values whose rows sum to 1."""
        a = jax.random.uniform(key=key, shape=shape)
        return a / jnp.sum(a, axis=1, keepdims=True)

    def run(self, key, x0, n):
        return self.model.simulate(key, (jnp.array(x0), jnp.arange(n, dtype=float)))


# %%
key, sub_key = jax.random.split(key)
hmm = HiddenMarkovModel(sub_key, 4, 5)
# %%
key, sub_key = jax.random.split(key)
goal_trace = hmm.run(key, 0, 5)
print(f"inner states: {goal_trace.get_choices()['x']}")
print(f"observations: {goal_trace.get_choices()['y']}")
# %% [markdown]
# How can we infer the inner states from the observations?
# We could try the brute force approach of importance sampling.
# %%
observations = C["y"].set(goal_trace.get_choices()["y"])
# %%
key, sub_key = jax.random.split(key)
imp_trace, score = hmm.model.importance(sub_key, observations, (0, jnp.arange(5.0)))

# %%
imp_trace.get_choices()["x"]
# %%
jnp.count_nonzero(imp_trace.get_choices()["x"] - goal_trace.get_choices()["x"])
# Not so good. Try a bunch of samples:
# %%
key, sub_key = jax.random.split(key)
imp_traces, ws = jax.vmap(hmm.model.importance, in_axes=(0, None, None))(
    jax.random.split(sub_key, 50000), observations, (0, jnp.arange(5.0))
)

# %%
winners = jax.random.categorical(key=sub_key, logits=ws, shape=(5,))
winner, imp_traces.get_choices()["x"][winners]
# %%
# None of the winners is exactly right, but it does seem like the choicemap
# has had some influence. We can take more winners, and form the "majority opinion:"
more_winners = jax.random.categorical(key=sub_key, logits=ws, shape=(50,))
counts = jax.vmap(lambda v: jnp.bincount(v, length=5))(
    imp_traces.get_choices()["x"][more_winners].T
)
majority = jnp.argmax(counts, axis=1)
majority
# %%
# That doesn't work very well either! Time for the gibbs approach
