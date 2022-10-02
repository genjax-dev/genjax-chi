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
import jax.numpy as jnp
import numpy as np
import pandas as pd
import genjax
from model_config import hidden_markov_model
from inference_config import (
    meta_initial_position,
    hmm_meta_next_target,
    transition_proposal,
    prior_proposal,
)
import genjax.experimental.prox as prox
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import ptitprince as pt
from rich.progress import track


# Global setup.
key = jax.random.PRNGKey(314159)
num_steps = 50
config = genjax.DiscreteHMMConfiguration.new(50, 2, 1, 0.4, 0.1)

sns.set()

SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Set pretty printing + tracebacks.
console = genjax.go_pretty()


@genjax.gen(
    prox.ChoiceMapDistribution,
    selection=genjax.AllSelection(),
)
def exact_hmm_posterior(key, target):
    config = target.args[1]
    observations = target.constraints["z", "observation"]
    key, v = genjax.trace(("z", "latent"), genjax.DiscreteHMM)(
        key, (config, observations)
    )
    return (key,)


inf_selection = genjax.Selection([("z", "latent")])
key, bounds, d = jax.jit(
    genjax.iee(
        hidden_markov_model,
        exact_hmm_posterior,
        inf_selection,
        2000,
        1,
    )
)(key, (num_steps, config))
console.print(bounds)
