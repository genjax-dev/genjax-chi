# %%
# ---
# title: Visualizing SMC
# subtitle: Designing and visualizing custom proposals
# ---

# %%
import genstudio.plot as Plot
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from scipy import ndimage

import genjax
from genjax import ChoiceMapBuilder as C
from genjax import Pytree, gen, pretty
from genjax._src.generative_functions.distributions.distribution import Distribution
from genjax.typing import FloatArray, PRNGKey

pretty()
key = jax.random.PRNGKey(0)

# %% [markdown]
# Here's a simple image representing a black and white version of the GenJAX logo, along with its bold outline.


# %%
def image_to_bw(image):
    """Convert image to black and white."""
    im_array = np.array(image)
    im_mask = np.logical_not(np.amax(im_array[:, :, :2], 2) < 230)
    return im_mask.astype(float)


def save_image(image, file_path):
    """Save the image to a file."""
    Image.fromarray((image * 255).astype(np.uint8)).save(file_path)


def detect_edges(image):
    """Detect edges in the image using gradient-based edge detection."""
    dx = np.diff(image, axis=1, prepend=image[:, [0]])
    dy = np.diff(image, axis=0, prepend=image[[0], :])
    edges = np.sqrt(dx**2 + dy**2)
    return (edges > 0.1).astype(float)


def create_bold_outline(edges):
    """Create a bold outline from the detected edges."""
    outline = np.ones_like(edges)
    outline[edges > 0] = 0
    outline = ndimage.binary_dilation(1 - outline, iterations=2)
    return 1 - outline.astype(float)


bw_image = image_to_bw(Image.open("../../../docs/assets/img/logo.png").convert("RGB"))
height, width = bw_image.shape

# Save the black and white image
save_image(bw_image, "../../../docs/assets/img/logo-bw.png")

# Save the bold outline image
save_image(
    create_bold_outline(detect_edges(bw_image)),
    "../../../docs/assets/img/logo-bold-outline.png",
)

base_plot = Plot.new(
    Plot.aspectRatio(1),
    Plot.hideAxis(),
    Plot.domain([0, width], [0, height]),
    {"y": {"reverse": True}},
)

logo_plot = base_plot + Plot.img(
    ["docs/assets/img/logo-bold-outline.png"],
    x=0,
    y=height,
    width=width,
    height=-height,
    src=Plot.identity,
)
logo_plot

# %% [markdown]
# We can see this as a uniform distribution on the black part of the image.
# Let's write a GenJAX model that captures this idea.


# %%
# Define the model that generated the image
@Pytree.dataclass
class Logo(Distribution):
    image: FloatArray = Pytree.static()
    threshold: float = Pytree.static(default=1e2)

    def log_likelihood(self, x, y, height, width, temperature):
        floor_x, floor_y = jnp.floor(x), jnp.floor(y)
        floor_x, floor_y = (
            jnp.astype(floor_x, jnp.int32),
            jnp.astype(floor_y, jnp.int32),
        )
        out_of_bounds = (
            (floor_x < 0) | (floor_x >= width) | (floor_y < 0) | (floor_y >= height)
        )
        value = 1.0 / (height * width) - temperature * jax.lax.cond(
            out_of_bounds,
            lambda *_: -self.threshold,
            lambda arg: self.threshold * (self.image[arg[1], arg[0]] == 0),
            operand=(floor_x, floor_y),
        )
        return value

    def random_weighted(self, key: PRNGKey, height, width, temperature):
        key, subkey = jax.random.split(key)
        x = jax.random.uniform(key, minval=0, maxval=width)
        y = jax.random.uniform(subkey, minval=0, maxval=height)
        logpdf = self.log_likelihood(x, y, height, width, temperature)
        return -logpdf, (x, y)

    def estimate_logpdf(self, key: PRNGKey, z, height, width, temperature):
        x, y = z
        return self.log_likelihood(x, y, height, width, temperature)


im_jax = jnp.array(bw_image)
logo = Logo(image=im_jax)


# %%
@gen
def model(height, width, temperature):
    z = logo(height, width, temperature) @ "z"
    return z


# Testing
key, subkey = jax.random.split(key)
model.simulate(subkey, (height, width, 0.0))

# %% [markdown]
# Now for inference, we will use SMC with a variety of custom proposals. Each intermediate target will be an annealed version of the posterior. We will start from a "high temperature" which we can think of as a more diffuse distribution for which it'll be easier for particles to find a region of decent likelihood, and progressively cool down the process by using smaller temperature parameters in the intermediate targets.

# %% [markdown]
# Let's define the initial proposal:


# %%
@genjax.gen
def sub_proposal():
    x = genjax.uniform(0.0, float(width)) @ "x"
    y = genjax.uniform(0.0, float(height)) @ "y"
    return x, y


@genjax.gen
def init_proposal():
    z = sub_proposal() @ "z"
    return z


# Testing
key, subkey = jax.random.split(key)
jitted_init_proposal = jax.jit(lambda x: init_proposal.simulate(x, ()))
jitted_init_proposal(subkey).get_sample()["z", "x"]

# %% [markdown]
# From just this basic proposal, we can test SIR.

# %%
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, 10000)
trs = jax.vmap(jitted_init_proposal)(keys)

sampled_z = jax.vmap(lambda x: (x.get_choices()["z", "x"], x.get_choices()["z", "y"]))(
    trs
)
proposal_scores = jax.vmap(lambda x: x.get_score())(trs)

jitted_model = model.importance
key, subkey = jax.random.split(key)
jitted_model(subkey, C.n(), (height, width, 0.0))

# %%
# Initial high temperature: the model is uniform
# key is not actually used in jitted_model
key, subkey = jax.random.split(key)
trs, ht_model_scores = jax.jit(
    jax.vmap(lambda x: jitted_model(subkey, C["z"].set(x), (height, width, 0.0)))
)(sampled_z)
ht_importance_scores = ht_model_scores / proposal_scores
key, subkey = jax.random.split(key)
ht_resampled_indices = jax.random.categorical(
    subkey, ht_importance_scores, shape=(10000,)
)
sampled_x, sampled_y = sampled_z
ht_resampled_x = sampled_x[ht_resampled_indices]
ht_resampled_y = sampled_y[ht_resampled_indices]

# %%
# Final low temperature: the model is close to uniform on the logo
# key is not actually used in jitted_model
key, subkey = jax.random.split(key)
trs, lt_model_scores = jax.jit(
    jax.vmap(lambda x: jitted_model(subkey, C["z"].set(x), (height, width, 1.0)))
)(sampled_z)
lt_importance_scores = lt_model_scores / proposal_scores
key, subkey = jax.random.split(key)
lt_resampled_indices = jax.random.categorical(
    subkey, lt_importance_scores, shape=(10000,)
)
sampled_x, sampled_y = sampled_z
lt_resampled_x = sampled_x[lt_resampled_indices]
lt_resampled_y = sampled_y[lt_resampled_indices]

# %% [markdown]
# Plotting the results

# Plot 1: Original image
logo_plot

# Plot 2: Image with originally sampled points

(
    logo_plot
    + Plot.dot({"x": sampled_x, "y": sampled_y}, fill="green", r=2)
    + Plot.subtitle("Sampled points")
)

# %%
# Plot 3: Image with resampled Points: high temperature

(
    logo_plot
    + Plot.dot({"x": ht_resampled_x, "y": ht_resampled_y}, fill="red", r=2, opacity=0.5)
    + Plot.subtitle("Resampled points: high temperature")
)

# %%
# Plot 4: Image with resampled Points: low temperature

(
    logo_plot
    + Plot.dot(
        {"x": lt_resampled_x, "y": lt_resampled_y}, fill="blue", r=2, opacity=0.5
    )
    + Plot.subtitle("Resampled points: low temperature")
)

# %% [markdown]
# Let's try a few intermediate stages between the high and low temperatures.

# %%
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, 10000)
trs = jax.vmap(jitted_init_proposal)(keys)
sampled_z = jax.vmap(lambda x: (x.get_choices()["z", "x"], x.get_choices()["z", "y"]))(
    trs
)
proposal_scores = jax.vmap(lambda x: x.get_score())(trs)

all_sampled_x = []
all_sampled_y = []
number_steps = 10

for tmp in jnp.arange(0.0, 1.0 + 1.0 / number_steps, 1.0 / number_steps):
    key, subkey = jax.random.split(key)
    trs, model_scores = jax.jit(
        jax.vmap(lambda x: jitted_model(subkey, C["z"].set(x), (height, width, tmp)))
    )(sampled_z)
    importance_scores = model_scores / proposal_scores
    key, subkey = jax.random.split(key)
    resampled_indices = jax.random.categorical(subkey, importance_scores, shape=(1000,))
    sampled_x, sampled_y = sampled_z
    resampled_x = sampled_x[resampled_indices]
    resampled_y = sampled_y[resampled_indices]
    all_sampled_x.append(resampled_x)
    all_sampled_y.append(resampled_y)

# %% [markdown]
# Which we can visualize:

# %%
Plot.Frames(
    [
        logo_plot
        + Plot.dot(
            {"x": all_sampled_x[i], "y": all_sampled_y[i]},
            fill="green",
            r=2,
            opacity=0.5,
        )
        + Plot.title(f"Temperature: {i / number_steps:.1f}")
        for i in range(0, number_steps + 1)
    ],
    fps=5,
)

# %% [markdown]
# Let's now evolve the particles over time using SMC.

# %%
number_steps = 20
num_particles = 2500


# v1: random wiggling proposal
@genjax.gen
def proposal1(x, y):
    x = genjax.normal(x, 1.0 * min(height, width) / 10) @ "x"
    y = genjax.normal(y, 1.0 * min(height, width) / 10) @ "y"
    return x, y


# v2: rotating proposal distribution
@genjax.gen
def proposal2(x, y):
    theta = 0.15
    center_x, center_y = width / 2, height / 2
    aspect_ratio = height / width
    x_scaled = (x - center_x) * aspect_ratio
    y_scaled = y - center_y
    x_rotated = x_scaled * jnp.cos(theta) - y_scaled * jnp.sin(theta)
    y_rotated = x_scaled * jnp.sin(theta) + y_scaled * jnp.cos(theta) + center_y
    x_rotated = x_rotated / aspect_ratio + center_x

    x = genjax.normal(x_rotated, 1.0 * min(height, width) / 10) @ "x"
    y = genjax.normal(y_rotated, 1.0 * min(height, width) / 10) @ "y"
    return x, y


# v3: spiraling inwards proposal distribution
@genjax.gen
def proposal3(x, y):
    inward_coeff = 0.9
    theta = 0.15
    center_x, center_y = width / 2, height / 2
    aspect_ratio = height / width
    x_scaled = (x - center_x) * aspect_ratio
    y_scaled = y - center_y
    x_rotated = (x_scaled * jnp.cos(theta) - y_scaled * jnp.sin(theta)) * inward_coeff
    y_rotated = x_scaled * jnp.sin(theta) + y_scaled * jnp.cos(theta) + center_y
    x_rotated = (x_rotated / aspect_ratio + center_x) * inward_coeff

    x = genjax.normal(x_rotated, 1.0 * min(height, width) / 10) @ "x"
    y = genjax.normal(y_rotated, 1.0 * min(height, width) / 10) @ "y"
    return x, y


all_sampled_x = []
all_sampled_y = []

# samples and scores from proposal
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num_particles)
trs = jax.vmap(jitted_init_proposal)(keys)
sampled_z = jax.vmap(lambda x: (x.get_choices()["z", "x"], x.get_choices()["z", "y"]))(
    trs
)
proposal_scores = jax.vmap(lambda x: x.get_score())(trs)

# scores from model at high temperature
key, subkey = jax.random.split(key)
trs, model_scores = jax.vmap(
    lambda x: jitted_model(subkey, C["z"].set(x), (height, width, 0.0))
)(sampled_z)

# importance scores and resampling
importance_scores = model_scores / proposal_scores
key, subkey = jax.random.split(key)
resampled_indices = jax.random.categorical(
    subkey, importance_scores, shape=(num_particles,)
)
sampled_x, sampled_y = sampled_z
resampled_x = sampled_x[resampled_indices]
resampled_y = sampled_y[resampled_indices]

resampled_z = (resampled_x, resampled_y)
# store results for later visualization
all_sampled_x.append(resampled_x)
all_sampled_y.append(resampled_y)

# jitting the step_proposal
jitted = jax.jit(jax.vmap(proposal1.simulate, in_axes=(0, 0)))

for tmp in jnp.arange(0.0, 1.0 + 1.0 / number_steps, 1.0 / number_steps):
    # samples and scores from proposal kernels using the chm from previous step
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num_particles)
    trs = jitted(keys, (resampled_z))
    sampled_z = jax.jit(
        jax.vmap(lambda x: (x.get_choices()["x"], x.get_choices()["y"]))
    )(trs)
    proposal_scores = jax.jit(jax.vmap(lambda x: x.get_score()))(trs)

    # scores from model at intermediate temperature
    key, subkey = jax.random.split(key)
    trs, model_scores = jax.jit(
        jax.vmap(lambda x: jitted_model(subkey, C["z"].set(x), (height, width, tmp**2)))
    )(sampled_z)

    # importance scores and resampling
    importance_scores = model_scores / proposal_scores
    key, subkey = jax.random.split(key)
    resampled_indices = jax.random.categorical(
        subkey, importance_scores, shape=(num_particles,)
    )
    sampled_x, sampled_y = sampled_z
    resampled_x = sampled_x[resampled_indices]
    resampled_y = sampled_y[resampled_indices]
    resampled_z = (resampled_x, resampled_y)
    chm = jax.vmap(lambda p: C["z", "x"].set(p))(resampled_x) ^ jax.vmap(
        lambda p: C["z", "y"].set(p)
    )(resampled_y)

    # store results for later visualization
    all_sampled_x.append(resampled_x)
    all_sampled_y.append(resampled_y)

# %% [markdown]
# Visualizing what we got!

Plot.Frames(
    [
        base_plot
        + Plot.dot(
            {"x": all_sampled_x[i], "y": all_sampled_y[i]},
            fill=Plot.js(f"(d, i) => d3.interpolateCividis(1 - {i}/{number_steps})"),
            r=2,
            opacity=0.5,
        )
        + Plot.title(f"Temperature: {i / number_steps:.1f}")
        for i in range(0, number_steps + 1)
    ],
    fps=10,
)

# %% [markdown]
# Adding rejuvenation

# %%
# not sure how to do a non-hacky thing for now
key, unused_key = jax.random.split(key)
proposal_scorer = lambda x, y: proposal1.importance(
    unused_key, C["x"].set(x[0]) ^ C["y"].set(x[1]), y
)
final_model_scorer = lambda x: jitted_model(
    unused_key, C["z"].set(x), (height, width, 1.0)
)


@gen
def boltzman_rejuv(x, y):
    mb_x, mb_y = proposal1(x, y) @ "mb"
    # old_score = final_model_scorer((x,y))/proposal_scorer((x,y))
    # This feels wrong: what should old_score be?
    old_score = 1.0
    _, p_score = final_model_scorer((mb_x, mb_y))
    _, q_score = proposal_scorer((mb_x, mb_y), (x, y))
    new_score = p_score / q_score
    i = genjax.categorical(jnp.array([old_score, new_score])) @ "i"
    return jax.lax.cond(i == 0, lambda: (x, y), lambda: (mb_x, mb_y))


# Test the rejuvenation process
key, subkey = jax.random.split(key)
x, y = (0.5, 0.5)
tr = boltzman_rejuv.simulate(subkey, (x, y))
x, y = tr.get_retval()
print(x, y)

# %% [markdown]
# Now we can add a circular mask for the initial proposal:

# %%
# We need a mask on the image, independent of the inference
# for inference, what we want is a proposal clipped to the mask, which doesn't use masking
# a proper way would be to use meta-inference, but we can just have exponentially decaying probabilities outside the disk, and hope for the best

# %%
# Desired interactions:
# - mask disk centred around mouse cursor. truncated to borders of the image
# - mouse scroll: change radius of mask between 10 and 100 pixel
# - left click: higher temperature (max exp^0)
# - right click: lower temperature (min exp^{-1})
# - SMC running while all of this is happening.
# - may want yet another way to change the number of particles: with 1000 it's hard to see the logo but with 10k it's obvious.
# - the rejuv dynamics could be guided by the mouse cursor moving. so that we can have particle move and bump more against the letters which a human can infer.
# - we could infer the dynamics of the cursor between linear, quadratic, etc. by deterministic regression and propose dynamics according to the chosen one.


# %%


# %%
# TODO: because of the "z" vs "x", "y" distinction in the trace, we'd need some kind of trace translator to tell the algorithm to use the [“z”, (“x”, “y”)] sample as a "z" for the model.
target = genjax.Target(model, (height, width, 0.0), C.n())


# %%
# Method 2: Define the model that generated the image
# It will be a more noisy version of the uniform distribution we just described.
@genjax.gen
def pixel_model(temperature):
    x = genjax.uniform(0.0, 1.0) @ "x"
    obs = genjax.normal(x, 0.01 * exp(-temperature)) @ "obs"
    return obs


model = pixel_model.vmap(in_axes=(0,)).vmap(in_axes=(0,))

chm = C[jnp.arange(height), jnp.indices((height, width))[1], "obs"].set(bw_image)


# %%
# Define the prior model
@genjax.gen
def prior_model(height, width):
    # Particles are uniformly randomly initialized within the image frame
    x = genjax.uniform(0.0, height) @ "x"
    y = genjax.uniform(0.0, width) @ "y"

    obs_x = genjax.normal(x, 1.0) @ "obs_x"
    obs_y = genjax.normal(y, 1.0) @ "obs_y"
    return x, y


jax_im = jnp.array(bw_image.astype(float))
n_samples = 5000
batched_prior_model = prior_model.repeat(n=n_samples)

key, subkey = jax.random.split(key)
tr = batched_prior_model.simulate(subkey, (float(height), float(width)))
xs_init, ys_init = tr.get_choices()[..., "x"], tr.get_choices()[..., "y"]
zs_init = np.stack([xs_init, ys_init], axis=1)
