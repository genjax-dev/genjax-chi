# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
# pyright: reportUnusedExpression=false

# %% [markdown]
# # Gradient Ascent Visualization
#
# This notebook demonstrates gradient ascent on a probabilistic function using JAX. We'll visualize:
# 1. The function and its mathematical representation
# 2. The JAX implementation
# 3. The gradient ascent process
# 4. The evolution of gradients and function values over time

# %%
# Import and constants
import genstudio.plot as Plot
import jax
import jax.numpy as jnp
from genstudio.plot import js

# Define the function and its mathematical representation
math_representation = r"""
f(\theta) = \mathbb{E}_{x\sim P(\theta)}[x] = \int_{\mathbb{R}}\left[\theta\frac{1}{\sqrt{2\pi \sigma^2}}e^{\left(\frac{x}{\sigma}\right)^2} + (1-\theta)\frac{1}{\sqrt{2\pi \sigma^2}}e^{\left(\frac{x-\frac{1}{3}\theta}{\sigma}\right)^2}\right]dx =\frac{\theta-\theta^2}{3}
"""

# JAX implementation
jax_code = """
# Approximate integral
expected_val = lambda g: jnp.mean(
    jax.vmap(lambda key: g(key, theta, sigma))
    (jax.random.split(jax.random.key(0), 10000)))

g = expected_val(noisy_jax_model)

# Gradient of approximation
h = jax.grad(g)
"""

# Constants
EPOCHS = 100
LEARNING_RATE = 0.1
INITIAL_THETA = 0.2
SIGMA = 0.05
VIZ_HEIGHT = 300
COMPARISON_HEIGHT = 200
NUM_SAMPLES = 10000


# %%
# Define the function and its gradient
def noisy_jax_model(key, theta, sigma):
    b = jax.random.bernoulli(key, theta)
    x = jax.random.normal(key)
    return jax.lax.cond(
        b,
        lambda theta: x * sigma * theta,
        lambda theta: x * sigma + theta / 3,
        theta,
    )


def expected_value(f, key, theta, sigma, num_samples=NUM_SAMPLES):
    keys = jax.random.split(key, num_samples)
    vals = jax.vmap(lambda key: f(key, theta, sigma))(keys)
    return jnp.mean(vals)


# Generate points for plotting
thetas = jnp.linspace(0, 1, 100)
key = jax.random.key(314159)
values = jax.vmap(lambda theta: expected_value(noisy_jax_model, key, theta, SIGMA))(
    thetas
)


# Run gradient ascent
def run_gradient_ascent(initial_theta, epochs, learning_rate):
    theta = initial_theta
    thetas = []
    values = []
    gradients = []
    current_key = key

    for _ in range(epochs):
        # Compute gradient using finite differences
        current_key, subkey1, subkey2 = jax.random.split(current_key, 3)
        eps = 0.001
        y1 = expected_value(noisy_jax_model, subkey1, theta + eps / 2, SIGMA)
        y2 = expected_value(noisy_jax_model, subkey2, theta - eps / 2, SIGMA)
        grad = (y1 - y2) / eps

        theta = theta + learning_rate * grad  # Note: using +grad for ascent
        theta = jnp.clip(theta, 0.0, 1.0)  # Keep theta in valid range

        current_key, subkey = jax.random.split(current_key)
        value = expected_value(noisy_jax_model, subkey, theta, SIGMA)

        thetas.append(theta)
        values.append(value)
        gradients.append(grad)

    return jnp.array(thetas), jnp.array(values), jnp.array(gradients)


# Run the optimization
thetas_opt, values_opt, gradients_opt = run_gradient_ascent(
    INITIAL_THETA, EPOCHS, LEARNING_RATE
)


# %%
# Create the first visualization (static)
def render_function_visualization():
    # Plot options
    plot_options = Plot.new(
        Plot.color_legend(),
        {"x": {"label": "θ"}, "y": {"label": "f(θ)"}},
        Plot.aspect_ratio(1),
        Plot.grid(),
        {"width": 400, "height": 300},
    )

    # Function plot
    function_plot = (
        Plot.line(
            {"x": thetas, "y": values},
            stroke=Plot.constantly("f(θ)"),
            strokeWidth=2,
        )
        + plot_options
        + Plot.title("Expected Value Plot")
    )

    return Plot.html([
        "div.flex.flex-row.gap-4.p-4",
        [
            "div.flex-1",
            function_plot,
        ],
        [
            "div.flex-1.flex.flex-col.gap-4",
            [
                "div",
                ["h3.font-bold", "Mathematical Representation"],
                ["div.p-2.bg-gray-100.rounded", Plot.katex(math_representation)],
            ],
            [
                "div",
                ["h3.font-bold", "JAX Implementation"],
                [
                    "pre.p-2.bg-gray-100.rounded.font-mono.text-sm",
                    jax_code,
                ],
            ],
        ],
    ])


# Create the second visualization (animated)
def render_gradient_ascent_animation():
    # Plot options
    plot_options = Plot.new(
        Plot.color_legend(),
        {"x": {"label": "θ"}, "y": {"label": "f(θ)"}},
        Plot.aspect_ratio(1),
        Plot.grid(),
        {"width": 400, "height": VIZ_HEIGHT},
    )

    # Gradient ascent path
    path_plot = (
        Plot.line(
            {"x": thetas, "y": values},
            stroke=Plot.constantly("f(θ)"),
            strokeWidth=2,
        )
        + Plot.dot(
            js("$state.gradients.slice(0, $state.frame+1)"),
            x="0",
            y="1",
            fill=js("(_, i) => d3.interpolateHsl('red', 'green')(i/EPOCHS)"),
            r=4,
        )
        + plot_options
        + Plot.title("Gradient Ascent Path")
    )

    return Plot.initialState({
        "frame": 0,
        "gradients": jnp.stack([thetas_opt, values_opt, gradients_opt], axis=1),
    }) | Plot.html([
        "div.flex.flex-col.gap-4.mb-8",
        [
            "div.flex.flex-row.gap-4.p-4",
            [
                "div.flex-1.flex.flex-col.gap-4",
                path_plot,
            ],
        ],
        Plot.Slider(
            key="frame",
            init=0,
            range=[0, EPOCHS - 1],
            step=1,
            fps=30,
            label="Iteration:",
        ),
    ])


def render_gradients_animation():
    # Plot options
    plot_options = Plot.new(
        Plot.color_legend(),
        Plot.grid(),
        {"width": 400, "height": COMPARISON_HEIGHT},
    )

    # Gradients over time
    gradients_plot = (
        Plot.line(
            js("$state.gradients.slice(0, $state.frame+1)"),
            x=Plot.index(),
            y="2",
            stroke=Plot.constantly("Gradient"),
            strokeWidth=2,
        )
        + {"x": {"label": "Iteration"}, "y": {"label": "Gradient"}}
        + Plot.title("Gradients Over Time")
        + Plot.domainX([0, EPOCHS])
        + plot_options
    )

    # Function values over time
    values_plot = (
        Plot.line(
            js("$state.gradients.slice(0, $state.frame+1)"),
            x=Plot.index(),
            y="1",
            stroke=Plot.constantly("Function Value"),
            strokeWidth=2,
        )
        + {"x": {"label": "Iteration"}, "y": {"label": "f(θ)"}}
        + Plot.title("Function Values Over Time")
        + Plot.domainX([0, EPOCHS])
        + plot_options
    )

    return Plot.initialState({
        "frame": 0,
        "gradients": jnp.stack([thetas_opt, values_opt, gradients_opt], axis=1),
    }) | Plot.html([
        "div.flex.flex-col.gap-4",
        [
            "div.flex.flex-row.gap-4.p-4",
            [
                "div.flex-1.flex.flex-col.gap-4",
                gradients_plot,
            ],
            [
                "div.flex-1.flex.flex-col.gap-4",
                values_plot,
            ],
        ],
        Plot.Slider(
            key="frame",
            init=0,
            range=[0, EPOCHS - 1],
            step=1,
            fps=30,
            label="Iteration:",
        ),
    ])


# Render both visualizations
render_function_visualization()
# render_gradients_animation()
# %%
