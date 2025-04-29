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
# ## Differentiating probabilistic programs
# ### Visualizing gradients and function values over time
# %%
# Import and constants

from functools import partial

import genstudio.plot as Plot
import jax
import jax.numpy as jnp
from genstudio.plot import js

# Constants
key = jax.random.key(314159)
EPOCHS = 400
default_sigma = 0.05
default_initial_val = 0.2
learning_rate = 0.01
default_num_samples_estimate_expected_vals = 100000
default_difference = 0.001
thetas = jnp.arange(0.0, 1.0, 0.0005)


# Model
def noisy_jax_model(key, theta, sigma):
    b = jax.random.bernoulli(key, theta)
    x = jax.random.normal(key)
    return jax.lax.cond(
        b,
        lambda theta: x * sigma * theta,
        lambda theta: x * sigma + theta / 3,
        theta,
    )


def make_samples(f, key, thetas, sigma):
    return jax.vmap(f, in_axes=(0, 0, None))(
        jax.random.split(key, len(thetas)), thetas, sigma
    )


@partial(jax.jit, static_argnames=["f", "num_samples", "epsilon"])
def grad_finite_difference(
    f,
    key,
    theta,
    sigma,
    num_samples=default_num_samples_estimate_expected_vals,
    epsilon=default_difference,
):
    key, subkey = jax.random.split(key)
    y_1 = expected_val(f, key, theta + epsilon / 2, sigma, num_samples=num_samples)
    y_2 = expected_val(f, subkey, theta - epsilon / 2, sigma, num_samples=num_samples)
    grad_estimate = (y_1 - y_2) / epsilon
    return grad_estimate


@partial(jax.jit, static_argnames=["f", "num_samples"])
def expected_val(
    f, key, theta, sigma, num_samples=default_num_samples_estimate_expected_vals
):
    keys = jax.random.split(key, num_samples)
    vals = jax.vmap(lambda key: f(key, theta, sigma))(keys)
    return jnp.mean(vals)


@partial(jax.jit, static_argnames=["f"])
def _compute_jax_step(f, key, current_theta, sigma):
    key, subkey = jax.random.split(key)
    gradient = grad_finite_difference(f, subkey, current_theta, sigma)
    key, subkey = jax.random.split(key)
    expected = expected_val(f, subkey, current_theta, sigma)
    return key, current_theta, expected, gradient


@partial(jax.jit, static_argnames=["f"])
def compute_jax_vals(f, key, initial_theta, sigma):
    def body_fun(carry, _):
        key, current_theta = carry
        key, current_theta, expected, gradient = _compute_jax_step(
            f, key, current_theta, sigma
        )
        new_theta = jnp.clip(current_theta + learning_rate * gradient, 0.0, 1.0)
        return (key, new_theta), jnp.array([current_theta, expected, gradient])

    _, out = jax.lax.scan(body_fun, (key, initial_theta), None, length=EPOCHS)
    return out


# Visualization settings
VIZ_HEIGHT = 300
COMPARISON_HEIGHT = 150
ANIMATION_STEP = 4

plot_options = Plot.new(
    Plot.color_legend(),
    {"x": {"label": "θ"}, "y": {"label": "y"}},
    Plot.aspect_ratio(1),
    Plot.grid(),
    {"width": 400, "height": VIZ_HEIGHT},
    Plot.domainY([-0.1, 0.4]),
)

samples_color_map = Plot.color_map({"Samples": "rgba(0, 128, 128, 0.5)"})


def make_samples_plot(thetas, samples):
    return (
        Plot.dot({"x": thetas, "y": samples}, fill=Plot.constantly("Samples"), r=2)
        + samples_color_map
        + plot_options
        + Plot.clip()
    )


def plot_tangents(gradients_id):
    tangents_plots = Plot.new(
        Plot.aspectRatio(0.5),
        {"width": 400, "height": VIZ_HEIGHT},
    )
    color = "blue" if gradients_id == "ADEV" else "orange"

    orange_to_red_plot = Plot.dot(
        js(f"$state.{gradients_id}_gradients"),
        x="0",
        y="1",
        fill=js(
            f"""(_, i) => d3.interpolateHsl('transparent', '{color}')(i/{EPOCHS})"""
        ),
        filter=(js("(d, i) => i <= $state.frame")),
    )

    tangents_plots += orange_to_red_plot

    tangents_plots += Plot.line(
        js(f"""$state.{gradients_id}_gradients.flatMap(([theta, expected_val, slope], i) => {{
                    const y_intercept = expected_val - slope * theta
                    return [[0, y_intercept, i], [1, slope + y_intercept, i]]
                }})
                """),
        z="2",
        stroke=Plot.constantly(f"{gradients_id} Tangent"),
        opacity=js("(data) => data[2] === $state.frame ? 1 : 0.5"),
        strokeWidth=js("(data) => data[2] === $state.frame ? 3 : 1"),
        filter=js(f"""(data) => {{
            const index = data[2];
            if (index === $state.frame) return true;
            if (index < $state.frame) {{
                const step = Math.floor({EPOCHS} / 10);
                return (index % step === 0);
            }}
            return false;
        }}"""),
    )

    color_map = (
        {
            "Values over time": "blue",
        }
        if gradients_id == "JAX"
        else {
            "Tangent: GenJAX": "blue",
        }
    )

    return Plot.new(
        tangents_plots,
        Plot.title(f"{gradients_id} Gradient Estimates"),
        Plot.color_map(color_map),
    )


def render_plot():
    currentKey = key

    def computeState(val, sigma):
        jax_key, samples_key = jax.random.split(currentKey, num=2)
        return {
            "JAX_gradients": compute_jax_vals(noisy_jax_model, jax_key, val, sigma),
            "samples": make_samples(noisy_jax_model, samples_key, thetas, sigma),
            "val": val,
            "sigma": sigma,
            "frame": 0,
        }

    initialState = Plot.initialState({
        "samples": make_samples(noisy_jax_model, key, thetas, default_sigma),
        "thetas": thetas,
        "JAX_gradients": compute_jax_vals(
            noisy_jax_model, key, default_initial_val, default_sigma
        ),
        "val": default_initial_val,
        "sigma": default_sigma,
        "frame": 0,
    })

    samples_plot = make_samples_plot(thetas, js("$state.samples"))
    jax_tangents_plot = samples_plot + plot_tangents("JAX")

    comparison_plot = (
        Plot.line(
            js("$state.JAX_gradients.slice(0, $state.frame+1)"),
            x=Plot.index(),
            y="2",
            stroke=Plot.constantly("Gradients: JAX"),
        )
        + {"x": {"label": "Iteration"}, "y": {"label": "y"}}
        + Plot.domainX([0, EPOCHS])
        + Plot.title("Computed gradients over time")
        + Plot.color_legend()
        + {"width": 400, "height": COMPARISON_HEIGHT}
    )

    optimization_plot = Plot.new(
        Plot.line(
            js("$state.JAX_gradients"),
            x=Plot.index(),
            y="1",
            stroke=Plot.constantly("Gradient ascent: JAX"),
            filter=js("(d, i) => i <= $state.frame"),
        )
        + {
            "x": {"label": "Iteration"},
            "y": {"label": "Expected Value"},
        }
        + Plot.domainX([0, EPOCHS])
        + Plot.title("Maximization of the expected value")
        + Plot.color_legend()
        + {"width": 400, "height": COMPARISON_HEIGHT}
    )

    frame_slider = Plot.Slider(
        key="frame",
        init=0,
        range=[0, EPOCHS],
        step=ANIMATION_STEP,
        fps=30,
        label="Iteration:",
    )

    return (
        initialState
        | jax_tangents_plot
        | Plot.html([
            "div.grid.grid-cols-2.gap-4",
            comparison_plot,
            optimization_plot,
        ])
        | frame_slider
    )


render_plot()
# %%
