# ---
# title: Differentiating probabilistic programs
# subtitle: How to take drastic differentiating measures by differentiating measures
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
# Import and constants
import genstudio.plot as Plot
import jax
import jax.numpy as jnp

from genjax._src.adev.core import Dual, expectation
from genjax._src.adev.primitives import flip_enum, normal_reparam

key = jax.random.key(314159)
EPOCHS = 400
default_sigma = 0.05


# %%
# Model
def noisy_jax_model(key, theta, sigma):
    b = jax.random.bernoulli(key, theta)
    return jax.lax.cond(
        b,
        lambda theta: jax.random.normal(key) * sigma * theta,
        lambda theta: jax.random.normal(key) * sigma + theta / 2,
        theta,
    )


def expected_val(theta):
    return (theta - theta**2) / 2


# %%
# Samples
thetas = jnp.arange(0.0, 1.0, 0.0005)


def make_samples(thetas, sigma):
    global key
    key, subkey = jax.random.split(key)
    return jax.vmap(noisy_jax_model, in_axes=(0, 0, None))(
        jax.random.split(subkey, len(thetas)), thetas, sigma
    )


noisy_samples = make_samples(thetas, default_sigma)

plot_options = Plot.new(
    Plot.color_legend(),
    {"x": {"label": "θ"}, "y": {"label": "y"}},
    Plot.aspect_ratio(1),
    Plot.grid(),
)


def make_samples_plot(thetas, samples):
    return (
        Plot.dot({"x": thetas, "y": samples}, fill=Plot.constantly("Samples"), r=2)
        + Plot.color_map({"Samples": "rgba(0, 128, 128, 0.5)"})
        + plot_options
        + Plot.clip()
    )


samples_plot = make_samples_plot(thetas, noisy_samples)

samples_plot

# %%
# Adding exact expectation
thetas_sparse = jnp.linspace(0.0, 1.0, 20)  # fewer points, for the plot
exact_vals = jax.vmap(expected_val)(thetas_sparse)

expected_value_plot = (
    Plot.line(
        {"x": thetas_sparse, "y": exact_vals},
        strokeWidth=2,
        stroke=Plot.constantly("Expected value"),
        curve="natural",
    )
    + Plot.color_map({"Expected value": "black"})
    + plot_options,
)

samples_plot + expected_value_plot

# %%
# JAX computed exact gradients
grad_exact = jax.jit(jax.grad(expected_val))
theta_tangent_points = [0.1, 0.3, 0.45]

# Optimization on ideal curve
arg = 0.2
vals = []
arg_list = []
for _ in range(EPOCHS):
    grad_val = grad_exact(arg)
    arg_list.append(arg)
    vals.append(expected_val(arg))
    arg = arg + 0.01 * grad_val
    if arg < 0:
        arg = 0
        break
    elif arg > 1:
        arg = 1

(
    Plot.line({"x": list(range(EPOCHS)), "y": vals})
    + {"x": {"label": "Iteration"}, "y": {"label": "y"}}
)

# %%
color1 = "rgba(255,165,0,0.5)"
color2 = "#FB575D"


def tangent_line_plot(theta_tan):
    slope = grad_exact(theta_tan)
    y_intercept = expected_val(theta_tan) - slope * theta_tan
    label = f"Tangent at θ={theta_tan}"

    return Plot.line(
        [[0, y_intercept], [1, slope + y_intercept]],
        stroke=Plot.constantly(label),
    ) + Plot.color_map({
        label: Plot.js(
            f"""d3.interpolateHsl("{color1}", "{color2}")({theta_tan}/{theta_tangent_points[-1]})"""
        )
    })


(
    plot_options
    + [tangent_line_plot(theta_tan) for theta_tan in theta_tangent_points]
    + expected_value_plot
    + Plot.domain([0, 1], [0, 0.4])
    + Plot.title("Expectation curve and its Tangent Lines")
)

# %%
theta_tan = 0.3

slope = grad_exact(theta_tan)
y_intercept = expected_val(theta_tan) - slope * theta_tan

exact_tangent_plot = Plot.line(
    [[0, y_intercept], [1, slope + y_intercept]],
    strokeWidth=2,
    stroke=Plot.constantly("Exact tangent at θ=0.3"),
)


def slope_estimate_plot(slope_est):
    y_intercept = expected_val(theta_tan) - slope_est * theta_tan
    return Plot.line(
        [[0, y_intercept], [1, slope_est + y_intercept]],
        strokeWidth=2,
        stroke=Plot.constantly("Tangent estimate"),
    )


slope_estimates = [slope + i / 20 for i in range(-4, 4)]

(
    samples_plot
    + expected_value_plot
    + [slope_estimate_plot(slope_est) for slope_est in slope_estimates]
    + exact_tangent_plot
    + Plot.title("Expectation curve and Tangent Estimates at θ=0.3")
    + Plot.color_map({
        "Expected value": "blue",
        "Tangent estimate": color1,
        "Exact tangent at θ=0.3": color2,
    })
    + Plot.domain([0, 1], [0, 0.4])
)

# %%
jax_grad = jax.jit(jax.grad(noisy_jax_model, argnums=1))

arg = 0.2
vals = []
grads = []
for _ in range(EPOCHS):
    key, subkey = jax.random.split(key)
    grad_val = jax_grad(subkey, arg, default_sigma)
    arg = arg + 0.01 * grad_val
    vals.append(expected_val(arg))
    grads.append(grad_val)

(
    Plot.line(
        {"x": list(range(EPOCHS)), "y": vals},
        stroke=Plot.constantly("Attempting gradient ascent with JAX"),
    )
    + Plot.title("Maximization of the expected value of a probabilistic function")
    + {"x": {"label": "Iteration"}, "y": {"label": "y"}}
    + Plot.domainX([0, EPOCHS])
    + Plot.color_legend()
)

# %%
theta_tangents = jnp.linspace(0, 1, 20)


def plot_tangents(gradients, title):
    tangents_plots = Plot.new(Plot.aspectRatio(0.5))

    for theta, slope in gradients:
        y_intercept = expected_val(theta) - slope * theta
        tangents_plots += Plot.line(
            [[0, y_intercept], [1, slope + y_intercept]],
            stroke=Plot.js(
                f"""d3.interpolateHsl("{color1}", "{color2}")({theta}/{theta_tangents[-1]})"""
            ),
            opacity=0.75,
        )
    return Plot.new(
        expected_value_plot,
        Plot.domain([0, 1], [0, 0.4]),
        tangents_plots,
        Plot.title(title),
        Plot.color_map({
            f"Tangent at θ={theta_tangents[0]}": color1,
            f"Tangent at θ={theta_tangents[-1]}": color2,
        }),
    )


gradients = []
for theta in theta_tangents:
    key, subkey = jax.random.split(key)
    gradients.append((theta, jax_grad(subkey, theta, default_sigma)))

plot_tangents(gradients, "Expectation curve and JAX-computed tangent estimates")


# %%
@expectation
def flip_approx_loss(theta, sigma):
    b = flip_enum(theta)
    return jax.lax.cond(
        b,
        lambda theta: normal_reparam(0.0, sigma) * theta,
        lambda theta: normal_reparam(theta / 2, sigma),
        theta,
    )


adev_grad = jax.jit(flip_approx_loss.jvp_estimate)


def compute_jax_vals(key, initial_theta, sigma):
    current_theta = initial_theta
    out = []
    for _ in range(EPOCHS):
        key, subkey = jax.random.split(key)
        gradient = jax_grad(subkey, current_theta, sigma)
        out.append((current_theta, expected_val(current_theta), gradient))
        current_theta = current_theta + 0.01 * gradient
    return out


def compute_adev_vals(key, initial_theta, sigma):
    current_theta = initial_theta
    out = []
    for _ in range(EPOCHS):
        key, subkey = jax.random.split(key)
        gradient = adev_grad(
            subkey, (Dual(current_theta, 1.0), Dual(sigma, 0.0))
        ).tangent
        out.append((current_theta, expected_val(current_theta), gradient))
        current_theta = current_theta + 0.01 * gradient
    return out


# %%
def select_evenly_spaced(items, num_samples=5):
    """Select evenly spaced items from a list."""
    if num_samples <= 1:
        return [items[0]]

    result = [items[0]]
    step = (len(items) - 1) / (num_samples - 1)

    for i in range(1, num_samples - 1):
        index = int(i * step)
        result.append(items[index])

    result.append(items[-1])
    return result


INITIAL_VAL = 0.2
SLIDER_STEP = 0.01
ANIMATION_STEP = 4

button_classes = "px-3 py-1 bg-blue-500 text-white text-sm font-medium rounded-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 transition duration-150 ease-in-out"

combined_plot = Plot.new()


def render_combined_plot(current_val, sigma):
    global key
    key, subkey1, subkey2 = jax.random.split(key, num=3)

    def current_state(val, sigma):
        return {
            "jax_gradients": compute_jax_vals(subkey1, val, sigma),
            "adev_gradients": compute_adev_vals(subkey2, val, sigma),
            "frame": 0,
            "current_val": val,
            "sigma": sigma,
            "samples": make_samples(thetas, sigma),
        }

    def recompute(widget, val, sigma):
        widget.update_state(current_state(val, sigma))

    def set_val(widget, new_val):
        nonlocal current_val
        current_val = new_val
        recompute(widget, current_val, sigma)

    def set_sigma(widget, new_sigma):
        nonlocal sigma
        sigma = new_sigma
        recompute(widget, current_val, sigma)

    def refresh(widget):
        global key
        nonlocal subkey1, subkey2
        key, subkey1, subkey2 = jax.random.split(key, num=3)
        recompute(widget, current_val, sigma)

    samples_plot = make_samples_plot(thetas, Plot.js("$state.samples"))

    def plot_tangents(gradients_id, title):
        tangents_plots = Plot.new(Plot.aspectRatio(0.5))

        orange_to_red_plot = Plot.dot(
            Plot.js(f"$state.{gradients_id}"),
            x="0",
            y="1",
            fill=Plot.js(
                f"""(_, i) => d3.interpolateHsl('{color1}', '{color2}')(i/{EPOCHS})"""
            ),
            filter=(Plot.js("(d, i) => i <= $state.frame")),
        )

        tangents_plots += orange_to_red_plot

        tangents_plots += Plot.line(
            Plot.js(f"""$state.{gradients_id}.flatMap(([theta, expected_val, slope], i) => {{
                        const y_intercept = expected_val - slope * theta
                        return [[0, y_intercept, i], [1, slope + y_intercept, i]]
                    }})
                    """),
            z="2",
            stroke=Plot.constantly("Tangent"),
            opacity=Plot.js("(data) => data[2] === $state.frame ? 1 : 0.5"),
            strokeWidth=Plot.js("(data) => data[2] === $state.frame ? 3 : 1"),
            filter=Plot.js(f"""(data) => {{
                const index = data[2];
                if (index === $state.frame) return true;
                if (index < $state.frame) {{
                    const step = Math.floor({EPOCHS} / 10);
                    return (index % step === 0);
                }}
                return false;
            }}"""),
        )

        return Plot.new(
            expected_value_plot,
            Plot.domain([0, 1], [0, 0.4]),
            tangents_plots,
            Plot.title(title),
            Plot.color_map({
                "Samples": "rgba(0, 128, 128, 0.5)",
                "Expected value": "blue",
                "Tangent": "rgba(255, 165, 0, 1)",
            }),
        )

    comparison_plot = (
        Plot.line(
            Plot.js("$state.jax_gradients.slice(0, $state.frame+1)"),
            x=Plot.index(),
            y="2",
            stroke=Plot.constantly("Gradients from JAX"),
        )
        + Plot.line(
            Plot.js("$state.adev_gradients.slice(0, $state.frame+1)"),
            x=Plot.index(),
            y="2",
            stroke=Plot.constantly("Gradients from ADEV"),
        )
        + {"x": {"label": "Iteration"}, "y": {"label": "y"}}
        + Plot.domainX([0, EPOCHS])
        + Plot.title("Comparison of computed gradients JAX vs ADEV")
        + Plot.color_legend()
    )

    optimization_plot = Plot.new(
        Plot.line(
            Plot.js("$state.jax_gradients"),
            x=Plot.index(),
            y="1",
            stroke=Plot.constantly("Gradient ascent with JAX"),
            filter=Plot.js("(d, i) => i <= $state.frame"),
        )
        + Plot.line(
            Plot.js("$state.adev_gradients"),
            x=Plot.index(),
            y="1",
            stroke=Plot.constantly("Gradient ascent with ADEV"),
            filter=Plot.js("(d, i) => i <= $state.frame"),
        )
        + {
            "x": {"label": "Iteration"},
            "y": {"label": "Expected Value"},
        }
        + Plot.domainX([0, EPOCHS])
        + Plot.title("Maximization of the expected value of a probabilistic function")
        + Plot.color_legend()
    )

    jax_tangents_plot = samples_plot + plot_tangents(
        "jax_gradients", "JAX Gradient Estimates"
    )
    adev_tangents_plot = samples_plot + plot_tangents(
        "adev_gradients", "ADEV Gradient Estimates"
    )

    frame_slider = Plot.Slider(
        key="frame", init=0, range=[0, EPOCHS], step=ANIMATION_STEP, fps=30
    )

    initial_val_slider = Plot.html([
        "div",
        {"class": "flex items-center space-x-4 mb-4"},
        [
            "label",
            "Initial Value:",
            [
                "span.font-bold.px-1",
                Plot.js("$state.current_val"),
            ],  # Format to 2 decimal places
            [
                "input",
                {
                    "type": "range",
                    "min": 0,
                    "max": 1,
                    "step": SLIDER_STEP,
                    "defaultValue": INITIAL_VAL,
                    "onChange": lambda e: set_val(e["widget"], float(e["value"])),
                    "class": "w-64 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer ml-2",
                },
            ],
        ],
        [
            "label",
            "Sigma:",
            [
                "span.font-bold.px-1",
                Plot.js("$state.sigma"),
            ],  # Format to 2 decimal places
            [
                "input",
                {
                    "type": "range",
                    "min": 0,
                    "max": 0.2,
                    "step": 0.01,
                    "defaultValue": sigma,
                    "onChange": lambda e: set_sigma(e["widget"], float(e["value"])),
                    "class": "w-64 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer ml-2",
                },
            ],
        ],
        [
            "button",
            {
                "onClick": lambda e: refresh(e["widget"]),
                "class": button_classes,
            },
            "Refresh",
        ],
    ])

    combined_plot.update_state(["frame", "reset", 0])
    combined_plot.reset(
        Plot.initial_state(current_state(current_val, sigma))
        | initial_val_slider
        | jax_tangents_plot & adev_tangents_plot
        | optimization_plot & comparison_plot
        | frame_slider
    )


render_combined_plot(INITIAL_VAL, default_sigma)

combined_plot
