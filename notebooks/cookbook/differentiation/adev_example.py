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
# pyright: reportUnusedExpression=false
# %%
# Import and constants
import genstudio.plot as Plot
import jax
import jax.numpy as jnp
from genstudio.plot import js

from genjax._src.adev.core import Dual, expectation
from genjax._src.adev.primitives import flip_enum, normal_reparam

key = jax.random.key(314159)
EPOCHS = 400
default_sigma = 0.05


# %% [markdown]
# We are often interested in the average returned value of a probabilistic program. For instance, it could be that
# a run of the program represents a run of a simulation of some form, and we would like to maximize the average reward across many simulations (or equivalently minimize a loss).


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


# %% [markdown]
# We can see that the simulation can have two "modes" that split further appart over time.

# %%
# Samples
thetas = jnp.arange(0.0, 1.0, 0.0005)


def make_samples(key, thetas, sigma):
    return jax.vmap(noisy_jax_model, in_axes=(0, 0, None))(
        jax.random.split(key, len(thetas)), thetas, sigma
    )


key, samples_key = jax.random.split(key)
noisy_samples = make_samples(samples_key, thetas, default_sigma)

plot_options = Plot.new(
    Plot.color_legend(),
    {"x": {"label": "θ"}, "y": {"label": "y"}},
    Plot.aspect_ratio(1),
    Plot.grid(),
    Plot.clip(),
)

samples_color_map = Plot.color_map({"Samples": "rgba(0, 128, 128, 0.5)"})


def make_samples_plot(thetas, samples):
    return (
        Plot.dot({"x": thetas, "y": samples}, fill=Plot.constantly("Samples"), r=2)
        + samples_color_map
        + plot_options
        + Plot.clip()
    )


samples_plot = make_samples_plot(thetas, noisy_samples)

samples_plot


# %% [markdown]
# We can also easily imagine a more noisy version of the same idea.


# %%
def more_noisy_jax_model(key, theta, sigma):
    b = jax.random.bernoulli(key, theta)
    return jax.lax.cond(
        b,
        lambda _: jax.random.normal(key) * sigma * theta**2 / 3,
        lambda _: (jax.random.normal(key) * sigma + theta) / 2,
        None,
    )


more_thetas = jnp.arange(0.0, 1.0, 0.0005)
key, *keys = jax.random.split(key, len(more_thetas) + 1)

noisy_sample_plot = (
    Plot.dot(
        {
            "x": more_thetas,
            "y": jax.vmap(more_noisy_jax_model, in_axes=(0, 0, None))(
                jnp.array(keys), more_thetas, default_sigma
            ),
        },
        fill=Plot.constantly("Samples"),
        r=2,
    )
    + samples_color_map
)

noisy_sample_plot + plot_options


# %% [markdown]
# As we can see better on the noisy version, the samples divide into two groups. One tends to go up as theta increases while the other stays relatively stable around 0 with a higher variance. For simplicity of the analysis, in the rest of this notebook we will stick to the simpler first example.

# %% [markdown]
#
# In that simple case, we can compute the exact average value of the random process as a function of $\theta$. We have probability $\theta$ to return $0$ and probablity $1-\theta$ to return $\frac{\theta}{2}$. So overall the expected value is
# $$\theta*0 + (1-\theta)*\frac{\theta}{2} = \frac{\theta-\theta^2}{2}$$

# %% [markdown]
# We can code this and plot the result for comparison.


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

# %% [markdown]
# We can see that the curve in yellow is a perfectly reasonable differentiable function. We can use JAX to compute its derivative (more generally its gradient) at various points.

# %%
grad_exact = jax.jit(jax.grad(expected_val))
theta_tangent_points = [0.1, 0.3, 0.45]


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

# %% [markdown]
# A popular technique from optimization is to use iterative methods such as (stochastic) gradient ascent.
# Starting from any location, say 0.2, we can use JAX to find the maximum of the function.

# %%
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

# %% [markdown]
# We can plot the evolution of the value of the function over the iterations of the algorithms.

# %%
(
    Plot.line({"x": list(range(EPOCHS)), "y": vals})
    + {"x": {"label": "Iteration"}, "y": {"label": "y"}}
)

# %% [markdown]
# We can also directly visualize the points on the curve.

# %%
(
    expected_value_plot
    + Plot.dot(
        {"x": arg_list, "y": vals},
        fill=Plot.js(
            f"""(_, i) => d3.interpolateHsl('{color1}', '{color2}')(i/{len(arg_list)})"""
        ),
    )
    + Plot.subtitle("Gradient descent from start to end")
    + Plot.aspect_ratio(0.25)
    + {"width": 600}
    + Plot.color_map({"start": color1, "end": color2})
)

# %% [markdown]
# We have this in this example that we can compute the average value exactly. But will not be the case in general. One popular technique to approximate an average value is to use Monte Carlo Integration: we sample a bunch from the program and take the average value.
#
# As we use more and more samples we will converge to the correct result by the Central limit theorem.

# %%
number_of_samples = sorted([1, 3, 5, 10, 20, 50, 100, 200, 500, 1000] * 7)
means = []
for n in number_of_samples:
    key, subkey = jax.random.split(key)
    keys = jax.random.split(key, n)
    samples = jax.vmap(noisy_jax_model, in_axes=(0, None, None))(
        keys, 0.3, default_sigma
    )
    mean = jnp.mean(samples)
    means.append(mean)

(
    Plot.dot(
        {"x": number_of_samples, "y": means},
        fill=Plot.js(
            f"""(_, i) => d3.interpolateHsl('{color1}', '{color2}')(i/{len(number_of_samples)})"""
        ),
    )
    + Plot.ruleY(
        [expected_val(0.3)],
        opacity=0.2,
        strokeWidth=2,
        stroke=Plot.constantly("True value"),
    )
    + Plot.color_map({"Mean estimate": color1, "True value": "green"})
    + Plot.color_legend()
    + {"x": {"label": "Number of samples", "type": "log"}, "y": {"label": "y"}}
)

# %% [markdown]
# As we just discussed, most of the time we will not be able to compute the average value and then compute the gradient using JAX. One thing we may want to try is to use JAX on the probabilistic program to get a gradient estimate, and hope that by using more and more samples this will converge to the correct gradient that we could use in optimization. Let's try it in JAX.

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
        "Tangent estimate": color1,
        "Exact tangent at θ=0.3": color2,
    })
    + Plot.domain([0, 1], [0, 0.4])
)

# %%
jax_grad = jax.jit(jax.grad(noisy_jax_model, argnums=1))

arg = 0.2
vals = []
for _ in range(EPOCHS):
    key, subkey = jax.random.split(key)
    grad_val = jax_grad(subkey, arg, default_sigma)
    arg = arg + 0.01 * grad_val
    vals.append(expected_val(arg))

# %% [markdown]
# JAX seems happy to compute something and we can use the iterative technique from before, but let's see if we managed to minimize the function.

# %%
(
    Plot.line(
        {"x": list(range(EPOCHS)), "y": vals},
        stroke=Plot.constantly("Attempting gradient ascent with JAX"),
    )
    + {"x": {"label": "Iteration"}, "y": {"label": "y"}}
    + Plot.domainX([0, EPOCHS])
    + Plot.title("Maximization of the expected value of a probabilistic function")
    + Plot.color_legend()
)

# %% [markdown]
# Woops! We seemed to start ok but then for some reason the curve goes back down and we end up minimizing the function instead of maximizing it!
#
# The reason is that we failed to account from the change of contribution of the coin flip from  `bernoulli` in the differentiation process, and we will come back to this in more details in follow up notebooks.
#
# For now, let's just get a sense of what the gradient estimates computed by JAX look like.


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


# %% [markdown]
# Ouch. They do not look even remotely close to valid gradient estimates.

# %% [markdown]
# ADEV is a new algorithm that computes correct gradient estimates of expectations of probabilistic programs. It  accounts for the change to the expectation coming from a change to the randomness present in the expectation.
#
# GenJAX implements ADEV. Slightly rewriting the example from above using GenJAX, we can see how different the behaviour of the optimization process with the corrected gradient estimates is.


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


button_classes = (
    "px-3 py-1 bg-blue-500 text-white text-sm font-medium rounded-md hover:bg-blue-600"
)


def input_slider(label, value, min, max, step, on_change, default):
    return [
        "label.flex.flex-col.gap-2",
        ["div", label, ["span.font-bold.px-1", value]],
        [
            "input",
            {
                "type": "range",
                "min": min,
                "max": max,
                "step": step,
                "defaultValue": default,
                "onChange": on_change,
                "class": "outline-none focus:outline-none",
            },
        ],
    ]


def input_checkbox(label, value, on_change):
    return [
        "label.flex.items-center.gap-2",
        [
            "input",
            {
                "type": "checkbox",
                "checked": value,
                "onChange": on_change,
                "class": "h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500",
            },
        ],
        label,
        ["span.font-bold.px-1", value],
    ]


def render_plot(initial_val, initial_sigma):
    SLIDER_STEP = 0.01
    ANIMATION_STEP = 4
    COMPARISON_HEIGHT = 200
    currentKey = key

    def computeState(val, sigma):
        jax_key, adev_key, samples_key = jax.random.split(currentKey, num=3)
        return {
            "JAX_gradients": compute_jax_vals(jax_key, val, sigma),
            "ADEV_gradients": compute_adev_vals(adev_key, val, sigma),
            "samples": make_samples(samples_key, thetas, sigma),
            "val": val,
            "sigma": sigma,
            "frame": 0,
        }

    initialState = Plot.initialState(
        computeState(initial_val, initial_sigma) | {"show_expected_value": True},
        sync={"sigma", "val"},
    )

    def refresh(widget):
        nonlocal currentKey
        currentKey = jax.random.split(currentKey)[0]
        widget.state.update(computeState(widget.state.val, widget.state.sigma))

    onChange = Plot.onChange({
        "val": lambda widget, e: widget.state.update(
            computeState(float(e["value"]), widget.state.sigma)
        ),
        "sigma": lambda widget, e: widget.state.update(
            computeState(widget.state.val, float(e["value"]))
        ),
    })

    samples_plot = make_samples_plot(thetas, js("$state.samples"))

    def plot_tangents(gradients_id):
        tangents_plots = Plot.new(Plot.aspectRatio(0.5))
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

        return Plot.new(
            js("$state.show_expected_value ? %1 : null", expected_value_plot),
            Plot.domain([0, 1], [0, 0.4]),
            tangents_plots,
            Plot.title(f"{gradients_id} Gradient Estimates"),
            Plot.color_map({"JAX Tangent": "orange", "ADEV Tangent": "blue"}),
        )

    comparison_plot = (
        Plot.line(
            js("$state.JAX_gradients.slice(0, $state.frame+1)"),
            x=Plot.index(),
            y="2",
            stroke=Plot.constantly("Gradients from JAX"),
        )
        + Plot.line(
            js("$state.ADEV_gradients.slice(0, $state.frame+1)"),
            x=Plot.index(),
            y="2",
            stroke=Plot.constantly("Gradients from ADEV"),
        )
        + {"x": {"label": "Iteration"}, "y": {"label": "y"}}
        + Plot.domainX([0, EPOCHS])
        + Plot.title("Comparison of computed gradients JAX vs ADEV")
        + Plot.color_legend()
        + {"height": COMPARISON_HEIGHT}
    )

    optimization_plot = Plot.new(
        Plot.line(
            js("$state.JAX_gradients"),
            x=Plot.index(),
            y="1",
            stroke=Plot.constantly("Gradient ascent with JAX"),
            filter=js("(d, i) => i <= $state.frame"),
        )
        + Plot.line(
            js("$state.ADEV_gradients"),
            x=Plot.index(),
            y="1",
            stroke=Plot.constantly("Gradient ascent with ADEV"),
            filter=js("(d, i) => i <= $state.frame"),
        )
        + {
            "x": {"label": "Iteration"},
            "y": {"label": "Expected Value"},
        }
        + Plot.domainX([0, EPOCHS])
        + Plot.title("Maximization of the expected value of a probabilistic function")
        + Plot.color_legend()
        + {"height": COMPARISON_HEIGHT}
    )

    jax_tangents_plot = samples_plot + plot_tangents("JAX")
    adev_tangents_plot = samples_plot + plot_tangents("ADEV")

    frame_slider = Plot.Slider(
        key="frame",
        init=0,
        range=[0, EPOCHS],
        step=ANIMATION_STEP,
        fps=30,
        label="Iteration:",
    )

    controls = Plot.html([
        "div.flex.mb-3.gap-4.bg-gray-200.rounded-md.p-3",
        [
            "div.flex.flex-col.gap-1.w-32",
            input_slider(
                label="Initial Value:",
                value=js("$state.val"),
                min=0,
                max=1,
                step=SLIDER_STEP,
                on_change=js("(e) => $state.val = parseFloat(e.target.value)"),
                default=initial_val,
            ),
            input_slider(
                label="Sigma:",
                value=js("$state.sigma"),
                min=0,
                max=0.2,
                step=0.01,
                on_change=js("(e) => $state.sigma = parseFloat(e.target.value)"),
                default=initial_sigma,
            ),
        ],
        [
            "div.flex.flex-col.gap-2.flex-auto",
            input_checkbox(
                label="Show expected value",
                value=js("$state.show_expected_value"),
                on_change=js("(e) => $state.show_expected_value = e.target.checked"),
            ),
            Plot.katex(r"""
y(\theta) = \mathbb{E}_{x\sim P(\theta)}[x] = \int_{\mathbb{R}}\left[\theta^2\frac{1}{\sqrt{2\pi \sigma^2}}e^{\left(\frac{x}{\sigma}\right)^2} + (1-\theta)\frac{1}{\sqrt{2\pi \sigma^2}}e^{\left(\frac{x-0.5\theta}{\sigma}\right)^2}\right]dx =\frac{\theta-\theta^2}{2}
        """),
            [
                "button.w-32",
                {
                    "onClick": lambda widget, e: refresh(widget),
                    "class": button_classes,
                },
                "Refresh",
            ],
        ],
    ])

    jax_code = """def noisy_jax_model(key, theta, sigma):
    b = jax.random.bernoulli(key, theta)
    return jax.lax.cond(
        b,
        lambda theta: jax.random.normal(key) * sigma * theta,
        lambda theta: jax.random.normal(key) * sigma + theta / 2,
        theta,
    )"""

    adev_code = """@expectation
def flip_approx_loss(theta, sigma):
    b = flip_enum(theta)
    return jax.lax.cond(
        b,
        lambda theta: normal_reparam(0.0, sigma) * theta,
        lambda theta: normal_reparam(theta / 2, sigma),
        theta,
    )"""

    GRID = "div.grid.grid-cols-2.gap-4"
    PRE = "pre.whitespace-pre-wrap.text-2xs.p-3.rounded-md.bg-gray-100.flex-1"

    return (
        initialState
        | onChange
        | controls
        | [
            GRID,
            jax_tangents_plot,
            adev_tangents_plot,
            [PRE, jax_code],
            [PRE, adev_code],
            comparison_plot,
            optimization_plot,
        ]
        | frame_slider
    )


render_plot(0.2, 0.05)


# %% [markdown]
# In the above example, by using `jvp_estimate` we used a forward-mode version of ADEV. GenJAX also supports a reverse-mode version which is also fully compatible with JAX and can be jitted.


# %%
@expectation
def flip_exact_loss(theta):
    b = flip_enum(theta)
    return jax.lax.cond(
        b,
        lambda _: 0.0,
        lambda _: -theta / 2.0,
        theta,
    )


rev_adev_grad = jax.jit(flip_exact_loss.grad_estimate)

arg = 0.2
rev_adev_vals = []
for _ in range(EPOCHS):
    key, subkey = jax.random.split(key)
    (grad_val,) = rev_adev_grad(subkey, (arg,))
    arg = arg - 0.01 * grad_val
    rev_adev_vals.append(expected_val(arg))

(
    Plot.line(
        {"x": list(range(EPOCHS)), "y": rev_adev_vals},
        stroke=Plot.constantly("Reverse mode ADEV"),
    )
    + {"x": {"label": "Iteration"}, "y": {"label": "y"}}
    + Plot.domainX([0, EPOCHS])
    + Plot.title("Maximization of the expected value of a probabilistic function")
    + Plot.color_legend()
)
