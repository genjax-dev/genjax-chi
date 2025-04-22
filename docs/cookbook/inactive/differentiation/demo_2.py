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
# ### Differentiating probabilistic programs
# %%
# Import and constants

from functools import partial

import genstudio.plot as Plot
import jax
import jax.numpy as jnp
from genstudio.plot import js

from genjax._src.adev.core import Dual, expectation
from genjax._src.adev.primitives import flip_enum, normal_reparam

# Code strings for editable models
jax_code = """def noisy_jax_model(key, theta, sigma):
    b = jax.random.bernoulli(key, theta)
    return jax.lax.cond(
        b,
        lambda theta: jax.random.normal(key) * sigma * theta,
        lambda theta: jax.random.normal(key) * sigma + theta / 3,
        theta,
    )"""

adev_code = """@expectation
def adev_model(theta, sigma):
    b = flip_enum(theta)
    return jax.lax.cond(
        b,
        lambda theta: normal_reparam(0.0, sigma) * theta,
        lambda theta: normal_reparam(theta / 3, sigma),
        theta,
    )"""

key = jax.random.key(314159)
EPOCHS = 400
default_sigma = 0.05
default_initial_val = 0.2
learning_rate = 0.01
default_num_samples_estimate_expected_vals = 10000
default_difference = 0.001
thetas = jnp.arange(0.0, 1.0, 0.0005)


# %%
# Model
def noisy_jax_model(key, theta, sigma):
    b = jax.random.bernoulli(key, theta)
    return jax.lax.cond(
        b,
        lambda theta: jax.random.normal(key) * sigma * theta,
        lambda theta: jax.random.normal(key) * sigma + theta / 3,
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
    gradient = jax.jit(jax.grad(f, argnums=1))(subkey, current_theta, sigma)
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
        new_theta = current_theta + learning_rate * gradient
        return (key, new_theta), (current_theta, expected, gradient)

    _, out = jax.lax.scan(body_fun, (key, initial_theta), None, length=EPOCHS)
    return out


@expectation
def adev_model(theta, sigma):
    b = flip_enum(theta)
    return jax.lax.cond(
        b,
        lambda theta: normal_reparam(0.0, sigma) * theta,
        lambda theta: normal_reparam(theta / 3, sigma),
        theta,
    )


# Pre-compile the gradient functions with initial models
jax_grad_compiled = jax.jit(jax.grad(noisy_jax_model, argnums=1))
adev_jvp_compiled = jax.jit(adev_model.jvp_estimate)


@partial(jax.jit, static_argnames=["g", "f"])
def _compute_adev_step(g, f, key, current_theta, sigma):
    key, subkey = jax.random.split(key)
    gradient = adev_jvp_compiled(
        subkey, (Dual(current_theta, 1.0), Dual(sigma, 0.0))
    ).tangent
    key, subkey = jax.random.split(key)
    expected = expected_val(f, subkey, current_theta, sigma)
    return key, current_theta, expected, gradient


@partial(jax.jit, static_argnames=["g", "f"])
def compute_adev_vals(g, f, key, initial_theta, sigma):
    def body_fun(carry, _):
        key, current_theta = carry
        key, current_theta, expected, gradient = _compute_adev_step(
            g, f, key, current_theta, sigma
        )
        new_theta = current_theta + learning_rate * gradient
        return (key, new_theta), (current_theta, expected, gradient)

    _, out = jax.lax.scan(body_fun, (key, initial_theta), None, length=EPOCHS)
    return out


# %%

### ADEV VIZ
plot_options = Plot.new(
    Plot.color_legend(),
    {"x": {"label": "θ"}, "y": {"label": "y"}},
    Plot.aspect_ratio(1),
    Plot.grid(),
)

samples_color_map = Plot.color_map({"Samples": "rgba(0, 128, 128, 0.5)"})


def make_samples_plot(thetas, samples):
    return (
        Plot.dot({"x": thetas, "y": samples}, fill=Plot.constantly("Samples"), r=2)
        + samples_color_map
        + plot_options
        + Plot.clip()
    )


thetas_sparse = jnp.linspace(0.0, 1.0, 20)
key, samples_key = jax.random.split(key)
exact_vals = jax.vmap(
    lambda theta: expected_val(noisy_jax_model, key, theta, default_sigma)
)(thetas_sparse)

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
            "JAX_gradients": compute_jax_vals(noisy_jax_model, jax_key, val, sigma),
            "ADEV_gradients": compute_adev_vals(
                adev_model, noisy_jax_model, adev_key, val, sigma
            ),
            "samples": make_samples(noisy_jax_model, samples_key, thetas, sigma),
            "val": val,
            "sigma": sigma,
            "frame": 0,
            "source": jax_code,
            "adev_source": adev_code,
            "toEval": "",
        }

    initialState = Plot.initialState({
        "samples": make_samples(noisy_jax_model, samples_key, thetas, default_sigma),
        "thetas": thetas,
        "toEval": "",
        "source": jax_code,
        "adev_source": adev_code,
        "JAX_gradients": compute_jax_vals(
            noisy_jax_model, key, default_initial_val, default_sigma
        ),
        "ADEV_gradients": compute_adev_vals(
            adev_model, noisy_jax_model, key, default_initial_val, default_sigma
        ),
        "val": default_initial_val,
        "sigma": default_sigma,
        "frame": 0,
    })

    def refresh(widget):
        nonlocal currentKey
        currentKey = jax.random.split(currentKey)[0]
        widget.state.update(computeState(widget.state.val, widget.state.sigma))

    def evaluate(widget, _e):
        # Update random key and evaluate new code from text editor
        global key, noisy_jax_model, adev_model, jax_grad_compiled, adev_jvp_compiled
        source = f"global noisy_jax_model, adev_model\n{widget.state.toEval}"
        exec(source)
        # Recompile gradient functions with new models
        jax_grad_compiled = jax.jit(jax.grad(noisy_jax_model, argnums=1))
        adev_jvp_compiled = jax.jit(adev_model.jvp_estimate)
        # Recompute values and reset frame to 0
        jax_key, adev_key, samples_key = jax.random.split(key, num=3)
        new_state = {
            "JAX_gradients": compute_jax_vals(
                noisy_jax_model, jax_key, widget.state.val, widget.state.sigma
            ),
            "ADEV_gradients": compute_adev_vals(
                adev_model,
                noisy_jax_model,
                adev_key,
                widget.state.val,
                widget.state.sigma,
            ),
            "samples": make_samples(
                noisy_jax_model, samples_key, thetas, widget.state.sigma
            ),
            "frame": 0,  # Reset frame to 0 when code changes
        }
        widget.state.update(new_state)

    onChange = Plot.onChange({
        "val": lambda widget, e: widget.state.update(
            computeState(float(e["value"]), widget.state.sigma)
        ),
        "sigma": lambda widget, e: widget.state.update(
            computeState(widget.state.val, float(e["value"]))
        ),
        "toEval": lambda widget, e: evaluate(widget, e.value),
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
            Plot.katex(r"""
y(\theta) = \mathbb{E}_{x\sim P(\theta)}[x] = \int_{\mathbb{R}}\left[\theta^2\frac{1}{\sqrt{2\pi \sigma^2}}e^{\left(\frac{x}{\sigma}\right)^2} + (1-\theta)\frac{1}{\sqrt{2\pi \sigma^2}}e^{\left(\frac{x-\frac{1}{3}\theta}{\sigma}\right)^2}\right]dx =\frac{\theta-\theta^2}{3}
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

    # Add code editors
    code_editors = Plot.html([
        "div.flex.flex-row.gap-4",
        [
            "div.flex.flex-col.gap-2.flex-1",
            ["h3.font-bold", "JAX Model"],
            [
                "form.!flex.flex-col.gap-3.h-full",
                {
                    "onSubmit": js(
                        "e => { e.preventDefault(); $state.toEval = $state.source}"
                    )
                },
                [
                    "textarea.whitespace-pre-wrap.text-[13px].lh-normal.p-3.rounded-md.bg-gray-100.flex-1.h-[300px].font-mono",
                    {
                        "rows": js("$state.source.split('\\n').length+1"),
                        "onChange": js("(e) => $state.source = e.target.value"),
                        "value": js("$state.source"),
                        "onKeyDown": js(
                            "(e) => { if (e.ctrlKey && e.key === 'Enter') { e.stopPropagation(); $state.toEval = $state.source } }"
                        ),
                    },
                ],
                [
                    "div.flex.items-stretch.mt-auto",
                    [
                        "button.flex-auto.!bg-blue-500.!hover:bg-blue-600.text-white.text-center.px-4.py-2.rounded-md.cursor-pointer",
                        {"type": "submit"},
                        "Evaluate and Plot",
                    ],
                    [
                        "div.flex.items-center.p-2",
                        {
                            "onClick": lambda widget, _: widget.state.update({
                                "source": jax_code
                            })
                        },
                        "Reset Source",
                    ],
                ],
            ],
        ],
        [
            "div.flex.flex-col.gap-2.flex-1",
            ["h3.font-bold", "ADEV Model"],
            [
                "form.!flex.flex-col.gap-3.h-full",
                {
                    "onSubmit": js(
                        "e => { e.preventDefault(); $state.toEval = $state.adev_source}"
                    )
                },
                [
                    "textarea.whitespace-pre-wrap.text-[13px].lh-normal.p-3.rounded-md.bg-gray-100.flex-1.h-[300px].font-mono",
                    {
                        "rows": js("$state.adev_source.split('\\n').length+1"),
                        "onChange": js("(e) => $state.adev_source = e.target.value"),
                        "value": js("$state.adev_source"),
                        "onKeyDown": js(
                            "(e) => { if (e.ctrlKey && e.key === 'Enter') { e.stopPropagation(); $state.toEval = $state.adev_source } }"
                        ),
                    },
                ],
                [
                    "div.flex.items-stretch.mt-auto",
                    [
                        "button.flex-auto.!bg-blue-500.!hover:bg-blue-600.text-white.text-center.px-4.py-2.rounded-md.cursor-pointer",
                        {"type": "submit"},
                        "Evaluate and Plot",
                    ],
                    [
                        "div.flex.items-center.p-2",
                        {
                            "onClick": lambda widget, _: widget.state.update({
                                "adev_source": adev_code
                            })
                        },
                        "Reset Source",
                    ],
                ],
            ],
        ],
    ])

    GRID = "div.grid.grid-cols-2.gap-4"
    # PRE = "pre.whitespace-pre-wrap.text-2xs.p-3.rounded-md.bg-gray-100.flex-1"

    return (
        initialState
        | onChange
        | controls
        | code_editors
        | [
            GRID,
            jax_tangents_plot,
            adev_tangents_plot,
            comparison_plot,
            optimization_plot,
        ]
        | frame_slider
    )


render_plot(default_initial_val, default_sigma)
# %%
