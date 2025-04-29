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
# # 3D Function Visualization
#
# This notebook demonstrates a 3D visualization of a quadratic function f(x,θ) : ℝ² → ℝ.

# %%
# Import and constants
import genstudio.plot as Plot
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Define the function and its mathematical representation
math_representation = r"""
\text{Function:} \\
f(x, \theta) = x^2 + \theta^2 - 2x\theta \\[2em]

\text{Integral over } \theta \text{ from } a \text{ to } b: \\
\int_a^b f(x, \theta) d\theta =  x^2(b-a) + \frac{b^3 - a^3}{3} - x(b^2 - a^2)
"""

# JAX implementation
jax_code = """def f(x, theta):
    return x**2 + theta**2 - 2*x*theta


# Approximate integral
g = lambda theta: jnp.mean(jax.vmap(lambda x: f(x, theta))(jnp.linspace(-100, 100, 10000)))
"""

# Constants
GRID_SIZE = 50
X_RANGE = (-3, 3)
THETA_RANGE = (-3, 3)


# %%
# Define the function in JAX
@jax.jit
def f(x, theta):
    return x**2 + theta**2 - 2 * x * theta


# Convert to numpy function for visualization
def f_np(x, theta):
    return np.array(f(jnp.array(x), jnp.array(theta)))


# Generate grid points
x = np.linspace(X_RANGE[0], X_RANGE[1], GRID_SIZE)
theta = np.linspace(THETA_RANGE[0], THETA_RANGE[1], GRID_SIZE)
X, Theta = np.meshgrid(x, theta)

# Compute function values
Z = f_np(X, Theta)

# %%
# Render math and code using Genstudio
Plot.html([
    "div.flex.flex-row.gap-4.p-4",
    [
        "div.flex-1",
        [
            "div",
            ["h3.font-bold", "Mathematical Representation"],
            ["div.p-2.bg-gray-100.rounded", Plot.katex(math_representation)],
        ],
    ],
    [
        "div.flex-1",
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


# %%
# Create the 3D visualization
def plot_3d_function():
    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the surface
    surf = ax.plot_surface(
        X,
        Theta,
        Z,
        cmap=cm.viridis,
        linewidth=0,
        antialiased=True,
        alpha=0.7,  # Make surface slightly transparent
    )

    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Compute and plot the integral along theta
    def integrate_theta(x_val):
        # Simple numerical integration using trapezoidal rule
        theta_points = np.linspace(THETA_RANGE[0], THETA_RANGE[1], 100)
        integrand = f_np(np.ones_like(theta_points) * x_val, theta_points)
        return np.trapz(integrand, theta_points)

    x_integral = np.linspace(X_RANGE[0], X_RANGE[1], 100)
    z_integral = np.array([integrate_theta(x) for x in x_integral])

    # Plot the integral line
    ax.plot(
        x_integral,
        np.ones_like(x_integral) * THETA_RANGE[1],
        z_integral,
        color="blue",
        linewidth=3,
        label="Integral over θ",
    )

    # Set labels
    ax.set_xlabel("x")
    ax.set_ylabel("θ")
    ax.set_zlabel("f(x,θ)")

    # Set title
    ax.set_title("3D Visualization of f(x,θ) = x² + θ² - 2xθ\nwith θ-integral")

    # Add legend
    ax.legend()

    # Adjust view angle
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.show()


# Show the 3D plot
plot_3d_function()
