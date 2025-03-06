"""
This module provides visualization tools for clustering results, specifically for the Gen2D model.
It creates an interactive animation showing the evolution of clusters over iterations, including:
- Cluster positions and standard deviations shown as ellipses
- Cluster weights and assignments shown in a sortable table
- Interactive highlighting of clusters on hover
- Background image with sampled pixels
- Frame-by-frame playback controls
"""

import json
from functools import partial

import genstudio.plot as Plot
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy import datasets


def prepare_sampled_pixels(image, pixel_sampling):
    """Sample pixels from the image at regular intervals.

    Returns:
        tuple: (sampled_pixels, sampled_xy, sampled_flattened_indices)
    """
    H, W, _ = image.shape
    sampled_pixels = []
    sampled_flattened_indices = []

    for y in range(0, H, pixel_sampling):
        for x in range(0, W, pixel_sampling):
            sampled_pixels.append([x, y, *image[y, x]])
            sampled_flattened_indices.append(y * W + x)

    sampled_pixels = np.array(sampled_pixels)
    sampled_xy = sampled_pixels[:, 0:2]
    sampled_flattened_indices = np.array(sampled_flattened_indices)

    return sampled_pixels, sampled_xy, sampled_flattened_indices


@jax.jit
def prepare_frame_data_jit(
    weights, xy_means, xy_variances, rgb_means, cluster_assignments
):
    """JIT-compiled version of frame data preparation."""
    # Count all assignments using one-hot encoding and sum
    num_clusters = weights.shape[0]

    # Create one-hot encoding and sum for all points
    one_hot = jax.nn.one_hot(cluster_assignments, num_clusters)
    assignment_counts = jnp.sum(one_hot, axis=0)

    return assignment_counts


def prepare_frame_data(all_posterior_data, sampled_flattened_indices):
    """Prepare data for all frames in the animation."""
    num_frames = len(all_posterior_data["xy_means"])

    # Pre-allocate lists with known sizes
    all_weights_js = [None] * num_frames
    all_means_js = [None] * num_frames
    all_variances_js = [None] * num_frames
    all_colors_js = [None] * num_frames
    all_assignments = [None] * num_frames
    all_total_assignments = [None] * num_frames  # New list for total assignments

    for frame_idx in range(num_frames):
        # Convert frame data to JAX arrays
        weights = jnp.array(all_posterior_data["weights"][frame_idx])
        xy_means = jnp.array(all_posterior_data["xy_means"][frame_idx])
        xy_variances = jnp.array(all_posterior_data["xy_variances"][frame_idx])
        rgb_means = jnp.array(all_posterior_data["rgb_means"][frame_idx])
        cluster_assignments = jnp.array(
            all_posterior_data["cluster_assignments"][frame_idx]
        )

        # Use JIT-compiled function for assignment counting using full assignments
        assignment_counts = prepare_frame_data_jit(
            weights,
            xy_means,
            xy_variances,
            rgb_means,
            cluster_assignments,  # Using full assignments, not sampled
        )

        # Convert back to Python lists
        assignments_array = np.array(assignment_counts)
        all_assignments[frame_idx] = assignments_array.tolist()
        all_total_assignments[frame_idx] = int(
            np.sum(assignments_array)
        )  # Store total assignments
        all_weights_js[frame_idx] = np.array(weights).tolist()
        all_means_js[frame_idx] = np.array(xy_means).tolist()
        all_variances_js[frame_idx] = np.array(xy_variances).tolist()
        all_colors_js[frame_idx] = np.array(rgb_means).tolist()

    return (
        all_weights_js,
        all_means_js,
        all_variances_js,
        all_colors_js,
        all_assignments,
        all_total_assignments,
    )


def create_frame_plot(
    frame_data,
    frame_idx,
    sampled_xy,
    sampled_flattened_indices,
    W,
    H,
    min_weight,
    confidence_factor,
):
    """Create a plot for a specific frame."""
    # Get data for this frame
    xy_means = frame_data["xy_means"][frame_idx]
    xy_variances = frame_data["xy_variances"][frame_idx]
    weights = frame_data["weights"][frame_idx]
    rgb_means = frame_data["rgb_means"][frame_idx]
    frame_assignments = frame_data["cluster_assignments"][frame_idx]

    # Create base plot
    plot = Plot.new(
        Plot.aspectRatio(1),
        Plot.hideAxis(),
        Plot.domain([0, W], [0, H]),
        {"y": {"reverse": True}},
        Plot.title(f"Iteration {frame_idx}/{len(frame_data['xy_means']) - 1}"),
    )

    # Add background image
    plot += Plot.img(
        ["face_temp.png"],
        x=0,
        y=H,
        width=W,
        height=-H,
        src=Plot.identity,
        opacity=0.3,
    )

    # Add points and clusters
    plot = add_sampled_points(
        plot, sampled_xy, frame_assignments, sampled_flattened_indices, rgb_means
    )
    plot = add_cluster_ellipses(
        plot, xy_means, xy_variances, weights, rgb_means, min_weight, confidence_factor
    )

    return plot


@partial(jax.jit, static_argnames=["num_points"])
def compute_ellipse_points(xy_means, xy_variances, confidence_factor, num_points=30):
    """JIT-compiled function to compute ellipse points."""
    theta = jnp.linspace(0, 2 * jnp.pi, num_points)

    var_x = jnp.maximum(xy_variances[0], 1.0)
    var_y = jnp.maximum(xy_variances[1], 1.0)
    x_stddev = jnp.sqrt(var_x) * confidence_factor
    y_stddev = jnp.sqrt(var_y) * confidence_factor

    center_x = xy_means[0]
    center_y = xy_means[1]

    ellipse_x = center_x + x_stddev * jnp.cos(theta)
    ellipse_y = center_y + y_stddev * jnp.sin(theta)

    return ellipse_x, ellipse_y


def add_sampled_points(
    plot, sampled_xy, frame_assignments, sampled_flattened_indices, rgb_means
):
    """Add sampled points to the plot with their assigned cluster colors."""
    sampled_assignments = frame_assignments[sampled_flattened_indices]
    x_values = sampled_xy[:, 0]
    y_values = sampled_xy[:, 1]

    # Create a mask for valid assignments
    valid_mask = sampled_assignments < len(rgb_means)

    # Group points by cluster assignment using numpy operations
    unique_assignments = np.unique(sampled_assignments[valid_mask])

    for assignment in unique_assignments:
        assignment = int(assignment)
        mask = sampled_assignments == assignment

        rgb = rgb_means[assignment].astype(float)
        plot += Plot.dot(
            {"x": x_values[mask].tolist(), "y": y_values[mask].tolist()},
            {
                "r": 2,
                "fill": f"rgb({rgb[0]},{rgb[1]},{rgb[2]})",
                "stroke": "none",
                "fillOpacity": 0.8,
            },
        )

    return plot


def add_cluster_ellipses(
    plot, xy_means, xy_variances, weights, rgb_means, min_weight, confidence_factor
):
    """Add ellipses and centers for each cluster."""
    # Show all clusters in the animation
    valid_indices = range(len(xy_means))

    for i in valid_indices:
        rgb = rgb_means[i].astype(float)
        color = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

        # Use JIT-compiled function to compute ellipse points
        ellipse_x, ellipse_y = compute_ellipse_points(
            xy_means[i], xy_variances[i], confidence_factor, num_points=30
        )

        center_x = float(xy_means[i][0])
        center_y = float(xy_means[i][1])

        plot += Plot.line(
            {"x": np.array(ellipse_x).tolist(), "y": np.array(ellipse_y).tolist()},
            {
                "data-cluster": str(i),
                "stroke": color,
                "strokeWidth": 2,
                "fill": color,
                "fillOpacity": 0.2,
                "class": f"cluster-{i}",
            },
        )

        # Add cluster center
        size = 5 + float(weights[i]) * 50
        plot += Plot.dot(
            {"x": [center_x], "y": [center_y]},
            {
                "data-cluster": str(i),
                "fill": color,
                "r": size,
                "stroke": "black",
                "strokeWidth": 1,
                "symbol": "star",
                "class": f"cluster-{i}",
            },
        )
    return plot


def create_cluster_visualization(
    all_posterior_xy_means,
    all_posterior_xy_variances,
    all_posterior_weights,
    all_posterior_rgb_means,
    all_cluster_assignments,
    image=None,
    num_frames=10,
    pixel_sampling=10,
    confidence_factor=3.0,
    min_weight=0.01,
):
    """Create an interactive visualization of clustering results."""
    # Load default image if none provided
    if image is None:
        image = datasets.face()

    H, W, _ = image.shape
    plt.imsave("face_temp.png", image)

    # Prepare data
    sampled_pixels, sampled_xy, sampled_flattened_indices = prepare_sampled_pixels(
        image, pixel_sampling
    )

    # Organize posterior data - do this once instead of accessing dict repeatedly
    all_posterior_data = {
        "xy_means": all_posterior_xy_means,
        "xy_variances": all_posterior_xy_variances,
        "weights": all_posterior_weights,
        "rgb_means": all_posterior_rgb_means,
        "cluster_assignments": all_cluster_assignments,
    }

    # Calculate frame indices first
    num_iteration = len(all_posterior_xy_means)
    step = max(1, num_iteration // num_frames)
    frame_indices = list(range(0, num_iteration, step))

    # Pre-allocate frames list
    frames = [None] * len(frame_indices)

    # Create frames
    for i, idx in enumerate(frame_indices):
        frames[i] = create_frame_plot(
            all_posterior_data,
            idx,
            sampled_xy,
            sampled_flattened_indices,
            W,
            H,
            min_weight,
            confidence_factor,
        )

    # Prepare JavaScript data once, outside the frame creation
    (
        all_weights_js,
        all_means_js,
        all_variances_js,
        all_colors_js,
        all_assignments,
        all_total_assignments,
    ) = prepare_frame_data(all_posterior_data, sampled_flattened_indices)

    frame_data_js = f"""
    const allWeights = {json.dumps(all_weights_js)};
    const allMeans = {json.dumps(all_means_js)};
    const allVariances = {json.dumps(all_variances_js)};
    const allColors = {json.dumps(all_colors_js)};
    const allAssignments = {json.dumps(all_assignments)};
    const allTotalAssignments = {json.dumps(all_total_assignments)};
    const imageWidth = {W};
    const imageHeight = {H};
    const numFrames = {len(all_posterior_xy_means)};
    const minWeight = {min_weight};
    """

    # Return the complete visualization
    return Plot.html([
        "div",
        {"className": "grid grid-cols-3 gap-4 p-4"},
        [
            "div",
            {"className": "col-span-2"},
            Plot.Frames(frames),
        ],
        [
            "div",
            {"className": "col-span-1"},
            Plot.js(
                """function() {
                """
                + frame_data_js
                + """
                // Get current frame index
                const frame = $state.frame || 0;

                // Get data for current frame
                const weights = allWeights[frame] || [];
                const colors = allColors[frame] || [];
                const assignments = allAssignments[frame] || [];
                const totalAssignments = allTotalAssignments[frame] || 1;  // Use 1 as fallback to avoid division by zero

                // Sort clusters by weight and take top 10
                const topClusters = weights
                    .map((weight, idx) => ({
                        id: idx,
                        weight: Number(weight),
                        color: colors[idx] || [0,0,0],
                        points: Number(assignments[idx] || 0),
                        percentage: ((assignments[idx] || 0) / totalAssignments * 100).toFixed(1)
                    }))
                    .sort((a, b) => b.weight - a.weight)
                    .slice(0, 10);

                return [
                    "div", {},
                    ["h3", {}, `Top ${topClusters.length} Clusters by Weight`],
                    ["div", {"style": {"height": "400px", "overflow": "auto"}},
                        ["table", {"className": "w-full mt-2"},
                            ["thead", ["tr",
                                ["th", {"className": "text-left"}, "Cluster"],
                                ["th", {"className": "text-left"}, "Weight"],
                                ["th", {"className": "text-left"}, "Points (%)"]
                            ]],
                            ["tbody",
                                ...topClusters.map(cluster =>
                                    ["tr", {
                                        "className": "h-8"
                                    },
                                    ["td", {"className": "py-1"},
                                        ["div", {"className": "flex items-center"},
                                            ["div", {
                                                "style": {
                                                    "backgroundColor": `rgb(${cluster.color[0]},${cluster.color[1]},${cluster.color[2]})`,
                                                    "width": "24px",
                                                    "height": "24px",
                                                    "borderRadius": "4px",
                                                    "border": "1px solid rgba(0,0,0,0.2)",
                                                    "display": "inline-block",
                                                    "marginRight": "8px"
                                                }
                                            }],
                                            `Cluster ${cluster.id}`
                                        ]
                                    ],
                                    ["td", {"className": "py-1"}, cluster.weight.toFixed(4)],
                                    ["td", {"className": "py-1"}, `${cluster.points} (${cluster.percentage}%)`]
                                    ]
                                )
                            ]
                        ]
                    ]
                ];
            }()"""
            ),
        ],
    ])
