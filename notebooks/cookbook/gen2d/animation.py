def create_cluster_visualization(
    all_posterior_xy_means,
    all_posterior_xy_variances,
    all_posterior_weights,
    all_posterior_rgb_means,
    image=None,
    num_frames=10,
    pixel_sampling=10,
    confidence_factor=3.0,
    min_weight=0.01,
):
    """
    Create an interactive visualization of clustering results.

    Parameters:
    -----------
    all_posterior_xy_means : list of arrays
        XY mean positions for each cluster in each iteration
    all_posterior_xy_variances : list of arrays
        XY variances for each cluster in each iteration
    all_posterior_weights : list of arrays
        Weights for each cluster in each iteration
    all_posterior_rgb_means : list of arrays
        RGB mean colors for each cluster in each iteration
    image : ndarray, optional
        Image to visualize (defaults to scipy.datasets.face())
    num_frames : int, optional
        Number of frames to show in the animation
    pixel_sampling : int, optional
        Sample every Nth pixel in both directions
    confidence_factor : float, optional
        Scale factor for ellipse size (larger = bigger ellipses)
    min_weight : float, optional
        Minimum weight for a cluster to be displayed

    Returns:
    --------
    Plot object
        The interactive visualization
    """
    import json
    import math

    import genstudio.plot as Plot
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import datasets
    from scipy.spatial.distance import cdist

    # Load the default image if none provided
    if image is None:
        image = datasets.face()

    H, W, _ = image.shape

    # Save the image to a file
    plt.imsave("face_temp.png", image)

    # Sample pixels
    sampled_pixels = []
    for y in range(0, H, pixel_sampling):
        for x in range(0, W, pixel_sampling):
            sampled_pixels.append([x, y, *image[y, x]])

    sampled_pixels = np.array(sampled_pixels)
    sampled_xy = sampled_pixels[:, 0:2]
    total_points = len(sampled_xy)

    # Calculate assignments for each frame
    all_assignments = []
    for frame_idx in range(len(all_posterior_xy_means)):
        # Get data for this frame
        xy_means = all_posterior_xy_means[frame_idx]
        weights = all_posterior_weights[frame_idx]

        # Calculate distance from each pixel to each cluster center
        distances = cdist(sampled_xy, xy_means)

        # Weight by cluster weights
        weighted_distances = distances - np.log(weights + 1e-10)[:, np.newaxis].T

        # Assign each pixel to closest weighted cluster
        assignments = np.argmin(weighted_distances, axis=1)

        # Count assignments per cluster
        unique, counts = np.unique(assignments, return_counts=True)
        assignment_counts = np.zeros(len(weights))
        assignment_counts[unique] = counts

        all_assignments.append(assignment_counts.tolist())

    # Prepare data for JavaScript
    all_weights_js = []
    all_means_js = []
    all_variances_js = []
    all_colors_js = []

    # Prepare data for all frames
    for frame_idx in range(len(all_posterior_xy_means)):
        # Get data for this frame
        weights = all_posterior_weights[frame_idx].tolist()
        xy_means = all_posterior_xy_means[frame_idx].tolist()
        xy_variances = all_posterior_xy_variances[frame_idx].tolist()
        rgb_means = all_posterior_rgb_means[frame_idx].tolist()

        all_weights_js.append(weights)
        all_means_js.append(xy_means)
        all_variances_js.append(xy_variances)
        all_colors_js.append(rgb_means)

    # Convert data to JSON for JavaScript
    frame_data_js = f"""
    const allWeights = {json.dumps(all_weights_js)};
    const allMeans = {json.dumps(all_means_js)};
    const allVariances = {json.dumps(all_variances_js)};
    const allColors = {json.dumps(all_colors_js)};
    const allAssignments = {json.dumps(all_assignments)};
    const imageWidth = {W};
    const imageHeight = {H};
    const numFrames = {len(all_posterior_xy_means)};
    const totalPoints = {total_points};
    const minWeight = {min_weight};
    """

    # Function to create a plot for a specific frame
    def create_frame_plot(frame_idx):
        # Get data for this frame
        xy_means = all_posterior_xy_means[frame_idx]
        xy_variances = all_posterior_xy_variances[frame_idx]
        weights = all_posterior_weights[frame_idx]
        rgb_means = all_posterior_rgb_means[frame_idx]

        # Start with a base plot
        plot = Plot.new(
            Plot.aspectRatio(1),
            Plot.hideAxis(),
            Plot.domain([0, W], [0, H]),
            {"y": {"reverse": True}},
            Plot.title(f"Iteration {frame_idx}/{len(all_posterior_xy_means) - 1}"),
        )

        # Add the background image
        plot += Plot.img(
            ["face_temp.png"],
            x=0,
            y=H,
            width=W,
            height=-H,
            src=Plot.identity,
            opacity=0.5,
        )

        # Add sampled pixels as dots
        x_values = sampled_xy[:, 0].tolist()
        y_values = sampled_xy[:, 1].tolist()
        colors = [f"rgb({r},{g},{b})" for r, g, b in sampled_pixels[:, 2:5]]

        # Add sampled pixels as dots
        plot += Plot.dot(
            {"x": x_values, "y": y_values}, {"r": 2, "fill": colors, "stroke": "none"}
        )

        # Add standard deviation ellipses and cluster centers
        for i in range(len(xy_means)):
            # Skip clusters with very small weights
            if weights[i] < min_weight:
                continue

            # Get RGB color
            rgb = rgb_means[i].astype(float)
            color = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

            # Extract and verify variance values
            var_x = float(xy_variances[i][0])
            var_y = float(xy_variances[i][1])

            # Ensure variances are positive
            var_x = max(var_x, 1.0)
            var_y = max(var_y, 1.0)

            # Calculate standard deviations
            x_stddev = math.sqrt(var_x)
            y_stddev = math.sqrt(var_y)

            # Scale to make them more visible
            x_stddev *= confidence_factor
            y_stddev *= confidence_factor

            # Create points for an ellipse
            theta = np.linspace(0, 2 * np.pi, 30)
            ellipse_x = xy_means[i][0] + x_stddev * np.cos(theta)
            ellipse_y = xy_means[i][1] + y_stddev * np.sin(theta)

            # Add data-cluster attribute for highlighting
            ellipse_attrs = {
                "data-cluster": str(i),
                "stroke": color,
                "strokeWidth": 2,
                "fill": color,
                "fillOpacity": 0.2,
            }

            # Add ellipse to represent standard deviation
            plot += Plot.line(
                {"x": ellipse_x.tolist(), "y": ellipse_y.tolist()}, ellipse_attrs
            )

            # Add the cluster center as a star
            size = 5 + weights[i] * 50
            dot_attrs = {
                "data-cluster": str(i),
                "fill": color,
                "r": size,
                "stroke": "black",
                "strokeWidth": 1,
                "symbol": "star",
            }

            plot += Plot.dot(
                {"x": [float(xy_means[i][0])], "y": [float(xy_means[i][1])]}, dot_attrs
            )

        return plot

    # Create a list of frames to visualize
    num_iteration = len(all_posterior_xy_means)
    step = max(1, num_iteration // num_frames)
    frame_indices = range(0, num_iteration, step)

    # Create all the frames
    frames = [create_frame_plot(idx) for idx in frame_indices]

    # Return the animation with legend and interactive controls
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
                const means = allMeans[frame] || [];
                const colors = allColors[frame] || [];
                const assignments = allAssignments[frame] || [];

                // Sort clusters by weight, filter by minimum weight
                const topClusters = weights
                    .map((weight, idx) => ({
                        id: idx,
                        weight: weight,
                        color: colors[idx] || [0,0,0],
                        points: assignments[idx] || 0,
                        percentage: ((assignments[idx] || 0) / totalPoints * 100).toFixed(1)
                    }))
                    .filter(c => c.weight >= minWeight)
                    .sort((a, b) => b.weight - a.weight)
                    .slice(0, 10);

                // Create placeholder rows for consistent height
                const placeholders = Array(Math.max(0, 10 - topClusters.length))
                    .fill(0)
                    .map(() => ["tr", {"className": "h-8"}, ["td", {"colSpan": 3}, ""]]);

                // Function to highlight/unhighlight clusters
                if (!$state.highlightCluster) {
                    $state.highlightCluster = function(id) {
                        // Unhighlight all clusters
                        document.querySelectorAll('[data-cluster]').forEach(el => {
                            el.style.filter = 'opacity(0.4)';
                        });

                        // Highlight the selected cluster
                        if (id !== null) {
                            document.querySelectorAll(`[data-cluster="${id}"]`).forEach(el => {
                                el.style.filter = 'opacity(1) drop-shadow(0 0 5px white)';
                            });
                        } else {
                            // Reset all if nothing selected
                            document.querySelectorAll('[data-cluster]').forEach(el => {
                                el.style.filter = '';
                            });
                        }
                    };
                }

                return [
                    "div", {},
                    ["h3", {}, `Top Clusters by Weight`],
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
                                        "className": "h-8",
                                        "style": {
                                            "cursor": "pointer",
                                            "backgroundColor": $state.hoveredCluster === cluster.id ? "#f0f0f0" : "transparent"
                                        },
                                        "onMouseEnter": () => {
                                            $state.hoveredCluster = cluster.id;
                                            $state.highlightCluster(cluster.id);
                                        },
                                        "onMouseLeave": () => {
                                            $state.hoveredCluster = null;
                                            $state.highlightCluster(null);
                                        }
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
                                ),
                                ...placeholders
                            ]
                        ]
                    ]
                ];
            }()"""
            ),
        ],
    ])
