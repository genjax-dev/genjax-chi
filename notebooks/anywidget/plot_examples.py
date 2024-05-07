# %%
import gen.studio.plot as plot
import numpy as np


# Generate random data from a normal distribution
def normal_100():
    return np.random.normal(loc=0, scale=1, size=1000)


# %% [markdown]
# ### Histogram

plot.rectY(normal_100(), plot.binX({"y": "count"})) + plot.ruleY

# %% [markdown]
# ### Scatter

plot.scatter(normal_100(), normal_100()) + plot.frame

# %% [markdown]
# ### One-dimensional heatmap

(
    plot.rect(normal_100(), plot.binX({"fill": "count"}))
    + plot.color_scheme("YlGnBu")
    + {"height": 75}
)
