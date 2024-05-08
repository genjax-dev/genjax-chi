# %% 
import gen.studio.plot as Plot
import numpy as np

# %% [markdown]
# ## Approach 
#
# - The [pyobsplot](https://github.com/juba/pyobsplot) library creates "stubs" in python which directly mirror the Observable Plot API. An AST-like "spec" is created in python and then interpreted in javascript.
# - The [Observable Plot](https://observablehq.com/plot/) library does not have "chart types" but rather "marks", which are layered to produce a chart. These are nicely composable via `+` in Python.
# - Compared to other common python plotting utilities, this is more functional and often has the effect of condensing and collecting plot-related code. 
#
# ## Instructions 
#
# The starting point for seeing what's possible is the [Observable Plot](https://observablehq.com/plot/what-is-plot) website.
# Plots are composed of **marks**, and you'll want to familiarize yourself with the available marks and how they're created.
#
# In `gen.studio.plot`,
#
#
# Generate random data from a normal distribution
def normal_100():
    return np.random.normal(loc=0, scale=1, size=1000)


# %% [markdown]
# ### Histogram

Plot.rectY(normal_100(), Plot.binX({"y": "count"})) + Plot.ruleY()

# %% [markdown]
# ### Scatter and Line plots
# Unlike other mark types which expect a single values argument, `dot` and `line`
# also accept separate `xs` and `ys` for passing in columnar data (usually the case
# when working with jax.)

Plot.dot(normal_100(), normal_100()) + Plot.frame()

# %% [markdown]
# ### One-dimensional heatmap

(
    Plot.rect(normal_100(), Plot.binX({"fill": "count"}))
    + Plot.color_scheme("YlGnBu")
    + {"height": 75}
)

 # %% [markdown]
 # ### Built-in docs
 # Evaluating a Plot function in a notebook will render its docstring as markdown.
 
 Plot.line
 
 # %% [markdown]
 # ### Plot composition 
 # 
 # Marks and options can be composed by including them as arguments to `Plot.new(...)`,
 # or by adding them to a plot. Adding marks or options does not change the underlying plot,
 # so you can re-use plots in different combinations.
 
gray_box = Plot.line([[0, 0], [0, 2], [2, 2], [2, 0]], fill='#e0e0e0')
gray_box

#%%
framed_titled = gray_box + Plot.frame() + Plot.title("Foo")
framed_titled

#%%

def random_dots(fill):
    return Plot.dot(np.random.uniform(0, 2, 100), np.random.uniform(0, 2, 100), fill=fill)

framed_titled + random_dots('black') + random_dots('magenta')
 
 
 
 