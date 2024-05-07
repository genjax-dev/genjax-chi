# %%

import gen.studio.plot as Plot
import importlib 
importlib.reload(Plot)
import pyobsplot
from gen.studio.plot import MarkDefault, PlotSpec

#%%

xs = [1, 2, 3, 4, 5]
ys = [2, 3, 2, 1, 8]

def test_plotspec_init():
    ps = Plot.new()
    assert ps.spec == {"marks": []}

    ps = Plot.dot(xs, ys)
    assert len(ps.spec["marks"]) == 1
    assert "pyobsplot-type" in ps.spec["marks"][0]

    ps = Plot.new(width=100)
    assert ps.spec == {"marks": [], "width": 100}


def test_plotspec_add():
    ps1 = Plot.new(Plot.dot(xs, ys), width=100)
    ps2 = Plot.new(Plot.line(xs, ys), height=200)

    ps3 = ps1 + ps2
    assert len(ps3.spec["marks"]) == 2
    assert ps3.spec["width"] == 100
    assert ps3.spec["height"] == 200

    ps4 = ps1 + Plot.text("foo")
    assert len(ps4.spec["marks"]) == 2

    ps5 = ps1 + {"color": "red"}
    assert ps5.spec["color"] == "red"

    try:
        ps1 + "invalid"
        assert False, "Expected TypeError"
    except TypeError:
        pass


def test_plotspec_plot():
    ps = Plot.new(Plot.dot(xs, ys), width=100)
    assert ps.spec["width"] == 100
    plot = ps.plot()
    assert isinstance(plot, pyobsplot.widget.ObsplotWidget)

    # Check plot is cached
    plot2 = ps.plot()
    assert plot is plot2


def test_mark_default():
    md = MarkDefault("frame", {"stroke": "red"})
    assert len(md.spec["marks"]) == 1
    assert md.spec["marks"][0]["args"][0]["stroke"] == "red"

    md2 = md(stroke="blue")
    assert md2.spec["marks"][0]["args"][0]["stroke"] == "blue"


def test_sugar():
    ps = Plot.new() + Plot.grid_x
    assert ps.spec["x"]["grid"] == True

    ps = Plot.new() + Plot.grid
    assert ps.spec["grid"] == True

    ps = Plot.new() + Plot.color_legend
    assert ps.spec["color"]["legend"] == True

    ps = Plot.new() + Plot.clip
    assert ps.spec["clip"] == True

    ps = Plot.new() + Plot.aspect_ratio(0.5)
    assert ps.spec["aspectRatio"] == 0.5

    ps = Plot.new() + Plot.color_scheme("blues")
    assert ps.spec["color"]["scheme"] == "blues"

    ps = Plot.new() + Plot.domainX([0, 10])
    assert ps.spec["x"]["domain"] == [0, 10]

    ps = Plot.new() + Plot.domainY([0, 20])
    assert ps.spec["y"]["domain"] == [0, 20]

    ps = Plot.new() + Plot.domain([0, 10], [0, 20])
    assert ps.spec["x"]["domain"] == [0, 10]
    assert ps.spec["y"]["domain"] == [0, 20]

    ps = Plot.new() + Plot.margin(10)
    assert ps.spec["margin"] == 10

    ps = Plot.new() + Plot.margin(10, 20)
    assert ps.spec["marginTop"] == 10
    assert ps.spec["marginBottom"] == 10
    assert ps.spec["marginLeft"] == 20
    assert ps.spec["marginRight"] == 20

    ps = Plot.new() + Plot.margin(10, 20, 30)  
    assert ps.spec["marginTop"] == 10
    assert ps.spec["marginLeft"] == 20
    assert ps.spec["marginRight"] == 20
    assert ps.spec["marginBottom"] == 30

    ps = Plot.new() + Plot.margin(10, 20, 30, 40)
    assert ps.spec["marginTop"] == 10
    assert ps.spec["marginRight"] == 20
    assert ps.spec["marginBottom"] == 30
    assert ps.spec["marginLeft"] == 40

def test_plot_new():
    ps = Plot.new(Plot.dot(xs, ys))
    assert isinstance(ps, PlotSpec)
    assert len(ps.spec["marks"]) == 1
    assert ps.spec["marks"][0]["method"] == "dot"

def test_plotspec_reset():
    ps = Plot.new(Plot.dot(xs, ys), width=100)
    assert ps.spec["width"] == 100
    assert len(ps.spec["marks"]) == 1
    
    ps.reset(marks=[Plot.rectY(xs)], height=200)
    assert ps.spec.get("width", None) == None  # width removed
    assert ps.spec["height"] == 200
    assert len(ps.spec["marks"]) == 1
    assert ps.spec["marks"][0]["method"] == "rectY"

def test_plotspec_update():
    ps = Plot.new(Plot.dot(xs, ys), width=100)
    assert ps.spec["width"] == 100
    assert len(ps.spec["marks"]) == 1
    
    ps.update(Plot.rectY(xs), height=200)
    assert ps.spec["width"] == 100
    assert ps.spec["height"] == 200
    assert len(ps.spec["marks"]) == 2
    assert ps.spec["marks"][0]["method"] == "dot"
    assert ps.spec["marks"][1]["method"] == "rectY"

def run_tests():
    test_plotspec_init()
    test_plotspec_add()
    test_plotspec_plot()
    test_mark_default()
    test_sugar()
    test_plot_new()
    test_plotspec_reset()
    test_plotspec_update()
    print("All tests passed!")


run_tests()

#%%
