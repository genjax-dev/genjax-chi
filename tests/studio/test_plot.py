import gen.studio.plot as Plot
import pyobsplot
from gen.studio.plot import MarkDefault, Plot, PlotSpec


def test_plotspec_init():
    ps = PlotSpec()
    assert ps.opts == {"marks": []}

    ps = PlotSpec(marks=[Plot.dot()])
    assert len(ps.opts["marks"]) == 1
    assert "pyobsplot-type" in ps.opts["marks"][0]

    ps = PlotSpec(width=100)
    assert ps.opts == {"marks": [], "width": 100}


def test_plotspec_add():
    ps1 = PlotSpec(marks=[Plot.dot()], width=100)
    ps2 = PlotSpec(marks=[Plot.line()], height=200)

    ps3 = ps1 + ps2
    assert len(ps3.opts["marks"]) == 2
    assert ps3.opts["width"] == 100
    assert ps3.opts["height"] == 200

    ps4 = ps1 + [Plot.text()]
    assert len(ps4.opts["marks"]) == 2

    ps5 = ps1 + {"color": "red"}
    assert ps5.opts["color"] == "red"

    try:
        ps1 + "invalid"
        assert False, "Expected TypeError"
    except TypeError:
        pass


def test_plotspec_plot():
    ps = PlotSpec(marks=[Plot.dot()], width=100)
    assert ps.opts["width"] == 100
    plot = ps.plot()
    assert isinstance(plot, pyobsplot.widget.ObsplotWidget)

    # Check plot is cached
    plot2 = ps.plot()
    assert plot is plot2


def test_mark_default():
    md = MarkDefault("frame", {"stroke": "red"})
    assert len(md.opts["marks"]) == 1
    assert md.opts["marks"][0]["args"][0]["stroke"] == "red"

    md2 = md(stroke="blue")
    assert md2.opts["marks"][0]["args"][0]["stroke"] == "blue"


def test_sugar():
    ps = PlotSpec() + Plot.grid_x
    assert ps.opts["x"]["grid"] == True

    ps = PlotSpec() + Plot.grid_y
    assert ps.opts["y"]["grid"] == True

    ps = PlotSpec() + Plot.grid
    assert ps.opts["grid"] == True

    ps = PlotSpec() + Plot.color_legend
    assert ps.opts["color"]["legend"] == True

def run_tests():
    test_plotspec_init()
    test_plotspec_add()
    test_plotspec_plot()
    test_mark_default()
    test_sugar()
    print("All tests passed!")


run_tests()
