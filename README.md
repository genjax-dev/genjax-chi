<br>
<p align="center">
<img width="400px" src="./docs/assets/img/logo.png"/>
</p>
<br>

<div align="center">
<b><i>Probabilistic programming with Gen, built on top of JAX.</i></b>
</div>
<br>

## 🔎 Welcome to the GenJAX Bug Hunt!

We invite you to take a look at a simple inference task accomplished with GenJAX,
in the form of a [Jupyter notebook](https://probcomp.github.io/genjax/notebooks/concepts/introduction/intro_interpreted.html).

Then follow these instructions to prepare a development environment capable of
running the unit tests.

## Development environment

[First, install `poetry` to your system.](https://python-poetry.org/docs/#installing-with-the-official-installer)

Assuming you have `poetry`, here's a simple script to setup a compatible
development environment - if you can run this script, you have a working
development environment which can be used to execute tests, build and serve the
documentation, etc.

```bash
conda create --name genjax-py311 python=3.11 --channel=conda-forge
conda activate genjax-py311
pip install nox
pip install nox-poetry
git clone https://github.com/probcomp/genjax.bughunt
cd genjax.bughunt
poetry self add "poetry-dynamic-versioning[plugin]"
poetry install
```

You can test your environment with:

```bash
poetry run pytest tests/misc/test_bug_hunt.py
```

That test file contains one test that succeeds and another that fails.
The task is to understand why and what might be done about it.
Your host(s) will work with you in a pair programming environment.
