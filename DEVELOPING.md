# Developer's Guide

This guide describes how to complete various tasks you'll encounter when working
on the GenJAX codebase.

## Development environment

This project uses:

- [poetry](https://python-poetry.org/) for dependency management
- [nox](https://nox.thea.codes/en/stable/) to automate testing/linting/building.
- [mkdocs](https://www.mkdocs.org/) to generate static documentation.
- [quarto](https://quarto.org/) to render Jupyter notebooks for tutorial notebooks.

### Commit Hooks

We use [pre-commit](https://pre-commit.com/) to manage a series of git
pre-commit hooks for the project; for example, each time you commit code, the
hooks will make sure that your python is formatted properly. If your code isn't,
the hook will format it, so when you try to commit the second time you'll get
past the hook.

All hooks are defined in `.pre-commit-config.yaml`. To install these hooks,
install `pre-commit` if you don't yet have it. I prefer using
[pipx](https://github.com/pipxproject/pipx) so that `pre-commit` stays globally
available.

```bash
pipx install pre-commit
```

Then install the hooks with this command:

```bash
pre-commit install
```

Now they'll run on every commit. If you want to run them manually, run the
following command:

```bash
pre-commit run --all-files
```

### (Option 1): Development environment setup with `poetry`

#### Step 1: Setting up the environment with `poetry`

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
git clone https://github.com/probcomp/genjax
cd genjax
poetry self add "poetry-dynamic-versioning[plugin]"
poetry install
poetry run jupyter-lab
```

You can test your environment with:

```bash
nox -r
```

#### Step 2: Choose a `jaxlib`

GenJAX does not manage the version of `jaxlib` that you use in your execution
environment. The exact version of `jaxlib` can change depending upon the target
deployment hardware (CUDA, CPU, Metal). It is your responsibility to install a
version of `jaxlib` which is compatible with the JAX bounds (`jax = "^0.4.10"`
currently) in GenJAX (as specified in `pyproject.toml`).

[For further information, see this discussion.](https://github.com/google/jax/discussions/16380)

[You can likely install CUDA compatible versions by following environment setup above with a `pip` installation of the CUDA-enabled JAX.](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier)

### (Option 2): Self-managed development environment with `requirements.txt`

#### Using `requirements.txt`

> **This is not the recommended way to develop on `genjax`**, but may be
> required if you want to avoid environment collisions with `genjax` installing
> specific versions of `jax` and `jaxlib`.

`genjax` includes a `requirements.txt` file which is exported from the
`pyproject.toml` dependency requirements -- but with `jax` and `jaxlib` removed.

If you wish to setup a usable environment this way, you must ensure that you
have `jax` and `jaxlib` installed in your environment, then:

```bash
pip install -r requirements.txt
```

This should install a working environment - subject to the conditions that your
version of `jax` and `jaxlib` resolve with the versions of packages in the
`requirements.txt`

### Documentation environment setup

If you want to deploy the documentation and Jupyter notebooks to static HTML,
you'll need [quarto](https://quarto.org/docs/get-started/).

In addition, you'll need `mkdocs`:

```bash
pip install mkdocs
```

GenJAX builds documentation using an insiders-only version of
[mkdocs-material](https://squidfunk.github.io/mkdocs-material/). GenJAX will
attempt to fetch this repository during the documentation build step.

With these dependencies installed (`mkdocs` into your active Python environment)
and on path, you can fully build the documentation:

```bash
nox -r -s docs-build
```

This command will use `mkdocs` to build the static site, and then use `quarto`
to render the notebooks into the static site directory.

Pushing the resulting changes to the `main` branch will trigger a CI job to
deploy to the GitHub Pages branch `gh-pages`, from which the documentation is
hosted.
