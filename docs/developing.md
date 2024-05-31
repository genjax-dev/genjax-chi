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
version of `jaxlib` which is compatible with the JAX bounds (`jax = "^0.4.28"`
currently) in GenJAX (as specified in `pyproject.toml`).

[For further information, see this discussion.](https://github.com/google/jax/discussions/16380)

[You can likely install CUDA compatible versions by following environment setup above with a `pip` installation of the CUDA-enabled JAX.](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier)

When running any of the `nox` commands, append `-- <jax_specifier>` to install
the proper `jaxlib` into the session. For example,

```sh
nox -s tests -- cpu
```

will run the tests with the CPU `jaxlib` installed, while

```sh
nox -s tests -- cuda12
```

will install the CUDA bindings. By default, the CPU bindings will be installed.

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
you'll need to install [quarto](https://quarto.org/docs/get-started/) on your
machine.

GenJAX builds documentation using an insiders-only version of
[mkdocs-material](https://squidfunk.github.io/mkdocs-material/). GenJAX will
attempt to fetch this repository during the documentation build step.

Run the following command to fully build the documentation:

```bash
nox -r -s docs-build
```

This command will use `mkdocs` to build the static site, and then use `quarto`
to render the notebooks into the static site directory.

To view the generated site, run:

```bash
nox -r -s docs-serve
```

or to run both commands in sequence:

```bash
nox -r -s docs-build-serve
```

## Granting someone access to GenJAX's Artifact Registry

- Visit the [artifact registry
  page](https://console.cloud.google.com/artifacts?hl=en&project=probcomp-caliban)
  and click the checkbox next to `probcomp` to bring up the "Permissions" tab on
  the right side of the page
- Click "Add Principal"
- Under "New principals", type in the email addresses (associated with a Google
  account) of the people you want to grant access
- Under "Role", fill in "Artifact Registry Reader"
- Click "Save"

These users should now be able to follow the README.md instructions to install
GenJAX.

## Releasing GenJAX

Published GenJAX artifacts live [on PyPI](https://pypi.org/project/genjax/) and
are published automatically by GitHub with each new
[release](https://github.com/probcomp/genjax/releases).

### Release checklist

Before cutting a new release:

- Update README.md to reference new GenJAX versions
- Make sure that the referenced `jax` and `jaxlib` versions match the version
  declared in `pyproject.toml`

### Releasing via GitHub

- Visit https://github.com/probcomp/genjax/releases/new to create a new release.
- From the "Choose a tag" dropdown, type the new version (using the format
  `v<MAJOR>.<MINOR>.<INCREMENTAL>`, like `v0.1.0`) and select "Create new tag
  on publish"
- Fill out an appropriate title, and add release notes generated by looking at
  PRs merged since the last release
- Click "Publish Release"

This will build and publish the new version to Artifact Registry.

### Manually publishing to Google Artifact Registry

- Ask @sritchie to add you to the `probcomp-caliban` project on Google Cloud.
- [Install the Google Cloud command line
  tools](https://cloud.google.com/sdk/docs/install).
- Follow the instructions on the [installation
  page](https://cloud.google.com/sdk/docs/install)
- run `gcloud init` as described [in this
  guide](https://cloud.google.com/sdk/docs/initializing) and configure the tool
  with the ID of your new Cloud project.
- Make sure you've added the dynamic versioning plugin, then configure poetry to
  deploy to the `probcomp-caliban` artifact registry:

```shell
poetry self add poetry-dynamic-versioning[plugin]
poetry self add keyrings.google-artifactregistry-auth
poetry config repositories.gcp https://us-west1-python.pkg.dev/probcomp-caliban/probcomp/
```

- create a new version tag on the `main` branch of the form
  `v<MAJOR>.<MINOR>.<INCREMENTAL>`, like `v0.1.0`, and push the tag to the
  remote repository:

```sh
git tag v0.1.0
git push --tags
```

- use Poetry to build and publish the artifact to Artifact Registry:

```sh
poetry publish --build --repository gcp
```


### Manually publishing to PyPI

> NOTE: please only do this once we've made the repository public and released a
> version to PyPI.

To publish a version manually, you'll need to be added to the GenJAX Maintainers
list on PyPI, or ask a [current maintainer from the project
page]((https://pypi.org/project/genjax/)) for help. Once that's settled:

- generate an API token on your [pypi account
  page](https://pypi.org/manage/account/token/), scoped to all projects or
  scoped specifically to genjax
- copy the token and install it on your machine by running the following
  command:

```sh
poetry config pypi-token.pypi <api-token>
```

- create a new version tag on the `main` branch of the form
  `v<MAJOR>.<MINOR>.<INCREMENTAL>`, like `v0.1.0`, and push the tag to the
  remote repository:

```sh
git tag v0.1.0
git push --tags
```

- use Poetry to build and publish the artifact to pypi:

```sh
poetry publish --build
```
