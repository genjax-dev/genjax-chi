# Copyright 2022 MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import sys
from pathlib import Path
from textwrap import dedent
from typing import Literal, get_args

import nox

try:
    from nox_poetry import Session, session
except ImportError:
    message = f"""\
    Nox failed to import the 'nox-poetry' package.

    Please install it using the following command:

    {sys.executable} -m pip install nox-poetry"""
    raise SystemExit(dedent(message)) from None

package = "genjax"
python_version = "3.11"
nox.needs_version = ">= 2021.6.6"
nox.options.sessions = ("tests", "lint", "build")

JAXSpecifier = Literal["cpu", "cuda12", "tpu"]


def install_package(session, lib: str):
    requirements = session.poetry.export_requirements()
    session.run(
        "poetry",
        "run",
        "pip",
        "install",
        f"--constraint={requirements}",
        lib,
        external=True,
    )


def install_jaxlib(session):
    jax_specifier = None
    if session.posargs and (session.posargs[0] in get_args(JAXSpecifier)):
        jax_specifier = session.posargs[0]
    else:
        jax_specifier = "cpu"

    install_package(session, f"jax[{jax_specifier}]")


@session(python=python_version)
def prepare(session, *with_strs):
    with_pairs = []

    for s in with_strs:
        with_pairs += ["--with", s]

    session.run(
        "poetry",
        "self",
        "add",
        "keyrings.google-artifactregistry-auth",
        external=True,
    )
    session.run_always(
        "poetry", "install", "--with", "dev", *with_pairs, "--all-extras", external=True
    )
    install_jaxlib(session)


@session(python=python_version)
def tests(session):
    prepare(session)
    session.run(
        "poetry",
        "run",
        "pytest",
        "--benchmark-disable",
        "--ignore",
        "scratch",
        "--ignore",
        "notebooks",
        "-n",
        "auto",
        external=True,
    )


@session(python=python_version)
def coverage(session):
    prepare(session)
    session.run(
        "poetry",
        "run",
        "coverage",
        "run",
        "-m",
        "pytest",
        "--benchmark-disable",
        "--ignore",
        "scratch",
        external=True,
    )
    session.run("poetry", "run", "coverage", "json", "--omit", "*/test*", external=True)
    session.run(
        "poetry", "run", "coverage", "report", "--omit", "*/test*", external=True
    )


@session(python=python_version)
def benchmark(session):
    prepare(session)
    session.run(
        "coverage",
        "run",
        "-m",
        "pytest",
        "--benchmark-warmup",
        "on",
        "--ignore",
        "tests",
        "--benchmark-disable-gc",
        "--benchmark-min-rounds",
        "5000",
    )


@session(python=python_version)
def xdoctests(session) -> None:
    """Run examples with xdoctest."""
    prepare(session)
    if session.posargs:
        args = [package, *session.posargs]
    else:
        args = [f"--modname={package}", "--command=all"]
        if "FORCE_COLOR" in os.environ:
            args.append("--colored=1")

    session.install("xdoctest[colors]")
    session.run("python", "-m", "xdoctest", *args)


@session(python=python_version)
def nbmake(session) -> None:
    """Execute Jupyter notebooks as tests"""
    prepare(session)
    session.run(
        "poetry",
        "run",
        "pytest",
        "-n",
        "auto",
        "--nbmake",
        "notebooks/active",
    )


@session(python=python_version)
def safety(session) -> None:
    """Scan dependencies for insecure packages."""
    install_package(session, "safety")
    requirements = session.poetry.export_requirements()
    # Ignore 70612 / CVE-2019-8341, Jinja2 is a safety dep, not ours
    session.run(
        "poetry",
        "run",
        "safety",
        "check",
        "--ignore",
        "70612",
        "--ignore",
        "73456",
        # tornado, dev dependency
        "--ignore",
        "74439",
        # jinja2, dev dependency
        "--ignore",
        "74735",
        "--full-report",
        f"--file={requirements}",
        external=True,
    )


@session(python=python_version)
def lint(session: Session) -> None:
    session.run_always("poetry", "install", "--with", "dev", external=True)
    session.run("ruff", "check", "--fix", ".", external=True)
    session.run("ruff", "format", ".", external=True)


@session(python=python_version)
def build(session):
    prepare(session)
    session.run("poetry", "build")


@session(name="mkdocs", python=python_version)
def mkdocs(session: Session) -> None:
    """Run the mkdocs-only portion of the docs build."""
    prepare(session, "docs")
    build_dir = Path("site")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    session.run("poetry", "run", "mkdocs", "build", "--strict", external=True)


@session(name="docs-build", python=python_version)
def docs_build(session: Session) -> None:
    """Build the documentation."""
    mkdocs(session)
    session.run(
        "poetry", "run", "quarto", "render", "notebooks", "--execute", external=True
    )


@session(name="docs-serve", python=python_version)
def docs_serve(session: Session) -> None:
    """Serve the already-built documentation."""
    session.run(
        "python",
        "-m",
        "http.server",
        "8080",
        "--bind",
        "127.0.0.1",
        "--directory",
        "site",
    )


@session(name="docs-deploy", python=python_version)
def docs_deploy(session: Session) -> None:
    """Deploy the already-built documentation."""
    session.run("poetry", "run", "mkdocs", "gh-deploy", "--force", external=True)


@session(name="docs-build-deploy", python=python_version)
def docs_build_deploy(session: Session) -> None:
    """Build and deploy the documentation site to GH Pages"""
    docs_build(session)
    docs_deploy(session)


@session(name="docs-build-serve", python=python_version)
def docs_build_serve(session: Session) -> None:
    """Build and serve the documentation site."""
    docs_build(session)
    docs_serve(session)


@session(name="notebooks-serve", python=python_version)
def notebooks_serve(session: Session) -> None:
    """Build the documentation."""
    prepare(session)
    session.run("quarto", "preview", "notebooks", external=True)


@session(name="jupyter", python=python_version)
def jupyter(session: Session) -> None:
    """Build the documentation."""
    prepare(session)
    session.run("jupyter-lab", external=True)
