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

import nox


try:
    from nox_poetry import Session
    from nox_poetry import session
except ImportError:
    message = f"""\
    Nox failed to import the 'nox-poetry' package.

    Please install it using the following command:

    {sys.executable} -m pip install nox-poetry"""
    raise SystemExit(dedent(message)) from None

package = "genjax"
python_version = "3.11"
nox.needs_version = ">= 2021.6.6"
nox.options.sessions = (
    "tests",
    "xdoctests",
    "lint",
    "build",
)


@session(python=python_version)
def tests(session):
    session.install("poetry")
    session.run("poetry", "install")
    session.run(
        "poetry",
        "run",
        "coverage",
        "run",
        "-m",
        "pytest",
        "--benchmark-disable",
        "--ignore",
        "experiments",
        "--ignore",
        "benchmarks",
    )
    session.run("poetry", "run", "coverage", "json")
    session.run("poetry", "run", "coverage", "report")


@session(python=python_version)
def benchmark(session):
    session.install("poetry")
    session.run("poetry", "install")
    session.run(
        "coverage",
        "run",
        "-m",
        "pytest",
        "--benchmark-warmup",
        "on",
        "--ignore",
        "experiments",
        "--ignore",
        "tests",
        "--benchmark-disable-gc",
        "--benchmark-min-rounds",
        "5000",
    )
    session.run("coverage", "json")
    session.run("coverage", "report")


@session(python=python_version)
def xdoctests(session) -> None:
    """Run examples with xdoctest."""
    if session.posargs:
        args = [package, *session.posargs]
    else:
        args = [f"--modname={package}", "--command=all"]
        if "FORCE_COLOR" in os.environ:
            args.append("--colored=1")

    session.install(".")
    session.install("xdoctest[colors]")
    session.run("python", "-m", "xdoctest", *args)


@session(python=python_version)
def safety(session) -> None:
    """Scan dependencies for insecure packages."""
    requirements = session.poetry.export_requirements()
    session.install("safety")
    session.run("safety", "check", "--full-report", f"--file={requirements}")


@session(python=python_version)
def mypy(session) -> None:
    """Type-check using mypy."""
    args = session.posargs or ["src", "tests", "docs/conf.py"]
    session.install(".")
    session.install("mypy", "pytest")
    session.run("mypy", *args)
    if not session.posargs:
        session.run(
            "mypy", f"--python-executable={sys.executable}", "noxfile.py"
        )


@session(python=python_version)
def lint(session: Session) -> None:
    session.install(".")
    session.install(
        "isort",
        "black[jupyter]",
        "autoflake8",
        "flake8",
        "docformatter[tomli]",
    )
    session.run("isort", ".")
    session.run("black", ".")
    session.run("docformatter", "--in-place", "--recursive", ".")
    session.run(
        "autoflake8",
        "--in-place",
        "--recursive",
        "--exclude",
        "__init__.py",
        ".",
    )
    session.run("flake8", ".")


@session(python=python_version)
def build(session):
    session.install("poetry")
    session.run("poetry", "install")
    session.run("poetry", "build")


@session(name="docs-build", python=python_version)
def docs_build(session: Session) -> None:
    """Build the documentation."""
    args = session.posargs or ["docs", "docs/_build"]
    if not session.posargs and "FORCE_COLOR" in os.environ:
        args.insert(0, "--color")

    session.install(".")
    session.install(
        "sphinx_book_theme",
        "sphinx",
        "jupyter-sphinx",
        "myst-parser",
    )

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-build", *args)
    session.run("quarto", "render", "docs/notebooks", external=True)


@session(name="docs-serve", python=python_version)
def docs_serve(session: Session) -> None:
    """Build and serve the documentation with live reloading on file
    changes."""
    args = session.posargs or ["docs", "docs/_build"]
    if not session.posargs and "FORCE_COLOR" in os.environ:
        args.insert(0, "--color")

    session.install(".")
    session.install(
        "sphinx_book_theme",
        "sphinx",
        "jupyter-sphinx",
        "sphinx-autobuild",
        "myst-parser",
    )

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-autobuild", *args)
