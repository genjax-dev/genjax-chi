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
nox.options.sessions = ("tests", "lint", "build")


@session(python=python_version)
def tests(session):
    session.run_always("poetry", "install", "--with", "dev", external=True)
    session.run(
        "poetry",
        "run",
        "pytest",
        "--benchmark-disable",
        "--ignore",
        "scratch",
        "--ignore",
        "benchmarks",
        "-n",
        "auto",
    )


@session(python=python_version)
def coverage(session):
    session.run_always("poetry", "install", "--with", "dev", external=True)
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
        "--ignore",
        "benchmarks",
    )
    session.run("poetry", "run", "coverage", "json")
    session.run("poetry", "run", "coverage", "report")


@session(python=python_version)
def benchmark(session):
    session.run_always("poetry", "install", "--with", "dev", external=True)
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
    session.run_always("poetry", "install", "--with", "dev", external=True)
    if session.posargs:
        args = [package, *session.posargs]
    else:
        args = [f"--modname={package}", "--command=all"]
        if "FORCE_COLOR" in os.environ:
            args.append("--colored=1")

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
    session.run_always("poetry", "install", "--with", "dev", external=True)
    session.run("mypy", *args)
    if not session.posargs:
        session.run("mypy", f"--python-executable={sys.executable}", "noxfile.py")


@session(python=python_version)
def lint(session: Session) -> None:
    session.run_always("poetry", "install", "--with", "dev", external=True)
    session.install(
        "isort", "black[jupyter]", "autoflake8", "flake8", "docformatter[tomli]"
    )

    # Source
    session.run("isort", "src")
    session.run("black", "src")
    session.run("docformatter", "--in-place", "--recursive", "src")
    session.run(
        "autoflake8", "--in-place", "--recursive", "--exclude", "__init__.py", "src"
    )
    session.run("flake8", "src")

    # Tests
    session.run("isort", "tests")
    session.run("black", "tests")
    session.run(
        "autoflake8", "--in-place", "--recursive", "--exclude", "__init__.py", "tests"
    )
    session.run("flake8", "tests")

    # Notebooks
    session.run("isort", "notebooks")
    session.run("black", "notebooks")
    session.run("docformatter", "--in-place", "--recursive", "notebooks")
    session.run(
        "autoflake8",
        "--in-place",
        "--recursive",
        "--exclude",
        "__init__.py",
        "notebooks",
    )
    session.run("flake8", "notebooks")


@session(python=python_version)
def build(session):
    session.run_always(
        "poetry",
        "install",
        "--with",
        "dev",
        external=True,
    )
    session.run("poetry", "build")


@session(name="docs-build", python=python_version)
def docs_build(session: Session) -> None:
    """Build the documentation."""
    session.run_always(
        "poetry", "install", "--with", "docs", "--with", "dev", external=True
    )
    session.install("mkdocs")
    session.install(
        "mkdocs-material @ git+https://github.com/probcomp/mkdocs-material-insiders"
    )
    session.install(
        "mkdocstrings-python @ git+https://github.com/pawamoy-insiders/mkdocstrings-python.git@1.5.2.1.3.0"
    )

    build_dir = Path("site")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    session.run("mkdocs", "build")
    session.run("quarto", "render", "notebooks", external=True)


@session(name="docs-serve", python=python_version)
def docs_serve(session: Session) -> None:
    """Build the documentation."""
    session.run_always(
        "poetry", "install", "--with", "docs", "--with", "dev", external=True
    )
    session.install("mkdocs")
    session.install(
        "mkdocs-material @ git+https://github.com/probcomp/mkdocs-material-insiders"
    )
    session.install(
        "mkdocstrings-python @ git+https://github.com/pawamoy-insiders/mkdocstrings-python.git@1.5.2.1.3.0"
    )
    session.run("mkdocs", "serve")


@session(name="notebooks-serve", python=python_version)
def notebooks_serve(session: Session) -> None:
    """Build the documentation."""
    session.run_always("poetry", "install", "--with", "dev", external=True)
    session.run("quarto", "preview", "notebooks", external=True)


@session(name="jupyter", python=python_version)
def jupyter(session: Session) -> None:
    """Build the documentation."""
    session.run_always("poetry", "install", "--with", "dev", external=True)
    session.run("jupyter-lab", external=True)
