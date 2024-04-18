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


@session(python=python_version)
def prepare(session):
    session.run_always(
        "poetry", "install", "--with", "dev", "--all-extras", external=True
    )


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
        "--ignore",
        "benchmarks",
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
        "--ignore",
        "benchmarks",
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
    """Execute jupyter notebooks as tests"""
    prepare(session)
    session.run(
        "poetry",
        "run",
        "pytest",
        "-n",
        "auto",
        "--nbmake",
        "notebooks",
    )


@session(python=python_version)
def safety(session) -> None:
    """Scan dependencies for insecure packages."""
    requirements = session.poetry.export_requirements()
    session.install("safety")
    session.run("safety", "check", "--full-report", f"--file={requirements}")


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
    session.run_always(
        "poetry",
        "install",
        "--with",
        "docs",
        "--with",
        "dev",
        "--all-extras",
        external=True,
    )
    build_dir = Path("site")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    session.run("poetry", "run", "mkdocs", "build", "--strict")


@session(name="docs-build", python=python_version)
def docs_build(session: Session) -> None:
    """Build the documentation."""
    session.run_always(
        "poetry",
        "install",
        "--with",
        "docs",
        "--with",
        "dev",
        "--all-extras",
        external=True,
    )
    build_dir = Path("site")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    session.run("poetry", "run", "mkdocs", "build", "--strict")
    session.run(
        "poetry", "run", "quarto", "render", "notebooks", "--execute", external=True
    )


@session(name="docs-serve", python=python_version)
def docs_serve(session: Session) -> None:
    """Build the documentation."""
    session.run_always(
        "poetry", "install", "--with", "docs", "--with", "dev", external=True
    )
    session.run("poetry", "run", "mkdocs", "serve")


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
