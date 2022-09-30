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

import nox


@nox.session(python=["3.10"])
def test(session):
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
        "--benchmark-disable-gc",
        "--benchmark-min-rounds",
        "5000",
    )
    session.run("coverage", "report")


@nox.session(python=["3.10"])
def lint(session):
    session.install("poetry")
    session.run("poetry", "install")
    session.run("black", ".")
    session.run(
        "autoflake8",
        "--in-place",
        "--recursive",
        "--exclude",
        "__init__.py",
        ".",
    )
    session.run("flake8", ".")


@nox.session(python=["3.10"])
def build(session):
    session.install("poetry")
    session.run("poetry", "install")
    session.run("poetry", "build")


@nox.session(python=["3.10"])
def docs(session):
    session.install("poetry")
    session.run("poetry", "install")
    session.run(
        "poetry",
        "run",
        "sphinx-build",
        "-b",
        "html",
        "docs",
        "docs/_build",
    )


@nox.session(python=["3.10"])
def extern_gen_fn(session):
    session.install("poetry")
    session.run("poetry", "install")
    session.run(
        "poetry",
        "run",
        "pip",
        "install",
        "examples/exposing_c++_gen_fn",
    )
