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
