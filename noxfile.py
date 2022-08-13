import nox


@nox.session(python=["3.10"])
def tests(session):
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
    )
    session.run("coverage", "report")


@nox.session
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


@nox.session
def build(session):
    session.install("poetry")
    session.run("poetry", "install")
    session.run("poetry", "build")


@nox.session
def docs(session):
    session.install("poetry")
    session.run("poetry", "install")
    session.run(
        "poetry", "run", "pdoc", "-t", "template", "-o", "docs/", "genjax"
    )
