import nox


@nox.session(python=["3.10"])
def tests(session):
    session.install("poetry")
    session.run("poetry", "install")
    session.run("coverage", "run", "-m", "pytest")
    session.run("coverage", "report")


@nox.session
def lint(session):
    session.install("poetry")
    session.run("poetry", "install")
    session.run("black", "--check", ".")
    # session.run("flake8", ".") @ nox.session


@nox.session
def build(session):
    session.install("poetry")
    session.run("poetry", "install")
    session.run("poetry", "build")


@nox.session
def docs(session):
    session.install("poetry")
    session.run("poetry", "install")
    session.run("poetry", "run", "pdoc", "-o", "docs/", "genjax")
