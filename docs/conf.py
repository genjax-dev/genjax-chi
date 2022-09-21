# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os
import inspect
import importlib


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "genjax"
copyright = "2022 MIT Probabilistic Computing Project"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.inheritance_diagram",
    "jupyter_sphinx",
]


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_title = "Gen ⊗ JAX"
html_logo = "_static/assets/logo.png"
html_theme_options = {
    "repository_url": "https://github.com/probcomp/genjax",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "repository_branch": "main",
    "logo_only": True,
}

autodoc_class_signature = "separated"

html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]

# Workaround for static content:
# https://github.com/sphinx-doc/sphinx/issues/2090


def env_get_outdated(app, env, added, changed, removed):
    return ["index"]


def setup(app):
    app.connect("env-get-outdated", env_get_outdated)


#####
# Linkcode resolve
#####

github_repo = "https://github.com/probcomp/genjax"

git_ref = None
try:
    git_ref = git("name-rev", "--name-only", "--no-undefined", "HEAD")
    git_ref = re.sub(r"^(remotes/[^/]+|tags)/", "", git_ref)
except Exception:
    pass

# (if no name found or relative ref, use commit hash instead)
if not git_ref or re.search(r"[\^~]", git_ref):
    try:
        git_ref = git("rev-parse", "HEAD")
    except Exception:
        git_ref = "main"

_genjax_module_path = os.path.dirname(importlib.util.find_spec("genjax").origin)  # type: ignore


def linkcode_resolve(domain, info):
    if domain != "py":
        return None

    try:
        obj: Any = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        obj = inspect.unwrap(obj)

        if isinstance(obj, property):
            obj = inspect.unwrap(obj.fget)  # type: ignore

        path = os.path.relpath(inspect.getsourcefile(obj), start=_genjax_module_path)  # type: ignore
        src, lineno = inspect.getsourcelines(obj)
    except Exception:
        return None

    path = f"{path}#L{lineno}-L{lineno + len(src) - 1}"
    return f"{github_repo}/blob/{git_ref}/genjax/{path}"


#####
# Path setup
#####

sys.path.insert(0, os.path.abspath(".."))
package_path = os.path.abspath("../..")
os.environ["PYTHONPATH"] = ":".join(
    (package_path, os.environ.get("PYTHONPATH", ""))
)
