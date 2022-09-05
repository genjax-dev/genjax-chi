# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "genjax"
copyright = "2022, MIT Probabilistic Computing Project"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.inheritance_diagram",
    "jupyter_sphinx",
]


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None
    filename = info["module"].replace(".", "/")
    return "https://github.com/probcomp/genjax/tree/main/%s.py" % filename


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_theme_options = {
    "logo": "assets/logo.png",
    "github_user": "probcomp",
    "github_repo": "genjax",
    "github_button": True,
    "show_powered_by": False,
    "body_text_align": "center",
}
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
html_title = "Gen ⊗ JAX"

import os
import sys

sys.path.insert(0, os.path.abspath(".."))
package_path = os.path.abspath("../..")
os.environ["PYTHONPATH"] = ":".join(
    (package_path, os.environ.get("PYTHONPATH", ""))
)
