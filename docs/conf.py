import os
import sys
sys.path.insert(0, os.path.abspath(".."))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'lightbench'
copyright = '2025, Filip Naudot'
author = 'Filip Naudot'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc','sphinx.ext.napoleon']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/filipnaudot/lightbench",
    "repository_branch": "main",
    "path_to_docs": "docs",

    "use_repository_button": True,
    "use_edit_page_button": False,

    "logo": {
        "image_light": "./readme_assets/lightbench_logo_lightmode.png",
        "image_dark": "./readme_assets/lightbench_logo_darkmode.png",
        # "text": "lightbench",
        "alt_text": "lightbench documentation",
    }
}

html_static_path = ['_static']

# Don not crash on minor nitpicks.
nitpicky = False