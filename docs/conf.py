# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'AdaptiveTSP'
author = 'Your Name'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'
html_static_path = ['_static']
