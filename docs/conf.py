# Copyright (c) 2026, The Folax Authors.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. All advertising materials mentioning features or use of this software must
#    display the following acknowledgement:
#      This product includes software developed by Reza Najian Asl, Shahed Rezaei.

# 4. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY COPYRIGHT HOLDER "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Configuration file for the Sphinx documentation builder."""


# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
import doctest

sys.path.insert(0, os.path.abspath('..'))
# Include local extension.
# sys.path.append(os.path.abspath('./_ext'))

# patch sphinx
# -- Project information -----------------------------------------------------

project = 'Folax'
copyright = '2026, The Folax authors'  # pylint: disable=redefined-builtin
author = 'The Folax authors'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
  'sphinx.ext.autodoc',
  'sphinx.ext.autosummary',
  'sphinx.ext.autosectionlabel',
  'sphinx.ext.doctest',
  'sphinx.ext.intersphinx',
  'sphinx.ext.mathjax',
  'sphinx.ext.napoleon',
  'sphinx.ext.viewcode',
  'myst_nb',
  'sphinx_design',
]

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames.
# Note: important to list ipynb before md here: we have both md and ipynb
# copies of each notebook, and myst will choose which to convert based on
# the order in the source_suffix list. Notebooks which are not executed have
# outputs stored in ipynb but not in md, so we must convert the ipynb.
source_suffix = ['.rst', '.ipynb', '.md']

autosummary_generate = True

master_doc = 'index'

autodoc_typehints = 'none'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'pydata_sphinx_theme'
html_theme = 'sphinx_book_theme'
# html_css_files = ['css/flax_theme.css']

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = './folax.png'
html_favicon = './folax.png'

# title of the website
html_title = ''

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named 'default.css' will overwrite the builtin 'default.css'.
html_static_path = ['_static']


html_theme_options = {
  'repository_url': 'https://github.com/Neural-Mechanics-Lab/folax',
  'use_repository_button': True,  # add a 'link to repository' button
  'use_issues_button': False,  # add an 'Open an Issue' button
  'path_to_docs': (
    'docs'
  ),  # used to compute the path to launch notebooks in colab
  'launch_buttons': {
    'colab_url': 'https://colab.research.google.com/',
  },
  'show_navbar_depth': 1
}


# -- Options for myst ----------------------------------------------
# uncomment line below to avoid running notebooks during development
nb_execution_mode = os.environ.get("NB_EXECUTION_MODE", 'off')
# Notebook cell execution timeout; defaults to 30.
nb_execution_timeout = 100
# List of patterns, relative to source directory, that match notebook
# files that will not be executed.
myst_enable_extensions = ['dollarmath']
nb_execution_excludepatterns = [
  'quick_start.ipynb'
]
# raise exceptions on execution so CI can catch errors
nb_execution_allow_errors = False
nb_execution_raise_on_error = True