import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'UAV Dynamic Route Optimization Engine'
copyright = '2026, Autonomous Systems Logistics Lab'
author = 'Autonomous Systems Engineering Team'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'alabaster'
html_static_path = ['_static']

mathjax3_config = {
    'tex': {'tags': 'ams', 'useLabelIds': True}
}