# conftest.py
import sys
import os

# Inject current working directory into Python search path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))