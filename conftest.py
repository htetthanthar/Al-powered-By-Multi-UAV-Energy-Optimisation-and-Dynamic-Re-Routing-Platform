"""Make the application source directory importable for the repository test suite."""

import sys
from pathlib import Path


SOURCE_DIR = Path(__file__).parent / "All sensor"
sys.path.insert(0, str(SOURCE_DIR))
