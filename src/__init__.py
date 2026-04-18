"""
src
---
Top-level source package for the MSc26 proteomics project.

Modules
-------
logging_utils
    Shared logger setup used by pipeline scripts.
styles
    Shared color palettes and visual style constants. Imported lazily via
    ``from src import styles`` to avoid pulling matplotlib into CLI scripts.
"""

from .logging_utils import setup_logging

__all__ = ["setup_logging"]