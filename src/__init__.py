"""
src
---
Top-level source package for the MSc26 proteomics project.

Modules
-------
logging_utils
    Shared logger setup used by pipeline scripts.
styles
    Shared color palettes and visual style constants.
"""

from . import styles
from .logging_utils import setup_logging

__all__ = ["styles", "setup_logging"]