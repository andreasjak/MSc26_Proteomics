"""
src
---
Top-level source package for the MSc26 proteomics project.

Sub-packages
------------
core
    Data utilities (subsetting, feature selection, correlation helpers).
styles
    Shared color palettes and visual style constants.
"""

from . import core, styles

__all__ = ["core", "styles"]