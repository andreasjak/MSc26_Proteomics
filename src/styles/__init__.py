"""
src.styles
----------
Shared visual style definitions for the MSc26 proteomics project.

Modules
-------
colors
    Project color palette (PALETTE, CATEGORICAL, SEQUENTIAL_TEAL,
    SEQUENTIAL_ORANGE, ARDS_COLORS) and the :func:`get_colors` helper.
"""

from .colors import (
    PRIMARY,
    SECONDARY,
    PALETTE,
    CATEGORICAL,
    SEQUENTIAL_TEAL,
    SEQUENTIAL_ORANGE,
    ARDS_COLORS,
    get_colors,
)

__all__ = [
    "PRIMARY",
    "SECONDARY",
    "PALETTE",
    "CATEGORICAL",
    "SEQUENTIAL_TEAL",
    "SEQUENTIAL_ORANGE",
    "ARDS_COLORS",
    "get_colors",
]