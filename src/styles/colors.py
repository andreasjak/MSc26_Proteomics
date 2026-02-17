# Color palette for data visualization across the project

# Primary colors
PRIMARY = "#D97706"  # Amber/Orange
SECONDARY = "#2F6F6D"  # Teal

# Complete Color Palette
PALETTE = {
    # Main distributions / baseline
    "primary":   "#D97706",  # Burnt Orange

    # Control / comparison groups
    "secondary": "#2F6F6D",  # Muted Teal

    # Severity / ARDS / risk
    "danger":    "#7A1F1F",  # Dark Red

    # Secondary group / alternative method
    "accent":    "#5B3A5D",  # Muted Plum

    # Highlights / annotations (sparsam!)
    "warning":   "#EAB308",  # Mustard Yellow

    # Structural / neutral elements (edges, outlines)
    "neutral":   "#7C4A1D",  # Warm Brown
}

# Color lists for categorical data
CATEGORICAL = [
    "#D97706",  # Burnt Orange (primary)
    "#2F6F6D",  # Muted Teal (control/contrast)
    "#5B3A5D",  # Muted Plum (secondary/method)
    "#7A1F1F",  # Dark Red (risk/severity)
    "#EAB308",  # Mustard Yellow (accent - sparsam)
    "#7C4A1D",  # Warm Brown (neutral/structure)
]

# Sequential colors for heatmaps etc
SEQUENTIAL_TEAL = [
    "#E6F2F1",  # very light teal
    "#CFE5E3",
    "#A9CFCC",
    "#7FB6B3",
    "#559B97",
    "#2F6F6D",  # muted teal
]

SEQUENTIAL_ORANGE = [
    "#FEF3E2",  # very light warm
    "#FDE6C8",
    "#FBCF9A",
    "#F7B56D",
    "#F09A3E",
    "#D97706",  # burnt orange
]


# ARDS/Sepsis specific
ARDS_COLORS = {
    "ARDS": "#5B3A5D",          # Dark Red (ARDS / outcome)
    "Not ARDS":"#EAB308",      # Muted Teal (control)
    "Mild ARDS": "#7A1F1F",     # Mustard (mild = "warning"/accent)
    "Moderate/Severe ARDS": "#7A1F1F",  # Dark Red
}


# Utility function to get colors for a specific context
def get_colors(context="categorical"):
    """
    Get color palette for a specific context.
    
    Parameters
    ----------
    context : str
        Type of color palette: 'categorical', 'sequential', 'ards', or 'palette'
    
    Returns
    -------
    list or dict
        Colors for the specified context
    """
    contexts = {
        "categorical": CATEGORICAL,
        "sequential_teal": SEQUENTIAL_TEAL,
        "sequential_orange": SEQUENTIAL_ORANGE,
        "ards": ARDS_COLORS,
        "palette": PALETTE,
    }
    return contexts.get(context, CATEGORICAL)
