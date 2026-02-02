# Color palette for data visualization across the project

# Primary colors
PRIMARY = "#D97706"  # Amber/Orange
SECONDARY = "#2F6F6D"  # Teal

# Extended palette
PALETTE = {
    "primary": "#D97706",      # Amber/Orange
    "secondary": "#2F6F6D",    # Teal
    "accent": "#7C3AED",       # Purple
    "success": "#059669",      # Green
    "warning": "#D97706",      # Amber/Orange
    "danger": "#DC2626",       # Red
    "info": "#0891B2",         # Cyan
}

# Color lists for categorical data
CATEGORICAL = [
    "#D97706",  # Amber
    "#2F6F6D",  # Teal
    "#7C3AED",  # Purple
    "#059669",  # Green
    "#DC2626",  # Red
    "#0891B2",  # Cyan
    "#F97316",  # Orange
    "#EC4899",  # Pink
]

# Sequential palette (light to dark)
SEQUENTIAL = [
    "#F3E8FF",  # Very light purple
    "#E9D5FF",  # Light purple
    "#D8B4FE",  # Medium-light purple
    "#C084FC",  # Medium purple
    "#A855F7",  # Medium-dark purple
    "#7C3AED",  # Dark purple
    "#6D28D9",  # Very dark purple
]

# ARDS/Sepsis specific
ARDS_COLORS = {
    "ARDS": "#DC2626",        # Red for ARDS
    "Not ARDS": "#059669",    # Green for Not ARDS
    "Mild ARDS": "#F59E0B",   # Amber for Mild ARDS
    "Moderate/Severe ARDS": "#DC2626",  # Red for Moderate/Severe ARDS
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
        "sequential": SEQUENTIAL,
        "ards": ARDS_COLORS,
        "palette": PALETTE,
    }
    return contexts.get(context, CATEGORICAL)
