"""Selection method implementations and interface."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from .base import SelectionMethod, validate_selection_output
from .mutual_info import MutualInfoSelection
from .random import RandomSelection
from .ttest import TTestSelection
from .rf_shap import RFSHAPSelection


MethodFactory = Callable[[argparse.Namespace], SelectionMethod]


METHOD_REGISTRY: dict[str, MethodFactory] = {
	"ttest": lambda _: TTestSelection(),
	"random": lambda args: RandomSelection(
		n_significant=getattr(args, "random_significant", 20),
	),
	#"mi": lambda args: MutualInfoSelection(
	#	variant=getattr(args, "mi_variant", "adaptive"),
	#),
    "mi": lambda args: MutualInfoSelection(),
    "rf_shap": lambda _: RFSHAPSelection(),
}


__all__ = [
	"SelectionMethod",
	"RandomSelection",
	"TTestSelection",
	"MutualInfoSelection",
    "RFSHAPSelection",
	"MethodFactory",
	"METHOD_REGISTRY",
	"validate_selection_output",
]
