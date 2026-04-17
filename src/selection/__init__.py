"""Selection method implementations and interface."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from .base import SelectionMethod, validate_selection_output
from .random import RandomSelection
from .ttest import TTestSelection


MethodFactory = Callable[[argparse.Namespace], SelectionMethod]


METHOD_REGISTRY: dict[str, MethodFactory] = {
	"ttest": lambda _: TTestSelection(),
	"random": lambda args: RandomSelection(
		n_significant=getattr(args, "random_significant", 30),
	),
}


__all__ = [
	"SelectionMethod",
	"RandomSelection",
	"TTestSelection",
	"MethodFactory",
	"METHOD_REGISTRY",
	"validate_selection_output",
]
