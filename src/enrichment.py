"""Enrichment analysis helpers for Stage 7."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

import numpy as np
import pandas as pd
import requests

try:
	import gseapy as gp
except ImportError as exc:  # pragma: no cover
	gp = None
	_GSEAPY_IMPORT_ERROR = exc
else:
	_GSEAPY_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

_LIBRARY_CACHE: dict[str, dict[str, int]] = {}

ENRICHR_LIBRARY_URL = "https://maayanlab.cloud/Enrichr/geneSetLibrary"


def get_enrichr_library(name: str) -> dict[str, list[str]]:
	"""Fetch a GMT-format gene-set library directly from Enrichr.

	Bypasses ``gp.get_library`` because of a gseapy bug: it passes
	``decode_unicode="utf-8"`` (a str) to ``Response.iter_lines`` (bool
	expected), so the iterator yields bytes and the subsequent
	``.split("\\t")`` fails with TypeError on charset-less responses.
	"""
	response = requests.get(
		ENRICHR_LIBRARY_URL,
		params={"mode": "text", "libraryName": name},
		timeout=60,
	)
	response.raise_for_status()
	response.encoding = "utf-8"

	library: dict[str, list[str]] = {}
	for line in response.text.splitlines():
		if not line.strip():
			continue
		parts = line.split("\t")
		if len(parts) < 3:
			continue
		term = parts[0]
		genes = [g.split(",", 1)[0].strip() for g in parts[2:]]
		genes = [g for g in genes if g]
		library[term] = genes
	return library


def _coerce_gene_list(values: Iterable[str]) -> list[str]:
	seen: set[str] = set()
	genes: list[str] = []
	for value in values:
		gene = str(value).strip()
		if not gene:
			continue
		if gene in seen:
			continue
		seen.add(gene)
		genes.append(gene)
	return genes


def _parse_genes(value: object) -> list[str]:
	if value is None or (isinstance(value, float) and np.isnan(value)):
		return []
	if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
		return _coerce_gene_list(str(v) for v in list(value))

	text = str(value).strip()
	if not text:
		return []
	for sep in (";", ",", "|"):
		if sep in text:
			return _coerce_gene_list(part.strip() for part in text.split(sep))
	return [text]


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
	lower_to_actual = {c.lower(): c for c in df.columns}
	for candidate in candidates:
		actual = lower_to_actual.get(candidate.lower())
		if actual is not None:
			return actual
	return None


def _parse_overlap_size(value: object) -> int | None:
	"""Parse the overlap (numerator) from gseapy's Overlap field.

	Accepts both ``"5/87"`` and ``"5/87/11000"`` forms. Only the numerator is
	returned; ``term_size`` is sourced separately from the gene-set library.
	"""
	if value is None:
		return None
	text = str(value).strip()
	match = re.match(r"^(\d+)\s*/", text)
	if not match:
		return None
	return int(match.group(1))


def _get_term_sizes(library: str) -> dict[str, int]:
	"""Return ``{term_name: term_size}`` for a gene-set library, cached per process."""
	if library in _LIBRARY_CACHE:
		return _LIBRARY_CACHE[library]
	try:
		members = get_enrichr_library(library)
	except Exception as exc:
		logger.warning(
			"get_enrichr_library failed for %r: %s: %s — term_size will be NaN",
			library,
			type(exc).__name__,
			exc,
		)
		_LIBRARY_CACHE[library] = {}
		return _LIBRARY_CACHE[library]
	sizes = {term: len(genes) for term, genes in members.items()}
	_LIBRARY_CACHE[library] = sizes
	return sizes


def run_ora(gene_list: list[str], background: list[str], library: str) -> pd.DataFrame:
	"""Run ORA via gseapy and return normalized term table."""
	if gp is None:  # pragma: no cover
		raise ImportError(
			"gseapy is required for enrichment analysis. Install with: conda install -c conda-forge gseapy"
		) from _GSEAPY_IMPORT_ERROR

	genes = _coerce_gene_list(gene_list)
	bg = _coerce_gene_list(background)

	if not genes:
		raise ValueError("gene_list is empty after cleaning.")
	if not bg:
		raise ValueError("background is empty after cleaning.")

	term_size_lookup = _get_term_sizes(library)

	kwargs = {
		"gene_list": genes,
		"gene_sets": library,
		"background": bg,
		"outdir": None,
		"no_plot": True,
		"cutoff": 1.0,
	}

	if hasattr(gp, "enrich"):
		result = gp.enrich(**kwargs)
	else:  # pragma: no cover
		result = gp.enrichr(
			gene_list=genes,
			gene_sets=library,
			background=bg,
			organism="Human",
			outdir=None,
			no_plot=True,
			cutoff=1.0,
		)

	raw: pd.DataFrame | None = None
	if isinstance(result, pd.DataFrame):
		raw = result.copy()
	elif hasattr(result, "results") and isinstance(result.results, pd.DataFrame):
		raw = result.results.copy()
	elif hasattr(result, "res2d") and isinstance(result.res2d, pd.DataFrame):
		raw = result.res2d.copy()

	if raw is None:
		raise ValueError("Unable to parse gseapy enrichment result table.")

	empty_template = pd.DataFrame(
		{
			"library": pd.Series(dtype="object"),
			"term": pd.Series(dtype="object"),
			"q_value": pd.Series(dtype="float64"),
			"p_value": pd.Series(dtype="float64"),
			"genes": pd.Series(dtype="object"),
			"overlap_size": pd.Series(dtype="int64"),
			"term_size": pd.Series(dtype="float64"),
		}
	)

	if raw.empty:
		return empty_template

	term_col = _pick_column(raw, ["Term", "Pathway", "name"])
	q_col = _pick_column(raw, ["Adjusted P-value", "adj_p", "q_value", "fdr"])
	p_col = _pick_column(raw, ["P-value", "p_value"])
	genes_col = _pick_column(raw, ["Genes", "genes"])
	overlap_col = _pick_column(raw, ["Overlap", "overlap"])
	overlap_size_col = _pick_column(raw, ["overlap_size", "hits"])

	if term_col is None or q_col is None:
		raise ValueError(
			"gseapy result is missing required columns for term/q-value parsing."
		)

	missing_terms_logged = 0
	records: list[dict[str, object]] = []
	for _, row in raw.iterrows():
		term = str(row[term_col]).strip()
		if not term:
			continue

		genes_hit = _parse_genes(row[genes_col]) if genes_col is not None else []

		overlap_size: int | None = None
		if overlap_col is not None:
			overlap_size = _parse_overlap_size(row[overlap_col])
		if overlap_size is None and overlap_size_col is not None:
			value = pd.to_numeric(pd.Series([row[overlap_size_col]]), errors="coerce").iloc[0]
			overlap_size = int(value) if pd.notna(value) else None
		if overlap_size is None:
			overlap_size = len(genes_hit)

		term_size_int = term_size_lookup.get(term)
		if term_size_int is None:
			if missing_terms_logged < 5:
				logger.warning(
					"term %r not found in library %r lookup; term_size=NaN",
					term,
					library,
				)
			missing_terms_logged += 1
			term_size_value: float = float("nan")
		else:
			term_size_value = float(term_size_int)

		q_value = pd.to_numeric(pd.Series([row[q_col]]), errors="coerce").iloc[0]
		p_value = (
			pd.to_numeric(pd.Series([row[p_col]]), errors="coerce").iloc[0]
			if p_col is not None
			else np.nan
		)

		records.append(
			{
				"library": library,
				"term": term,
				"q_value": float(q_value) if pd.notna(q_value) else np.nan,
				"p_value": float(p_value) if pd.notna(p_value) else np.nan,
				"genes": ";".join(genes_hit),
				"overlap_size": int(overlap_size),
				"term_size": term_size_value,
			}
		)

	if missing_terms_logged > 5:
		logger.warning(
			"%d terms total were missing from library %r lookup (only first 5 logged individually).",
			missing_terms_logged,
			library,
		)

	out = pd.DataFrame.from_records(records)
	if out.empty:
		return empty_template

	return out.sort_values(["q_value", "p_value", "term"], kind="mergesort").reset_index(drop=True)


def compute_bes(
	term_df: pd.DataFrame,
	gene_list: list[str],
	c: float,
	q_threshold: float,
) -> dict:
	"""Compute BES with uniform weights w_i = 1.

	BES = sum_i s_i over terms with q_i < q_threshold, where
	s_i = min(-log10(q_i), c). See ``texfiles/validation.tex``
	§sec:enrichment_validation.
	"""
	if c <= 0:
		raise ValueError(f"c must be > 0, got {c}.")
	if not 0 < q_threshold <= 1:
		raise ValueError(f"q_threshold must be in (0, 1], got {q_threshold}.")

	genes = _coerce_gene_list(gene_list)
	n_gene_list = len(genes)
	if n_gene_list == 0:
		raise ValueError("gene_list is empty after cleaning.")

	required = {"term", "q_value", "overlap_size", "term_size"}
	missing = required.difference(term_df.columns)
	if missing:
		raise ValueError(f"term_df missing required columns: {sorted(missing)}")

	df = term_df.copy()
	df["q_value"] = pd.to_numeric(df["q_value"], errors="coerce")
	df["overlap_size"] = pd.to_numeric(df["overlap_size"], errors="coerce")
	df["term_size"] = pd.to_numeric(df["term_size"], errors="coerce")

	sig = df[df["q_value"].notna() & (df["q_value"] < q_threshold)].copy()
	if sig.empty:
		return {
			"bes_raw": 0.0,
			"n_terms_input": int(len(df)),
			"n_significant_terms": 0,
			"q_threshold": float(q_threshold),
			"cap_c": float(c),
			"gene_list_size": int(n_gene_list),
			"per_term_contributions": [],
		}

	sig = sig.reset_index(drop=True)
	q_vals = sig["q_value"].to_numpy(dtype=float)
	overlap_size = sig["overlap_size"].to_numpy(dtype=float)
	term_size = sig["term_size"].to_numpy(dtype=float)

	s = np.minimum(-np.log10(np.clip(q_vals, np.finfo(float).tiny, 1.0)), float(c))
	bes_raw = float(np.sum(s))

	per_term: list[dict[str, object]] = []
	for i in range(len(sig)):
		per_term.append(
			{
				"term": str(sig.loc[i, "term"]),
				"q_value": float(q_vals[i]),
				"overlap_size": int(overlap_size[i]) if np.isfinite(overlap_size[i]) else 0,
				"term_size": int(term_size[i]) if np.isfinite(term_size[i]) else None,
				"s": float(s[i]),
			}
		)

	per_term.sort(key=lambda d: d["s"], reverse=True)

	return {
		"bes_raw": bes_raw,
		"n_terms_input": int(len(df)),
		"n_significant_terms": int(len(sig)),
		"q_threshold": float(q_threshold),
		"cap_c": float(c),
		"gene_list_size": int(n_gene_list),
		"per_term_contributions": per_term,
	}


def permutation_null(
	background: list[str],
	gene_list_size: int,
	library: str,
	b_perm: int,
	c: float,
	q_threshold: float,
	seed: int,
	progress_callback=None,
) -> np.ndarray:
	"""Compute permutation-null BES values via random background sampling.

	``progress_callback``, if provided, is invoked as ``callback(done, total)``
	after every iteration so callers can log long-running loops.
	"""
	bg = _coerce_gene_list(background)
	if gene_list_size <= 0:
		raise ValueError(f"gene_list_size must be > 0, got {gene_list_size}.")
	if gene_list_size > len(bg):
		raise ValueError(
			f"gene_list_size={gene_list_size} exceeds background size={len(bg)}."
		)
	if b_perm <= 0:
		raise ValueError(f"b_perm must be > 0, got {b_perm}.")

	rng = np.random.default_rng(seed)
	total = int(b_perm)
	null_values = np.full(total, np.nan, dtype=float)
	n_failed = 0

	for i in range(total):
		sampled = rng.choice(bg, size=gene_list_size, replace=False).tolist()
		try:
			term_df = run_ora(sampled, bg, library)
			bes_info = compute_bes(
				term_df=term_df,
				gene_list=sampled,
				c=c,
				q_threshold=q_threshold,
			)
			null_values[i] = float(bes_info["bes_raw"])
		except Exception as exc:
			n_failed += 1
			logger.warning(
				"permutation_null iteration %d/%d failed: %s: %s",
				i + 1,
				total,
				type(exc).__name__,
				exc,
			)
			null_values[i] = np.nan

		if progress_callback is not None:
			progress_callback(i + 1, total)

	if n_failed > 0:
		logger.warning(
			"permutation_null: %d/%d iterations failed (NaN-filled).",
			n_failed,
			total,
		)

	return null_values


__all__ = ["run_ora", "compute_bes", "permutation_null"]
