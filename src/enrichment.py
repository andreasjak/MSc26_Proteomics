"""Enrichment analysis helpers for Stage 7."""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

try:
	import gseapy as gp
except ImportError as exc:  # pragma: no cover
	gp = None
	_GSEAPY_IMPORT_ERROR = exc
else:
	_GSEAPY_IMPORT_ERROR = None


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


def _parse_overlap(value: object) -> tuple[int | None, int | None]:
	if value is None:
		return None, None
	text = str(value).strip()
	match = re.match(r"^(\d+)\s*/\s*(\d+)$", text)
	if not match:
		return None, None
	return int(match.group(1)), int(match.group(2))


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

	if raw.empty:
		return pd.DataFrame(
			columns=[
				"library",
				"term",
				"q_value",
				"p_value",
				"genes",
				"overlap_size",
				"term_size",
			]
		)

	term_col = _pick_column(raw, ["Term", "Pathway", "name"])
	q_col = _pick_column(raw, ["Adjusted P-value", "adj_p", "q_value", "fdr"])
	p_col = _pick_column(raw, ["P-value", "p_value"])
	genes_col = _pick_column(raw, ["Genes", "genes"])
	overlap_col = _pick_column(raw, ["Overlap", "overlap"])
	overlap_size_col = _pick_column(raw, ["overlap_size", "hits"])
	term_size_col = _pick_column(raw, ["term_size", "Term_size", "setSize"])

	if term_col is None or q_col is None:
		raise ValueError(
			"gseapy result is missing required columns for term/q-value parsing."
		)

	records: list[dict[str, object]] = []
	for _, row in raw.iterrows():
		term = str(row[term_col]).strip()
		if not term:
			continue

		genes_hit = _parse_genes(row[genes_col]) if genes_col is not None else []
		overlap_size, term_size = _parse_overlap(row[overlap_col]) if overlap_col else (None, None)

		if overlap_size is None and overlap_size_col is not None:
			value = pd.to_numeric(pd.Series([row[overlap_size_col]]), errors="coerce").iloc[0]
			overlap_size = int(value) if pd.notna(value) else None
		if term_size is None and term_size_col is not None:
			value = pd.to_numeric(pd.Series([row[term_size_col]]), errors="coerce").iloc[0]
			term_size = int(value) if pd.notna(value) else None

		if overlap_size is None:
			overlap_size = len(genes_hit)
		if term_size is None:
			term_size = max(len(genes_hit), overlap_size)

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
				"term_size": int(term_size),
			}
		)

	out = pd.DataFrame.from_records(records)
	if out.empty:
		return pd.DataFrame(
			columns=[
				"library",
				"term",
				"q_value",
				"p_value",
				"genes",
				"overlap_size",
				"term_size",
			]
		)

	return out.sort_values(["q_value", "p_value", "term"], kind="mergesort").reset_index(drop=True)


def compute_bes(
	term_df: pd.DataFrame,
	gene_list: list[str],
	c: float,
	tau: float,
	q_threshold: float,
) -> dict:
	"""Compute BES and contribution diagnostics for a term table."""
	if c <= 0:
		raise ValueError(f"c must be > 0, got {c}.")
	if not 0 <= tau <= 1:
		raise ValueError(f"tau must be in [0, 1], got {tau}.")
	if not 0 < q_threshold <= 1:
		raise ValueError(f"q_threshold must be in (0, 1], got {q_threshold}.")

	genes = _coerce_gene_list(gene_list)
	n_gene_list = len(genes)
	if n_gene_list == 0:
		raise ValueError("gene_list is empty after cleaning.")

	required = {"term", "q_value", "genes", "overlap_size", "term_size"}
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
			"tau": float(tau),
			"gene_list_size": int(n_gene_list),
			"per_term_contributions": [],
		}

	sig = sig.reset_index(drop=True)
	term_sets: list[set[str]] = [set(_parse_genes(v)) for v in sig["genes"].tolist()]
	m = len(term_sets)

	jaccard = np.zeros((m, m), dtype=float)
	for i in range(m):
		for j in range(i + 1, m):
			union = term_sets[i] | term_sets[j]
			jac = 0.0 if len(union) == 0 else len(term_sets[i] & term_sets[j]) / len(union)
			jaccard[i, j] = jac
			jaccard[j, i] = jac

	redundant_mask = jaccard >= tau
	np.fill_diagonal(redundant_mask, False)
	redundant_counts = redundant_mask.sum(axis=1)
	# Redundancy penalty: u_i = 1 / (1 + #{j != i: Jaccard(T_i, T_j) >= tau}).
	# See the BES definition in texfiles/ for the analytical form this realizes.
	u = np.ones(m) # 1.0 / (1.0 + redundant_counts)

	term_size = sig["term_size"].to_numpy(dtype=float)
	overlap_size = sig["overlap_size"].to_numpy(dtype=float)
	q_vals = sig["q_value"].to_numpy(dtype=float)

	term_size = np.where(term_size > 0, term_size, np.nan)
	overlap_size = np.where(overlap_size >= 0, overlap_size, 0.0)

	a = np.ones(m) # np.where(np.isfinite(term_size), 1.0 / np.log2(term_size + 1.0), 0.0)
	g = np.ones(m) # overlap_size / float(n_gene_list)
	s = np.minimum(-np.log10(np.clip(q_vals, np.finfo(float).tiny, 1.0)), float(c))

	weight = u * a * g
	contribution = weight * s
	bes_raw = float(np.nansum(contribution))

	per_term: list[dict[str, object]] = []
	for i in range(m):
		per_term.append(
			{
				"term": str(sig.loc[i, "term"]),
				"q_value": float(q_vals[i]),
				"overlap_size": int(overlap_size[i]),
				"term_size": int(term_size[i]) if np.isfinite(term_size[i]) else 0,
				"n_redundant_terms": int(redundant_counts[i]),
				"u": float(u[i]),
				"a": float(a[i]),
				"g": float(g[i]),
				"s": float(s[i]),
				"weight": float(weight[i]),
				"contribution": float(contribution[i]),
			}
		)

	per_term = sorted(per_term, key=lambda d: d["contribution"], reverse=True)

	return {
		"bes_raw": bes_raw,
		"n_terms_input": int(len(df)),
		"n_significant_terms": int(m),
		"q_threshold": float(q_threshold),
		"cap_c": float(c),
		"tau": float(tau),
		"gene_list_size": int(n_gene_list),
		"per_term_contributions": per_term,
	}


def permutation_null(
	background: list[str],
	gene_list_size: int,
	library: str,
	b_perm: int,
	c: float,
	tau: float,
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

	for i in range(total):
		sampled = rng.choice(bg, size=gene_list_size, replace=False).tolist()
		try:
			term_df = run_ora(sampled, bg, library)
			bes_info = compute_bes(
				term_df=term_df,
				gene_list=sampled,
				c=c,
				tau=tau,
				q_threshold=q_threshold,
			)
			null_values[i] = float(bes_info["bes_raw"])
		except Exception:
			null_values[i] = np.nan

		if progress_callback is not None:
			progress_callback(i + 1, total)

	return null_values


__all__ = ["run_ora", "compute_bes", "permutation_null"]
