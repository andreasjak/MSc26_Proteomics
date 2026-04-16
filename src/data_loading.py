"""Data loading and cohort filtering for Stage 2.

This module provides the analysis-cohort loader used by the pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PATIENT_ID_CANDIDATES = (
	"patient_id",
	"PatientID",
	"subject_id",
	"SubjectID",
	"SampleId",
	"SampleID",
	"SsfExtId",
	"ExtIdentifier",
)

PROTEIN_PREFIX = "seq."


def _coerce_bool(series: pd.Series) -> pd.Series:
	"""Coerce mixed boolean-like values to pandas nullable booleans."""
	coerced = pd.Series(pd.NA, index=series.index, dtype="boolean")

	# Numeric booleans (0/1)
	numeric = pd.to_numeric(series, errors="coerce")
	coerced.loc[numeric == 1] = True
	coerced.loc[numeric == 0] = False

	# String booleans
	text = series.astype(str).str.strip().str.lower()
	truthy = {"true", "t", "yes", "y", "1"}
	falsy = {"false", "f", "no", "n", "0"}
	coerced.loc[text.isin(truthy)] = True
	coerced.loc[text.isin(falsy)] = False

	# Preserve explicit missing values as missing
	coerced.loc[series.isna()] = pd.NA
	return coerced


def _is_not_mild(row: pd.Series) -> bool | None:
	"""Resolve not-mild ARDS from available severity fields.

	Uses any available value from ``ards_severity``, ``ards_mild``, and
	``ards_notmild``. If provided signals are contradictory or all missing,
	returns ``None`` so the row can be dropped from ARDS-positive cohorting.
	"""
	votes_not_mild: list[bool] = []

	if "ards_severity" in row.index and pd.notna(row["ards_severity"]):
		votes_not_mild.append(str(row["ards_severity"]).strip().lower() != "mild")

	if "ards_mild" in row.index and pd.notna(row["ards_mild"]):
		mild = _coerce_bool(pd.Series([row["ards_mild"]], dtype="object")).iloc[0]
		if pd.notna(mild):
			votes_not_mild.append(not bool(mild))

	if "ards_notmild" in row.index and pd.notna(row["ards_notmild"]):
		not_mild = _coerce_bool(pd.Series([row["ards_notmild"]], dtype="object")).iloc[0]
		if pd.notna(not_mild):
			votes_not_mild.append(bool(not_mild))

	if not votes_not_mild:
		return None

	if len(set(votes_not_mild)) > 1:
		return None

	return votes_not_mild[0]


def _choose_patient_id_column(df: pd.DataFrame) -> str | None:
	for col in PATIENT_ID_CANDIDATES:
		if col in df.columns:
			return col
	return None


def _protein_columns(df: pd.DataFrame) -> list[str]:
	return [col for col in df.columns if col.startswith(PROTEIN_PREFIX)]


def _build_cohort_dataframe(
	raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, int | str]]:
	"""Filter raw data to analysis cohort and return a model-ready table.

	Output columns are ``patient_id``, all protein columns, and ``y``.
	"""
	if "Sepsis" not in raw_df.columns or "ards" not in raw_df.columns:
		raise ValueError("Input data must contain 'Sepsis' and 'ards' columns.")

	audit: dict[str, int | str] = {"rows_input": int(len(raw_df))}

	sepsis = _coerce_bool(raw_df["Sepsis"])
	ards = _coerce_bool(raw_df["ards"])

	missing_required = sepsis.isna() | ards.isna()
	audit["dropped_missing_sepsis_or_ards"] = int(missing_required.sum())

	labelled = raw_df.loc[~missing_required].copy()
	labelled["_sepsis"] = sepsis.loc[~missing_required].astype(bool)
	labelled["_ards"] = ards.loc[~missing_required].astype(bool)

	non_sepsis = ~labelled["_sepsis"]
	audit["dropped_non_sepsis"] = int(non_sepsis.sum())
	sepsis_only = labelled.loc[~non_sepsis].copy()

	cohort_non_ards = sepsis_only.loc[~sepsis_only["_ards"]].copy()
	audit["kept_sepsis_no_ards"] = int(len(cohort_non_ards))

	ards_positive = sepsis_only.loc[sepsis_only["_ards"]].copy()
	resolved_not_mild = ards_positive.apply(_is_not_mild, axis=1).astype("boolean")

	missing_or_conflicting = resolved_not_mild.isna()
	mild = resolved_not_mild == False

	audit["dropped_missing_or_conflicting_severity"] = int(missing_or_conflicting.sum())
	audit["dropped_mild_ards"] = int(mild.sum())

	cohort_ards = ards_positive.loc[resolved_not_mild.fillna(False)].copy()
	audit["kept_sepsis_not_mild_ards"] = int(len(cohort_ards))

	cohort_non_ards["y"] = 0
	cohort_ards["y"] = 1

	combined = pd.concat([cohort_non_ards, cohort_ards], ignore_index=True)

	patient_col = _choose_patient_id_column(combined)
	audit["patient_id_source"] = patient_col if patient_col is not None else "generated_row_index"

	if patient_col is not None:
		patient_ids = combined[patient_col].astype("string")
		generated_fallback = pd.Series(
			[f"row_{i}" for i in combined.index],
			index=combined.index,
			dtype="string",
		)
		patient_ids = patient_ids.where(patient_ids.notna(), generated_fallback)
	else:
		patient_ids = pd.Series(
			[f"row_{i}" for i in combined.index],
			index=combined.index,
			dtype="string",
		)

	protein_ids = _protein_columns(combined)
	if not protein_ids:
		raise ValueError(
			"No protein columns found. Expected SomaScan columns starting with 'seq.'."
		)

	proteins = combined[protein_ids].apply(pd.to_numeric, errors="coerce")

	cohort_table = pd.concat(
		[
			patient_ids.rename("patient_id"),
			proteins,
			combined["y"].astype(int),
		],
		axis=1,
	)

	audit["rows_output"] = int(len(cohort_table))
	audit["n_proteins"] = int(len(protein_ids))
	return cohort_table, protein_ids, audit


def load_cohort_table(
	data_path: Path,
) -> tuple[pd.DataFrame, list[str], dict[str, int | str]]:
	"""Load raw data and return filtered cohort table with audit metadata."""
	raw_df = pd.read_csv(data_path)
	return _build_cohort_dataframe(raw_df)


def load_cohort(data_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
	"""Return X, y, protein_ids, patient_ids for the Stage 2 cohort.

	Returns
	-------
	tuple[np.ndarray, np.ndarray, list[str], list[str]]
		X is shape (n_patients, n_proteins), y is binary ARDS label,
		protein_ids preserves feature order, and patient_ids preserves row order.
	"""
	cohort_table, protein_ids, _ = load_cohort_table(data_path)

	X = cohort_table[protein_ids].to_numpy()
	y = cohort_table["y"].to_numpy(dtype=int)
	patient_ids = cohort_table["patient_id"].astype(str).tolist()
	return X, y, protein_ids, patient_ids

