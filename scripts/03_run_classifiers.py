"""Run Stage 6 classifier validation for a selection method."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.classifiers import build_logreg, build_rf, build_xgb
from src.config import DATA_PROCESSED, RANDOM_SEED, RESULTS_DIR, TOPK_VALUES
from src.logging_utils import setup_logging
from src.metrics import compute_auc, compute_aupr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate LR, RF, XGB per split and top-k for one method.",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="Selection method name; used to locate results/selection/<method>/selections.parquet.",
    )
    parser.add_argument(
        "--cohort-path",
        type=Path,
        default=DATA_PROCESSED / "cohort.parquet",
        help="Path to prepared cohort parquet (default: data/processed/cohort.parquet).",
    )
    parser.add_argument(
        "--splits-path",
        type=Path,
        default=DATA_PROCESSED / "splits.pkl",
        help="Path to split cache pickle (default: data/processed/splits.pkl).",
    )
    parser.add_argument(
        "--selections-path",
        type=Path,
        default=None,
        help="Optional explicit path to selections.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "classifier",
        help="Base output directory (default: results/classifier).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=None,
        help="Optional split-count override (defaults to all compatible splits).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Global seed for per-split model seeds (default: {RANDOM_SEED}).",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="If set, write logs to file via shared logging utility.",
    )
    return parser.parse_args()


def _load_cohort(cohort_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not cohort_path.exists():
        raise FileNotFoundError(
            f"Cohort file not found at {cohort_path}. Run scripts/00_prepare_data.py first."
        )

    cohort = pd.read_parquet(cohort_path)
    if "y" not in cohort.columns:
        raise ValueError("Cohort parquet must contain a 'y' column.")

    protein_cols = [c for c in cohort.columns if c not in {"patient_id", "y"}]
    if not protein_cols:
        raise ValueError("No protein columns found in cohort parquet.")

    X = cohort[protein_cols].to_numpy(dtype=float)
    y = cohort["y"].to_numpy(dtype=int)
    return X, y


def _load_splits(splits_path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Split cache not found at {splits_path}. Run scripts/01_generate_splits.py first."
        )

    with splits_path.open("rb") as f:
        raw = pickle.load(f)

    if not isinstance(raw, list) or not raw:
        raise ValueError("Split cache must be a non-empty list of (train_idx, test_idx).")

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for i, pair in enumerate(raw):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(f"Split #{i} is not a valid (train_idx, test_idx) tuple.")
        splits.append(
            (
                np.asarray(pair[0], dtype=np.int64),
                np.asarray(pair[1], dtype=np.int64),
            )
        )
    return splits


def _load_selections(selections_path: Path) -> pd.DataFrame:
    if not selections_path.exists():
        raise FileNotFoundError(
            f"Selections file not found at {selections_path}. Run scripts/02_run_selection.py first."
        )

    selections = pd.read_parquet(selections_path)
    required = {"split_id", "rank", "protein_idx", "significant"}
    missing = required.difference(selections.columns)
    if missing:
        raise ValueError(f"Selections parquet missing required columns: {sorted(missing)}")

    selections = selections.loc[:, ["split_id", "rank", "protein_idx", "significant"]].copy()
    selections["split_id"] = selections["split_id"].astype(int)
    selections["rank"] = selections["rank"].astype(int)
    selections["protein_idx"] = selections["protein_idx"].astype(int)
    selections["significant"] = selections["significant"].astype(bool)

    return selections


def _predict_score(model: object, X_test: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        return np.asarray(proba)[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X_test), dtype=float)
    raise ValueError(f"Model {type(model).__name__} has no score method.")


def main() -> None:
    start_time = time.time()
    args = parse_args()

    logger = setup_logging(
        save_results=args.save_results,
        log_subdir=f"classifier/{args.method}",
        script_name="03_run_classifiers",
    )

    selections_path = args.selections_path
    if selections_path is None:
        selections_path = RESULTS_DIR / "selection" / args.method / "selections.parquet"

    logger.info("Starting 03_run_classifiers.py")
    logger.info(
        "Args: method=%s cohort_path=%s splits_path=%s selections_path=%s output_dir=%s seed=%d",
        args.method,
        args.cohort_path,
        args.splits_path,
        selections_path,
        args.output_dir,
        args.seed,
    )

    X, y = _load_cohort(args.cohort_path)
    splits = _load_splits(args.splits_path)
    selections = _load_selections(selections_path)

    max_split_in_selections = int(selections["split_id"].max()) + 1
    compatible_splits = min(len(splits), max_split_in_selections)
    if args.n_splits is not None:
        if args.n_splits <= 0:
            raise ValueError(f"--n-splits must be > 0, got {args.n_splits}.")
        n_splits = min(int(args.n_splits), compatible_splits)
    else:
        n_splits = compatible_splits

    if n_splits <= 0:
        raise ValueError("No compatible splits found between split cache and selections file.")

    selections = selections[selections["split_id"] < n_splits].copy()

    grouped: dict[int, pd.DataFrame] = {
        int(split_id): df.sort_values("rank", kind="mergesort")
        for split_id, df in selections.groupby("split_id", sort=True)
    }

    rows: list[dict[str, object]] = []
    failed_fits = 0

    for split_id in range(n_splits):
        if split_id not in grouped:
            raise ValueError(f"Selections are missing split_id={split_id}.")

        train_idx, test_idx = splits[split_id]
        X_train_all = X[train_idx]
        y_train = y[train_idx]
        X_test_all = X[test_idx]
        y_test = y[test_idx]

        split_df = grouped[split_id]
        ranked_idx = split_df["protein_idx"].to_numpy(dtype=np.int64)
        significant_mask = split_df["significant"].to_numpy(dtype=bool)

        feature_sets: list[tuple[object, np.ndarray]] = []
        for k in TOPK_VALUES:
            top_idx = ranked_idx[: min(int(k), ranked_idx.size)]
            feature_sets.append((int(k), top_idx))

        native_idx = ranked_idx[significant_mask]
        feature_sets.append(("native", native_idx))

        split_seed = int(args.seed + split_id)

        for k_label, feat_idx in feature_sets:
            n_features_actual = int(feat_idx.size)

            if n_features_actual == 0:
                for clf_name in ("LogisticRegression", "RandomForest", "XGBoost"):
                    rows.append(
                        {
                            "split_id": int(split_id),
                            "classifier": clf_name,
                            "k": str(k_label),
                            "auc": float("nan"),
                            "aupr": float("nan"),
                            "n_features_actual": 0,
                        }
                    )
                continue

            X_train = X_train_all[:, feat_idx]
            X_test = X_test_all[:, feat_idx]

            models = {
                "LogisticRegression": build_logreg(random_state=split_seed),
                "RandomForest": build_rf(random_state=split_seed),
                "XGBoost": build_xgb(y_train=y_train, random_state=split_seed),
            }

            for clf_name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    y_score = _predict_score(model, X_test)
                    auc = compute_auc(y_test, y_score)
                    aupr = compute_aupr(y_test, y_score)
                except Exception as exc:  # pragma: no cover
                    failed_fits += 1
                    logger.warning(
                        "Model failed (split=%d, k=%s, classifier=%s): %s",
                        split_id,
                        k_label,
                        clf_name,
                        exc,
                    )
                    auc = float("nan")
                    aupr = float("nan")

                rows.append(
                    {
                        "split_id": int(split_id),
                        "classifier": clf_name,
                        "k": str(k_label),
                        "auc": float(auc),
                        "aupr": float(aupr),
                        "n_features_actual": n_features_actual,
                    }
                )

    scores = pd.DataFrame(rows)
    scores = scores.loc[:, ["split_id", "classifier", "k", "auc", "aupr", "n_features_actual"]]

    out_dir = args.output_dir / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "scores.parquet"
    meta_path = out_dir / "meta.json"

    scores.to_parquet(scores_path, index=False)

    runtime_seconds = float(time.time() - start_time)
    meta = {
        "method": args.method,
        "n_splits": int(n_splits),
        "topk_values": [int(k) for k in TOPK_VALUES],
        "includes_native": True,
        "cohort_path": str(args.cohort_path),
        "splits_path": str(args.splits_path),
        "selections_path": str(selections_path),
        "scores_path": str(scores_path),
        "seed": int(args.seed),
        "split_seed_rule": "seed + split_id",
        "failed_fits": int(failed_fits),
        "n_rows": int(len(scores)),
        "runtime_seconds": runtime_seconds,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved scores: %s", scores_path)
    logger.info("Saved metadata: %s", meta_path)
    logger.info("Rows written: %d", len(scores))
    logger.info("Failed fits: %d", failed_fits)
    logger.info("Finished in %.2f s", runtime_seconds)


if __name__ == "__main__":
    main()
