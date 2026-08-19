#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from three_route_helpers import (
    SEEDS, TASKS, docking_mean, impute, load_npz, prepare_raw_static, static_active_matrix,
)


def new_model(seed: int) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.04,
        max_depth=3,
        min_samples_leaf=1,
        subsample=1.0,
        random_state=seed,
    )


def metric_values(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y, probability)),
        "ap": float(average_precision_score(y, probability)),
        "ap_baseline": float(y.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
    }


def canonicalize(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--supplementary-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)
    model_dir = args.out_dir / "final_models"
    model_dir.mkdir()

    selected = pd.read_csv(args.root / "RRCS/rrcs_1abd_selection.csv")
    repeat_rows: list[dict] = []
    oof_rows: list[dict] = []
    fold_rows: list[dict] = []
    coverage_rows: list[dict] = []
    active_a_reference: pd.DataFrame | None = None

    for task, subtype in TASKS.items():
        original = load_npz(args.train_features / f"state_features_{task}.npz")
        supplementary = load_npz(args.supplementary_features / f"state_features_{task}.npz")
        overlap = set(original["canonical_smiles"].astype(str)) & set(supplementary["canonical_smiles"].astype(str))
        if overlap:
            raise RuntimeError(f"{task}: exact overlap before concatenation")

        static_groups = prepare_raw_static(args.root / f"raw_data/equi_raw_data/{subtype}_cortellis_train_merged_clean.csv")
        static_original, static_original_missing = static_active_matrix(
            static_groups, original["canonical_smiles"], original["pIC50"]
        )
        static_supplementary, static_supplementary_missing = static_active_matrix(
            static_groups, supplementary["canonical_smiles"], supplementary["pIC50"]
        )
        docking_original, docking_original_scored = docking_mean(
            args.root, task, original["canonical_smiles"], selected
        )
        docking_supplementary, docking_supplementary_scored = docking_mean(
            args.root, task, supplementary["canonical_smiles"], selected
        )

        y = np.concatenate([original["label"], supplementary["label"]]).astype(int)
        smiles = np.concatenate([original["canonical_smiles"], supplementary["canonical_smiles"]]).astype(str)
        matrices = {
            "docking_only": np.vstack([docking_original, docking_supplementary]).astype(np.float32),
            "static_structure": np.vstack([static_original, static_supplementary]).astype(np.float32),
            "dynamic_glep": np.vstack([original["combined"], supplementary["combined"]]).astype(np.float32),
        }
        if task == "A":
            active_a_reference = pd.DataFrame({
                "canonical_smiles": smiles[y == 1],
                "label": y[y == 1],
            }).drop_duplicates("canonical_smiles")

        coverage_rows.append({
            "task": task,
            "total_n": len(y),
            "active": int(y.sum()),
            "inactive": int((1-y).sum()),
            "docking_dimensions": matrices["docking_only"].shape[1],
            "static_dimensions": matrices["static_structure"].shape[1],
            "dynamic_dimensions": matrices["dynamic_glep"].shape[1],
            "docking_scored_n": docking_original_scored + docking_supplementary_scored,
            "static_missing_zero_filled": static_original_missing + static_supplementary_missing,
        })

        for seed in SEEDS:
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            splits = list(splitter.split(np.zeros(len(y)), y))
            fold_assignment = np.full(len(y), -1, dtype=int)
            for fold, (_, valid) in enumerate(splits):
                fold_assignment[valid] = fold
                fold_rows.extend({
                    "task": task, "seed": seed, "row_index": int(i), "fold": fold,
                    "canonical_smiles": str(smiles[i]), "label": int(y[i]),
                } for i in valid)
            for route, x in matrices.items():
                probability = np.full(len(y), np.nan)
                for fit, valid in splits:
                    xfit, xvalid = impute(x[fit], x[valid])
                    fitted = new_model(seed)
                    fitted.fit(xfit, y[fit])
                    probability[valid] = fitted.predict_proba(xvalid)[:, 1]
                repeat_rows.append({
                    "configuration": "pure_route",
                    "task": task, "route": route, "seed": seed,
                    "n": len(y), "n_active": int(y.sum()),
                    **metric_values(y, probability),
                })
                oof_rows.extend({
                    "task": task, "route": route, "seed": seed,
                    "row_index": i, "fold": int(fold_assignment[i]),
                    "canonical_smiles": str(smiles[i]),
                    "label": int(y[i]), "probability": float(probability[i]),
                } for i in range(len(y)))

                xfull, _ = impute(x, x[:1])
                median = np.nanmedian(x, axis=0)
                median = np.where(np.isfinite(median), median, 0.0)
                final = new_model(seed)
                final.fit(xfull, y)
                route_dir = model_dir / route
                route_dir.mkdir(exist_ok=True)
                joblib.dump({
                    "model": final,
                    "impute_median": median,
                    "task": task,
                    "route": route,
                    "seed": seed,
                    "feature_dimensions": int(x.shape[1]),
                    "activity_definition": "pIC50 > 7.0",
                    "n_train": len(y),
                    "n_active": int(y.sum()),
                }, route_dir / f"model_{task}_seed{seed}.joblib")

    repeat = pd.DataFrame(repeat_rows)
    repeat.to_csv(args.out_dir / "repeat_metrics.csv", index=False)
    summary = repeat.groupby(["configuration", "task", "route", "n", "n_active"])[
        ["auc", "ap", "ap_baseline", "balanced_accuracy", "f1"]
    ].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(args.out_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.out_dir / "fold_assignments.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(args.out_dir / "coverage.csv", index=False)
    if active_a_reference is None:
        raise RuntimeError("Missing alpha1A active reference")
    active_a_reference.to_csv(args.out_dir / "alpha1a_active_ecfp4_reference.csv", index=False)
    (args.out_dir / "definition.json").write_text(json.dumps({
        "data": "Final subtype-specific modeling datasets.",
        "sample_sizes": {"A": 287, "B": 134, "D": 231},
        "activity": "pIC50 > 7.0",
        "classifier": "Unbalanced GradientBoostingClassifier(n_estimators=150, learning_rate=0.04, max_depth=3, min_samples_leaf=1, subsample=1.0).",
        "seeds": list(SEEDS),
        "cv": "Three newly generated stratified shuffled random five-fold repetitions shared across routes.",
        "routes": {
            "docking_only": "One arithmetic-mean Glide docking-score feature over RRCS-selected active and inactive conformations; fold-training median imputation.",
            "static_structure": "128-dimensional EquiScore embedding from active_000 only.",
            "dynamic_glep": "256-dimensional mean plus population-SD pooling over RRCS-selected conformational EquiScore embeddings.",
        },
        "fair_comparison": "All three routes use identical labels, folds, and classifier settings.",
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(pd.DataFrame(coverage_rows).to_string(index=False))


if __name__ == "__main__":
    main()
