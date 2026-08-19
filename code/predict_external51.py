#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


TASKS = ("A", "B", "D")
SEEDS = (4096, 4097, 4098)


def canonicalize(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--external-dir", type=Path, required=True)
    parser.add_argument("--alpha1a-active-reference", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)

    feature_columns = [f"f{i:03d}" for i in range(256)]
    metadata = pd.read_csv(args.external_dir / "ext_A.csv")[["Name", "Number", "Index", "canonical_smiles"]]
    if metadata["Name"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate external compound names")
    result = metadata.copy()
    seed_rows = []

    for task in TASKS:
        external = pd.read_csv(args.external_dir / f"ext_{task}.csv")
        external = external.assign(Name=external["Name"].astype(str)).set_index("Name").loc[
            metadata["Name"].astype(str)
        ].reset_index()
        x = external[feature_columns].to_numpy(np.float32)
        scoreable = external["scoreable"].astype(bool).to_numpy()
        predictions = []
        for seed in SEEDS:
            bundle = joblib.load(args.model_dir / "dynamic_glep" / f"model_{task}_seed{seed}.joblib")
            if bundle["feature_dimensions"] != 256:
                raise RuntimeError(f"{task}/{seed}: unexpected feature dimensions")
            median = np.asarray(bundle["impute_median"], dtype=np.float32)
            xi = np.where(np.isfinite(x), x, median)
            probability = bundle["model"].predict_proba(xi)[:, 1]
            probability = np.where(scoreable, probability, np.nan)
            predictions.append(probability)
            seed_rows.extend({
                "Name": str(metadata.iloc[i]["Name"]), "task": task, "seed": seed,
                "probability": float(probability[i]) if np.isfinite(probability[i]) else np.nan,
            } for i in range(len(metadata)))
        stacked = np.vstack(predictions)
        result[f"prob_{task}"] = np.nanmean(stacked, axis=0)
        result[f"seed_sd_{task}"] = np.nanstd(stacked, axis=0, ddof=1)

    complete = np.isfinite(result[["prob_A", "prob_B", "prob_D"]]).all(axis=1)
    reference = pd.read_csv(args.alpha1a_active_reference)
    active_unique = sorted({canonicalize(s) for s in reference["canonical_smiles"].astype(str)})
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    active_fps = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in active_unique]
    similarities, nearest, exact_excluded = [], [], []
    for query in result["canonical_smiles"].astype(str):
        query_canonical = canonicalize(query)
        query_fp = generator.GetFingerprint(Chem.MolFromSmiles(query_canonical))
        eligible = [i for i, ref in enumerate(active_unique) if ref != query_canonical]
        excluded = len(eligible) != len(active_unique)
        sims = DataStructs.BulkTanimotoSimilarity(query_fp, [active_fps[i] for i in eligible])
        best_local = int(np.argmax(sims))
        similarities.append(float(sims[best_local]))
        nearest.append(active_unique[eligible[best_local]])
        exact_excluded.append(excluded)
    result["alpha1a_active_max_ecfp4_tanimoto"] = similarities
    result["nearest_nonidentical_alpha1a_active_smiles"] = nearest
    result["exact_active_reference_excluded"] = exact_excluded
    result["A_evidence"] = 0.9 * result["prob_A"] + 0.1 * result["alpha1a_active_max_ecfp4_tanimoto"]
    result["score_1a"] = result["A_evidence"] - result[["prob_B", "prob_D"]].max(axis=1)
    result["pred_1a_preferred"] = complete & (result["score_1a"] > 0)
    result["A_rank"] = result["score_1a"].rank(method="min", ascending=False).astype(int)

    result = result.sort_values(["A_rank", "Name"]).reset_index(drop=True)
    result.to_csv(args.out_dir / "external51_dynamic_gbc_predictions.csv", index=False)
    result.drop(columns=["canonical_smiles", "nearest_nonidentical_alpha1a_active_smiles"]).to_csv(
        args.out_dir / "external51_dynamic_gbc_predictions_public.csv", index=False
    )
    pd.DataFrame(seed_rows).to_csv(args.out_dir / "seed_predictions.csv", index=False)
    pd.DataFrame({
        "criterion": ["similarity_informed_alpha1A_preference"],
        "count": [int(result["pred_1a_preferred"].sum())],
        "total": [len(result)],
    }).to_csv(args.out_dir / "summary.csv", index=False)
    (args.out_dir / "definition.json").write_text(json.dumps({
        "model": "Frozen current-data Dynamic-GLEP unbalanced GBC models; three subtype models and three seeds per subtype.",
        "features": "Subtype-specific 256D Dynamic-GLEP mean+population-SD external representations.",
        "external_only_prior": "A_evidence = 0.9*prob_A + 0.1*maximum ECFP4 similarity to a non-identical alpha1A-active training ligand.",
        "final_score": "A_evidence - max(prob_B, prob_D).",
        "decision": "score_1a > 0 indicates exploratory relative alpha1A preference; no B-versus-D assignment.",
        "prior_not_used_in_training": True,
    }, indent=2), encoding="utf-8")
    print(pd.read_csv(args.out_dir / "summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
