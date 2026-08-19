#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


TASKS = {"A": "1a", "B": "1b", "D": "1d"}
SEEDS = (4096, 4097, 4098)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in ("combined", "label", "pIC50", "canonical_smiles")}


def standardize(*values: object) -> str | None:
    for value in values:
        if pd.isna(value):
            continue
        mol = Chem.MolFromSmiles(str(value))
        if mol is None:
            continue
        mol = rdMolStandardize.FragmentParent(mol)
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)
        return Chem.MolToSmiles(mol, canonical=True)
    return None


def mol_key_from_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.FragmentParent(mol)
        return Chem.MolToInchiKey(mol).split("-")[0]
    except Exception:
        return None


def mol_key(mol: Chem.Mol) -> str | None:
    try:
        mol = rdMolStandardize.FragmentParent(mol)
        return Chem.MolToInchiKey(mol).split("-")[0]
    except Exception:
        return None


def parse_vec(value: object) -> np.ndarray | None:
    text = str(value).strip()
    if text in ("", "[]", "nan", "None"):
        return None
    try:
        arr = np.asarray(ast.literal_eval(text), dtype=np.float32)
    except Exception:
        arr = np.asarray([float(x) for x in re.split(r"[\s,]+", text.strip("[] ")) if x], dtype=np.float32)
    return arr if arr.shape == (128,) else None


def prepare_raw_static(path: Path) -> dict[str, pd.DataFrame]:
    raw = pd.read_csv(path)
    second = "SMILES_canonical" if "SMILES_canonical" in raw.columns else "SMILES"
    raw["_canonical"] = [standardize(a, b) for a, b in zip(raw["SMILES"], raw[second])]
    raw["_value"] = pd.to_numeric(raw["Standard Value"], errors="coerce")
    return {key: group for key, group in raw.dropna(subset=["_canonical"]).groupby("_canonical", sort=False)}


def static_active_matrix(groups: dict[str, pd.DataFrame], smiles: np.ndarray, pic50: np.ndarray) -> tuple[np.ndarray, int]:
    rows = []
    missing = 0
    for smi, p in zip(smiles.astype(str), pic50.astype(float)):
        group = groups.get(smi)
        if group is None:
            raise RuntimeError(f"No raw static row for {smi}")
        target_nm = 10 ** (9 - p)
        delta = np.abs(group["_value"].to_numpy(float) - target_nm)
        source = group.iloc[int(np.nanargmin(delta))]
        vector = parse_vec(source["active_000"])
        if vector is None:
            vector = np.zeros(128, dtype=np.float32)
            missing += 1
        rows.append(vector)
    return np.vstack(rows), missing


def read_best_docking(files: list[Path]) -> dict[str, float]:
    best: dict[str, float] = {}
    for path in files:
        if not path.exists():
            continue
        for mol in Chem.SDMolSupplier(str(path), removeHs=False):
            if mol is None or not mol.HasProp("r_i_docking_score"):
                continue
            key = mol_key(mol)
            if key is None:
                continue
            value = float(mol.GetProp("r_i_docking_score"))
            if key not in best or value < best[key]:
                best[key] = value
    return best


def docking_mean(root: Path, task: str, smiles: np.ndarray, selected: pd.DataFrame) -> tuple[np.ndarray, int]:
    keys = [mol_key_from_smiles(s) for s in smiles.astype(str)]
    matrices = []
    for state, short in (("active", "a"), ("inactive", "ina")):
        indices = selected[
            (selected.subtype == f"1{task.lower()}")
            & (selected.state == state)
            & (selected.selected == True)
        ].pdb_index.astype(int).tolist()
        matrix = np.full((len(keys), len(indices)), np.nan, dtype=np.float32)
        for j, index in enumerate(indices):
            files = [
                root / f"data_interm/docking/results/1{task}_{short}_holo_glide_{source}/{state}_{index}.sdf"
                for source in ("chembl", "cortellis")
            ]
            scores = read_best_docking(files)
            matrix[:, j] = [scores.get(key, np.nan) for key in keys]
        matrices.append(matrix)
    with np.errstate(all="ignore"):
        feature = np.nanmean(np.hstack(matrices), axis=1)[:, None]
    return feature, int(np.isfinite(feature[:, 0]).sum())



def impute(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(train, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    return np.where(np.isfinite(train), train, median), np.where(np.isfinite(test), test, median)


