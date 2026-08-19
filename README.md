# Dynamics-Aware Activity Prediction for α1-Adrenoceptor Subtypes

This repository contains the public code, frozen analysis configuration, and aggregate cross-validation results for a Dynamic-GLEP workflow applied to α1A-, α1B-, and α1D-adrenoceptor ligand activity prediction.

> **Data-release status:** compound-level data are intentionally not included in this version. The repository does not distribute molecular structures, SMILES, activity records, docking poses, EquiScore feature matrices, candidate-level predictions, or trained model files.

## Overview

Three independent binary classifiers were trained for α1A-, α1B-, and α1D-AR. Compounds with pIC50 > 7.0 were labeled active, while compounds with pIC50 ≤ 7.0 were labeled inactive.

| Target | Total | Active | Inactive | Active fraction |
|---|---:|---:|---:|---:|
| α1A-AR | 287 | 126 | 161 | 43.9% |
| α1B-AR | 134 | 21 | 113 | 15.7% |
| α1D-AR | 231 | 89 | 142 | 38.5% |

The models predict subtype-specific ligand activity. They were not trained as a direct three-class subtype-selectivity classifier, and their outputs should not be interpreted as experimentally validated selectivity probabilities.

## Compared representations

The following representations were evaluated using the same labels, classifier settings, and cross-validation partitions:

1. **Docking-only:** the arithmetic mean of available best-pose Glide scores across RRCS-selected active- and inactive-state conformations.
2. **Static-structure:** a 128-dimensional EquiScore representation generated from the fixed `active_000` receptor snapshot.
3. **Dynamic-GLEP:** a 256-dimensional representation formed by concatenating the dimension-wise mean and population standard deviation of 128-dimensional EquiScore embeddings across the RRCS-selected receptor ensemble.

## Classifier and validation

Each subtype used an unweighted `GradientBoostingClassifier` with:

```text
n_estimators     = 150
learning_rate    = 0.04
max_depth        = 3
min_samples_leaf = 1
subsample        = 1.0
```

Performance was evaluated by stratified shuffled five-fold cross-validation, repeated using seeds 4096, 4097, and 4098. Metrics were calculated from complete out-of-fold predictions for each repeat and are reported as mean ± standard deviation across repeats. Balanced accuracy and F1 used a probability threshold of 0.5.

This random cross-validation analysis estimates performance within the chemical space represented by the curated datasets. It is not a scaffold-disjoint estimate for structurally unrelated compounds.

## Aggregate results

| Model | α1A AUC | α1A AP | α1B AUC | α1B AP | α1D AUC | α1D AP | Mean AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Docking-only | 0.652 ± 0.027 | 0.645 ± 0.012 | 0.506 ± 0.051 | 0.200 ± 0.050 | 0.618 ± 0.016 | 0.466 ± 0.023 | 0.592 |
| Static-structure | 0.749 ± 0.010 | 0.662 ± 0.016 | 0.518 ± 0.039 | 0.196 ± 0.043 | 0.645 ± 0.045 | 0.547 ± 0.018 | 0.637 |
| Dynamic-GLEP | 0.813 ± 0.009 | 0.767 ± 0.025 | 0.580 ± 0.053 | 0.280 ± 0.027 | 0.613 ± 0.018 | 0.540 ± 0.034 | 0.669 |

Dynamic-GLEP achieved the highest mean AUC, with the clearest improvement for α1A-AR. α1B-AR performance remained limited, while α1D-AR performance was comparable to the structure-based baselines.

## Repository structure

```text
.
├── code/
│   ├── train_evaluate_three_routes.py
│   ├── three_route_helpers.py
│   └── predict_external51.py
├── configs/
│   ├── environment.json
│   └── final_config.json
├── data/
│   └── README.md
├── results/
│   ├── model_performance_table.csv
│   ├── pure_route_repeat_metrics.csv
│   └── pure_route_summary.csv
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

Python 3.10 is recommended. The archived analysis environment used Python 3.10.13, NumPy 2.0.1, pandas 2.2.3, scikit-learn 1.7.0, RDKit 2025.03.4, and joblib 1.5.1.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell activation:

```powershell
.venv\Scripts\Activate.ps1
```

## Reproducibility and unavailable inputs

The public code documents the final preprocessing, representation comparison, cross-validation, model-fitting, and candidate-inference logic. Full reruns require upstream files that are not distributed here, including:

- curated compound-level activity records;
- molecular structures and identifiers;
- RRCS receptor-conformation selections;
- Glide docking outputs;
- EquiScore embeddings and processed feature matrices;
- trained `joblib` model files;
- compound-level out-of-fold and candidate predictions.

The inference script is included for methodological transparency but cannot reproduce candidate predictions without the withheld feature matrices, ligand-similarity reference set, and frozen models.

## Data availability

Some records used during dataset construction originated from Cortellis and may be subject to third-party licensing restrictions. Compound-level data and derived files are therefore withheld from this initial public code release. Access may be considered upon reasonable request, subject to institutional approval and applicable data-use agreements.

## Citation

If you use this code, please cite the associated manuscript. Full citation details will be added after publication.

```text
[Authors]. [Manuscript title]. [Journal, year, DOI].
```

## License

No open-source license has yet been assigned. A license will be added after confirmation with the project owners and participating institutions.

