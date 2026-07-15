# Aggregation Granularity Reverses Confusion Diagnostics

Code and data for **"Aggregation Granularity Reverses Confusion Diagnostics:
Persuasion Technique Confusion Is Cross-Lingually Shared"**.

We apply an Optimal Transport Confusion Matrix (TCM) with IPF bi-normalisation
to nine configurations (EN/PL/RU × Sup-FT/Prompt-A/Iter-Ens) at two aggregation
granularities, holding predictions, gold, code and taxonomy fixed.

## Quick start

```bash
pip install -r requirements.txt
# put the data in place first — see data/README.md
jupyter notebook 01_paragraph_index.ipynb
```

## Notebooks — run in order

| Notebook | Produces | Paper |
|---|---|---|
| `01_paragraph_index` | Paragraph index; asserts 3,127 / 779 / 503 and gold counts | §III-C |
| `02_tcm_article` | Article-level TCM ×9 → Table IV, `Article` column | §III-C, §IV-B |
| `03_tcm_paragraph` | Paragraph-level TCM, support-floor sweep, K=9 | Tables I, II, V |
| `04_coupling_support` | ‖d‖₀, ‖e‖₀, cells → Table III, Fig. 2 | §IV-A (RQ1) |
| `05_rho_and_bootstrap` | Rank correlations, both subset treatments → Table IV, Fig. 3 | §IV-B, §IV-C (RQ2) |
| `06_modularity_permtest` | Newman Q + permutation null + K=8 | §IV-D |
| `07_ipf_pathology` | IPF without a support floor: 3/9 fail, max 0.997 | §III-B |

`02`–`07` are standalone (each rebuilds what it needs from source, ~1–2 min);
they share nothing but `src/common.py`, so there is no hidden notebook state.
`01` is a data check — if it fails, nothing downstream is valid.

## Key parameters (`src/common.py`)

| | |
|---|---|
| Paragraph segmentation | `splitlines()`, drop blank, drop `len > 1000` |
| Support floor | `s = 1` (gold **and** predicted mass ≥ 1 in **all nine** configs) → K = 9 |
| IPF | tol `1e-7`, eps `1e-8`, cap `1e5`; the §III-B pathology demo caps at 2,000 |
| Robust-pair threshold | bi-norm weight ≥ 0.15, ≥ 3/9 configurations |
| Bootstrap / permutation | B = 5,000, seed 42 |

## Reproducing the headline numbers

| Claim | Notebook | Value |
|---|---|---|
| Cells spanned, disjoint by unit | `04` | article 21.7–72.2 vs paragraph 3.1–16.6; 2.7–13.2×, 9/9 |
| ‖d‖₀ tracks over-prediction | `04` | ρ = +0.93 (article), +0.88 (paragraph), +0.49 pooled |
| Cross-lingual ρ flips | `05` | −0.004 → +0.356 (9/9 positive) → +0.198 (8/9) |
| Paradigm ordering reverses | `05` | −0.274 / −0.215 → +0.531 / +0.547 (all), +0.567 / +0.583 (intersect) |
| **Δρ is not identified** | `05` | +0.076 (p = 0.246) vs +0.365 (p = 0.005) |
| Q is a null result | `06` | Q = −0.059, null mean −0.065, 0/9 significant |
| IPF without a floor | `07` | 3/9 fail at 2,000; one converges at 1,815 with max 0.997 |

## Outputs

```
matrices/
├── article/              {lang}_{method}_raw.npy, _bi.npy      # full 23×23
├── paragraph/            {lang}_{method}_raw.npy, _bi_K9.npy
└── paragraph_intersect/  {lang}_{method}_bi_K9.npy

results/
├── table1_supsweep.csv           table2_common_techniques.csv
├── table3_coupling_support.csv   table4_delta_rho.csv
├── table5_robust_pairs.csv       rho_all.csv
├── modularity_permtest.csv       modularity_K8.csv
├── ipf_no_floor.csv              paragraph_units_used.csv
└── figures/ fig_support.{pdf,png}  fig_rho.{pdf,png}
```

Raw matrices are the full 23×23 at both units, so the support floor can be
audited independently — the K=9 sub-space is a slice of them.

## Data

See `data/README.md`. Article texts and gold spans must be obtained from the
original benchmarks (licence). **Note the upstream 2× duplication in the Polish
gold** — it is handled in `load_spans`, but it will bite anyone who reimplements.

## Prompts

`prompts/` contains the 23 expert-agent prompts used by Prompt-A (technique
definition + curated example sentences, fixed at design time; not
retrieval-augmented) and the Iter-Ens configuration (τ = 0.4, three rounds,
vote threshold 0.34, union aggregation).
