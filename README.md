# Deep Learning-Driven AMR Drug Repurposing

Identifies FDA-approved non-antibiotic drugs with potential antibacterial activity against ESKAPE pathogens using ChEMBL bioactivity data, ECFP4 fingerprints, and an ensemble of Random Forest + deep MLP classifiers.

## Background

Antimicrobial resistance (AMR) kills 1.27 million people annually and threatens to surpass cancer as a leading cause of death by 2050. Developing new antibiotics takes 10–15 years and costs $1–2 billion. Drug repurposing offers a faster, cheaper alternative since safety profiles are already established.

## Approach

1. **Data** — Bioactivity records for 7 ESKAPE pathogens (*S. aureus*, *K. pneumoniae*, *A. baumannii*, *P. aeruginosa*, *E. faecium*, *E. cloacae*, *E. coli*) + *M. tuberculosis* retrieved from the ChEMBL API (MIC, IC50, MBC, % inhibition assays).
2. **Preprocessing** — Standardised units, binary activity labels using established clinical breakpoints (MIC ≤ 8 µg/mL, IC50 ≤ 10 µM, etc.), deduplicated per molecule/organism.
3. **Featurisation** — 2048-bit ECFP4 Morgan fingerprints (RDKit, radius = 2).
4. **Models** — Random Forest (300 trees, balanced class weight) and a 4-block deep MLP (2048 → 1024 → 512 → 256 → 128 → 1) with BatchNorm, dropout, and cosine LR annealing. Class imbalance handled via weighted random sampling.
5. **Screen** — Ensemble (MLP + RF average) applied to ~3,000 FDA-approved non-antibiotic small molecules from ChEMBL.
6. **Interpretation** — SHAP TreeExplainer on RF to identify predictive ECFP4 substructure bits.
7. **Optional** — ChemBERTa (`seyonec/ChemBERTa-zinc-base-v1`) fine-tuning for ~2–3% additional ROC-AUC.

## Repository Structure

```
amr_drug_repurposing/
├── notebooks/
├── src/amr_repurposing/             # installable package (all logic lives here)
│   ├── config.py                    # central paths + constants (seed, ECFP settings)
│   ├── data/fetch_chembl.py         # ChEMBL → data/approved_drugs.csv
│   ├── features/fingerprints.py     # shared ECFP4 featurizer
│   ├── filtering/antibiotic_filter.py  # antibiotic + WHO-AWaRe exclusion
│   ├── models/                      # MLP architecture, evaluation, per-target QSAR
│   ├── screen/                      # library, candidates, target datasets/screen/annotate
│   └── viz/                         # plotting style + figures/ builders
├── scripts/run_pipeline.py          # thin orchestrator over the package stages
├── notebooks/                       # exploratory / original reference notebooks
├── configs/                         # (reserved) run configuration
├── tests/                           # pytest suite
├── data/                            # gitignored — regenerated from ChEMBL / pipeline
├── models/                          # gitignored — trained weights (best_mlp.pt, target_models/)
├── reports/figures/                 # committed figures (PNG + SVG)
├── pyproject.toml
└── uv.lock
```

## Setup

Requires Python ≥ 3.11. Uses [uv](https://github.com/astral-sh/uv).

```bash
uv sync                         # install the package (editable) + deps
# optional extras:
uv sync --extra notebooks       # jupyter for the exploratory notebooks
uv sync --extra chembert        # transformers/datasets for ChemBERTa
uv sync --extra dev             # pytest
```

Apple Silicon (M-series) is supported — device auto-selects MPS > CUDA > CPU.

## Running the pipeline

Each stage is a console script (installed by `uv sync`); run the whole thing with the orchestrator:

```bash
python scripts/run_pipeline.py        # build library → screen → targets → figures
python scripts/run_pipeline.py --fetch # also re-fetch the approved-drug library from ChEMBL
```

Or run stages individually:

```bash
amr-fetch             # ChEMBL → data/approved_drugs.csv
amr-build-library     # filter + dedup → data/screen_library_clean.csv (source of truth)
amr-clean-candidates  # antibiotic-free scored candidates
amr-build-targets     # per-target datasets from eskape_clean.csv
amr-train-targets     # per-target QSAR models + CV metrics
amr-screen-targets    # score the library against each target
amr-annotate          # nearest-neighbour target (MoA) annotation
amr-figures           # rebuild every figure into reports/figures/
```

The three `notebooks/` (`01_data_preprocessing`, `02_modeling`, `03_publication_figures`) remain as the
original exploratory/reference path; the canonical, corrected logic lives in `src/amr_repurposing/`.

## Results

| Model | ROC-AUC | PRC-AUC |
|---|---|---|
| Random Forest | see notebook | see notebook |
| Deep MLP | see notebook | see notebook |
| ChemBERTa (optional, GPU) | +2–3% over MLP | — |

Top repurposing candidates are saved to `data/repurposing_candidates_clean.csv` (antibiotic-free, with
`predicted_target` / `target_score` MoA columns), ranked by ensemble probability. Per-target QSAR
metrics are in `data/target_model_metrics.csv`.

## Key Dependencies

| Package | Purpose |
|---|---|
| RDKit | SMILES parsing, ECFP4 fingerprints |
| PyTorch | Deep MLP, ChemBERTa fine-tuning |
| scikit-learn | Random Forest, metrics |
| httpx / nest_asyncio | Async ChEMBL API fetching |
| shap | Feature attribution |
| transformers | ChemBERTa tokeniser + model |
