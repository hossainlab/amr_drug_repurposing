"""Build per-target QSAR datasets from the ESKAPE bioactivity table.

Team review asked for a target-resolved analysis: the study reported only a pooled
whole-organism antibacterial score, but ChEMBL already carries the molecular target
of each assay (target_pref_name / target_chembl_id). This script isolates the rows
that hit a *protein* target (not a whole organism), pools species orthologs under
one protein name, majority-votes each molecule's activity, and keeps targets with
enough balanced data to train a QSAR model.

Output:
  data/targets/<slug>.csv      per-target: molecule_chembl_id, canonical_smiles, active
  data/targets/manifest.csv    target, slug, n_mols, n_active, active_rate

Run:
    python scripts/build_target_datasets.py
"""
from __future__ import annotations

import re

import pandas as pd

from amr_repurposing.config import DATA_DIR as DATA, PROJECT_ROOT as ROOT, TARGETS_DIR as OUT_DIR

ESKAPE = DATA / "eskape_clean.csv"
MANIFEST = OUT_DIR / "manifest.csv"

# The 8 ESKAPE(+Mtb) organism names appear in target_pref_name for whole-cell
# assays — those are phenotypic, not a molecular target, so they are excluded.
ESKAPE_ORGANISMS = {
    "Staphylococcus aureus", "Klebsiella pneumoniae", "Acinetobacter baumannii",
    "Pseudomonas aeruginosa", "Enterococcus faecium", "Enterobacter cloacae",
    "Escherichia coli", "Mycobacterium tuberculosis",
}

MIN_MOLS = 40          # minimum unique molecules to attempt a model
MIN_RATE, MAX_RATE = 0.10, 0.90   # keep classes usably balanced


def slugify(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).lower()).strip("_")
    return re.sub(r"_+", "_", s)


def main() -> int:
    if not ESKAPE.exists():
        print(f"missing {ESKAPE} — run the preprocessing notebook first")
        return 1

    df = pd.read_csv(ESKAPE, low_memory=False)
    prot = df[~df["target_pref_name"].isin(ESKAPE_ORGANISMS)].copy()
    prot = prot.dropna(subset=["canonical_smiles", "target_pref_name"])
    print(f"Protein-target activity rows : {len(prot):,} "
          f"({prot['target_pref_name'].nunique()} proteins)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for target, g in prot.groupby("target_pref_name"):
        # one label per molecule = majority vote across its assays on this target
        per_mol = (g.groupby("molecule_chembl_id")
                     .agg(active=("active", lambda s: int(s.mean() >= 0.5)),
                          canonical_smiles=("canonical_smiles", "first"))
                     .reset_index())
        n_mols = len(per_mol)
        rate = per_mol["active"].mean()
        if n_mols < MIN_MOLS or not (MIN_RATE <= rate <= MAX_RATE):
            continue
        slug = slugify(target)
        per_mol[["molecule_chembl_id", "canonical_smiles", "active"]].to_csv(
            OUT_DIR / f"{slug}.csv", index=False)
        rows.append(dict(target=target, slug=slug, n_mols=n_mols,
                         n_active=int(per_mol["active"].sum()), active_rate=round(rate, 3)))

    manifest = pd.DataFrame(rows).sort_values("n_mols", ascending=False).reset_index(drop=True)
    manifest.to_csv(MANIFEST, index=False)
    print(f"\nModelable targets kept : {len(manifest)}  (>= {MIN_MOLS} mols, "
          f"active-rate in [{MIN_RATE}, {MAX_RATE}])")
    print(manifest.to_string(index=False))
    print(f"\nWrote per-target datasets to {OUT_DIR.relative_to(ROOT)}/  and manifest.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
