"""
Post-filter the scored repurposing candidates to remove antibiotic leakage.

No model retraining needed: the candidate CSVs already carry model scores
(prob_mlp / prob_rf / prob_ensemble). This script just re-applies the robust
antibiotic exclusion (connectivity-propagated ATC + expanded name stems) and
writes a cleaned file, plus a report of exactly what was removed and why.

Run:
    python scripts/clean_candidates.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from antibiotic_filter import (  # noqa: E402
    build_antibiotic_index,
    connectivity_key,
    is_antibiotic,
    parse_atc,
    ANTIBIOTIC_ATC,
    _name_is_antibiotic,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

APPROVED = DATA / "approved_drugs.csv"
CANDIDATES_IN = DATA / "repurposing_candidates.csv"   # ADMET-rich scored screen (figures read this)
CANDIDATES_OUT = DATA / "repurposing_candidates_clean.csv"
REPORT_OUT = DATA / "antibiotic_leakage_report.csv"


def main() -> int:
    if not APPROVED.exists():
        print(f"missing {APPROVED}")
        return 1
    if not CANDIDATES_IN.exists():
        print(f"missing {CANDIDATES_IN}")
        return 1

    approved = pd.read_csv(APPROVED)
    cand = pd.read_csv(CANDIDATES_IN)
    print(f"Loaded approved drugs : {len(approved):,}")
    print(f"Loaded candidates     : {len(cand):,}")

    index = build_antibiotic_index(approved)
    print(f"Antibiotic structures (by connectivity) : {len(index['connectivity']):,}")
    print(f"indication_class column usable          : {index['has_indication']}")
    if not index["has_indication"]:
        print("  WARNING: indication_class absent from approved_drugs.csv — "
              "re-fetch with that field to re-enable the indication layer.")

    name_col = "drug_name" if "drug_name" in cand.columns else cand.columns[1]
    smi_col = "smiles" if "smiles" in cand.columns else "canonical_smiles"

    reasons = []
    for _, row in cand.iterrows():
        nm, smi = row[name_col], row.get(smi_col, "")
        why = []
        if _name_is_antibiotic(nm):
            why.append("name-stem")
        ck = connectivity_key(smi)
        if ck and ck in index["connectivity"]:
            why.append("connectivity->antibiotic-parent")
        reasons.append("|".join(why))

    cand = cand.copy()
    cand["_abx_reason"] = reasons
    leaked = cand[cand["_abx_reason"] != ""].copy()
    clean = cand[cand["_abx_reason"] == ""].drop(columns=["_abx_reason"]).reset_index(drop=True)

    print(f"\nRemoved antibiotic leakage : {len(leaked):,}")
    cols = [name_col, "_abx_reason"]
    if "prob_ensemble" in leaked.columns:
        cols.insert(1, "prob_ensemble")
    if len(leaked):
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(leaked.sort_values("prob_ensemble", ascending=False)[cols].to_string(index=False)
                  if "prob_ensemble" in leaked.columns
                  else leaked[cols].to_string(index=False))

    clean.to_csv(CANDIDATES_OUT, index=False)
    leaked[cols].to_csv(REPORT_OUT, index=False)
    print(f"\nClean candidates : {len(clean):,} -> {CANDIDATES_OUT.relative_to(ROOT)}")
    print(f"Leakage report   : {REPORT_OUT.relative_to(ROOT)}")

    # New top-10 drug-like ranking after cleaning
    if "drug_like" in clean.columns and "prob_ensemble" in clean.columns:
        dl = clean[clean["drug_like"].isin([1, True])].sort_values(
            "prob_ensemble", ascending=False).head(10)
        print("\nNew top-10 drug-like candidates (antibiotic-free):")
        print(dl[[name_col, "prob_ensemble"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
