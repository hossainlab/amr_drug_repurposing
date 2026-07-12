"""
Post-filter the scored repurposing candidates to remove antibiotic leakage.

No model retraining needed: the candidate CSVs already carry model scores
(prob_mlp / prob_rf / prob_ensemble). This script just re-applies the robust
antibiotic exclusion (connectivity-propagated ATC + expanded name stems) and
writes a cleaned file, plus a report of exactly what was removed and why.

Run:
    amr-clean-candidates    # or: python -m amr_repurposing.screen.candidates
"""
from __future__ import annotations

import pandas as pd

from amr_repurposing.config import DATA_DIR as DATA, PROJECT_ROOT as ROOT
from amr_repurposing.filtering.antibiotic_filter import (
    build_antibiotic_index,
    connectivity_key,
    parse_atc,
    is_antifungal_or_antiviral,
    _name_is_antibiotic,
    _is_aware_antibacterial,
)

APPROVED = DATA / "approved_drugs.csv"
CANDIDATES_RAW = DATA / "repurposing_candidates.csv"       # notebook output (gitignored)
CANDIDATES_OUT = DATA / "repurposing_candidates_clean.csv"
REPORT_OUT = DATA / "antibiotic_leakage_report.csv"
# Prefer the raw scored screen; if the notebook output is absent, re-filter the
# already-clean file. The antibiotic filter is a superset (AWaRe only ADDS
# removals), so re-applying it to a cleaned file is idempotent for old rules and
# still catches the newly-flagged second-line antibacterials.
CANDIDATES_IN = CANDIDATES_RAW if CANDIDATES_RAW.exists() else CANDIDATES_OUT


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
    print(f"Loaded candidates     : {len(cand):,}  (from {CANDIDATES_IN.name})")
    n_before = len(cand)

    index = build_antibiotic_index(approved)
    print(f"Antibiotic structures (by connectivity) : {len(index['connectivity']):,}")
    print(f"indication_class column usable          : {index['has_indication']}")
    if not index["has_indication"]:
        print("  WARNING: indication_class absent from approved_drugs.csv — "
              "re-fetch with that field to re-enable the indication layer.")

    name_col = "drug_name" if "drug_name" in cand.columns else cand.columns[1]
    smi_col = "smiles" if "smiles" in cand.columns else "canonical_smiles"

    atc_col = "atc_codes" if "atc_codes" in cand.columns else None
    reasons = []
    for _, row in cand.iterrows():
        nm, smi = row[name_col], row.get(smi_col, "")
        atc = parse_atc(row[atc_col]) if atc_col else []
        # KEEP-override: antifungals/antivirals are never antibiotics
        if is_antifungal_or_antiviral(atc, nm):
            reasons.append("")
            continue
        why = []
        if _name_is_antibiotic(nm):
            why.append("name-stem")
        if _is_aware_antibacterial(nm):
            why.append("aware")
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
    print(f"\nCandidates in  : {n_before:,}")
    print(f"Clean candidates : {len(clean):,} -> {CANDIDATES_OUT.relative_to(ROOT)}  "
          f"(delta {len(clean) - n_before:+,})")
    print(f"Leakage report   : {REPORT_OUT.relative_to(ROOT)}")
    # NB: the authoritative AWaRe removal audit (data/reference/aware_removals_audit.csv)
    # is written by build_screen_library.py from the full removed set.

    # New top-10 drug-like ranking after cleaning
    if "drug_like" in clean.columns and "prob_ensemble" in clean.columns:
        dl = clean[clean["drug_like"].isin([1, True])].sort_values(
            "prob_ensemble", ascending=False).head(10)
        print("\nNew top-10 drug-like candidates (antibiotic-free):")
        print(dl[[name_col, "prob_ensemble"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
