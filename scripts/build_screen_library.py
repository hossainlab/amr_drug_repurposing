"""
Build the canonical non-antibiotic screening library — the single source of truth.

Pipeline:
  1. load data/approved_drugs.csv
  2. flag antibiotics with the robust filter (own ATC + parent-hierarchy ATC +
     connectivity + INN name stems)
  3. drop antibiotics
  4. deduplicate salt/solvate forms by InChIKey connectivity (keep the form whose
     name has no salt/counterion suffix when possible)
  5. write data/screen_library_clean.csv

Any downstream step (modeling notebook, scoring script) must read
screen_library_clean.csv instead of re-deriving the library, so a fresh run can
no longer leak antibiotics.

Run:
    python scripts/build_screen_library.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from antibiotic_filter import (  # noqa: E402
    build_antibiotic_index,
    connectivity_key,
    parse_atc,
    parent_chembl_id,
    is_antifungal_or_antiviral,
    _atc_is_antibiotic,
    _name_is_antibiotic,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
APPROVED = DATA / "approved_drugs.csv"
OUT = DATA / "screen_library_clean.csv"
REMOVED = DATA / "screen_library_removed_antibiotics.csv"

SALT_TOKENS = {
    "sodium", "disodium", "potassium", "calcium", "magnesium", "zinc",
    "hydrochloride", "hydrobromide", "chloride", "bromide", "sulfate",
    "mesylate", "maleate", "citrate", "tosylate", "phosphate", "besylate",
    "fumarate", "succinate", "tartrate", "acetate", "nitrate", "bisulfate",
    "lysine", "arginine", "meglumine", "diolamine", "anhydrous", "monohydrate",
    "dihydrate", "trihydrate", "hydrate", "free", "acid", "base",
}


def _salt_penalty(name: str) -> int:
    """Lower = cleaner parent name. Counts salt/counterion tokens in the name."""
    toks = str(name).lower().replace(",", " ").split()
    return sum(t in SALT_TOKENS for t in toks)


def main() -> int:
    if not APPROVED.exists():
        print(f"missing {APPROVED} — run scripts/fetch_approved_drugs.py first")
        return 1

    df = pd.read_csv(APPROVED)
    df = df.dropna(subset=["canonical_smiles"])
    df = df[df["canonical_smiles"].astype(str).str.len() > 5].copy()
    print(f"Approved drugs (with structures) : {len(df):,}")

    index = build_antibiotic_index(df)
    print(f"  ATC parent-hierarchy propagation used : {index['used_parent']}")
    print(f"  indication_class column usable        : {index['has_indication']}")
    print(f"  antibiotic structures (connectivity)  : {len(index['connectivity']):,}")

    # per-row flag with reason, mirroring build_antibiotic_index logic
    id_to_atc = {str(r["molecule_chembl_id"]): parse_atc(r.get("atc_classifications"))
                 for _, r in df.iterrows()} if "molecule_chembl_id" in df.columns else {}

    reasons = []
    for _, row in df.iterrows():
        atc = parse_atc(row.get("atc_classifications"))
        # KEEP-override: antifungals/antivirals are never antibiotics
        if is_antifungal_or_antiviral(atc, row.get("pref_name", "")):
            reasons.append("")
            continue
        why = []
        if _atc_is_antibiotic(atc):
            why.append("own-ATC")
        pid = parent_chembl_id(row.get("molecule_hierarchy"))
        if pid and pid in id_to_atc and _atc_is_antibiotic(id_to_atc[pid]):
            why.append("parent-ATC")
        if _name_is_antibiotic(row.get("pref_name", "")):
            why.append("name-stem")
        ck = connectivity_key(row["canonical_smiles"])
        if ck and ck in index["connectivity"] and not why:
            why.append("connectivity")
        reasons.append("|".join(why))

    df = df.copy()
    df["_abx_reason"] = reasons
    removed = df[df["_abx_reason"] != ""].copy()
    keep = df[df["_abx_reason"] == ""].copy()
    print(f"\nFlagged antibiotics : {len(removed):,}")
    print(f"Non-antibiotic      : {len(keep):,}")

    # deduplicate salt/solvate by connectivity, prefer cleanest (parent) name
    keep["_ck"] = keep["canonical_smiles"].apply(connectivity_key)
    keep["_penalty"] = keep["pref_name"].apply(_salt_penalty)
    keep = (keep.sort_values(["_penalty", "pref_name"])
                .drop_duplicates(subset="_ck", keep="first")
                .reset_index(drop=True))
    n_dups = (len(df[df["_abx_reason"] == ""]) - len(keep))
    print(f"After salt dedup    : {len(keep):,}  (removed {n_dups:,} salt/solvate dups)")

    out_cols = [c for c in ("molecule_chembl_id", "pref_name", "canonical_smiles",
                            "atc_classifications", "max_phase") if c in keep.columns]
    keep[out_cols].to_csv(OUT, index=False)
    removed[[c for c in ("molecule_chembl_id", "pref_name", "atc_classifications",
                         "_abx_reason") if c in removed.columns]].to_csv(REMOVED, index=False)
    print(f"\nClean library  : {OUT.relative_to(ROOT)}  ({len(keep):,} rows)")
    print(f"Removed report : {REMOVED.relative_to(ROOT)}  ({len(removed):,} rows)")

    # sanity: the two known leaks must be gone
    leak_check = keep["pref_name"].astype(str).str.upper().str.contains(
        "MOXALACTAM|QUINUPRISTIN", na=False)
    print(f"\nSanity — MOXALACTAM/QUINUPRISTIN in clean library: {leak_check.sum()} "
          f"(must be 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
