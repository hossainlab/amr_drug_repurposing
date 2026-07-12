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
    amr-build-library    # or: python -m amr_repurposing.screen.library
"""
from __future__ import annotations

import pandas as pd

from amr_repurposing.config import DATA_DIR as DATA, PROJECT_ROOT as ROOT, REFERENCE_DIR
from amr_repurposing.filtering.antibiotic_filter import (
    build_antibiotic_index,
    connectivity_key,
    parse_atc,
    parent_chembl_id,
    is_antifungal_or_antiviral,
    _atc_is_antibiotic,
    _name_is_antibiotic,
    _is_aware_antibacterial,
    aware_category,
    KEEP_NAME,
)

# Pre-review KEEP-override ATC set (BEFORE this review), hardcoded so the audit
# reconstructs the OLD verdict regardless of later changes to the live KEEP_ATC.
# It had the whole "D06B" group (which wrongly kept D06BA topical sulfonamides)
# and lacked the J01/J04 defeat.
_PRE_REVIEW_KEEP_ATC = ("J02", "D01", "A07AC", "G01AF", "G01AG", "J05", "D06B")
# nitroimidazoles / off-label agents named explicitly by the post-review filter
# (secnidazole/nitazoxanide are P01-coded, so the pre-review filter never saw them)
_NEW_NAME_STEMS = ("tinidazole", "ornidazole", "secnidazole", "nimorazole", "nitazoxanide")


def _is_newly_excluded(atc: list[str], name: str, reason: str) -> bool:
    """True if this removed drug would have LEAKED under the pre-review filter.

    Three post-review changes ADD removals: (1) a hard J01/J04 ATC now defeats the
    antifungal KEEP-override — before, a gynae G01AF code wrongly rescued dual-coded
    nitroimidazoles (metronidazole/tinidazole/ornidazole) and the "D06B" keep let
    D06BA topical sulfonamides through; (2) the P01-coded off-label agents
    secnidazole/nitazoxanide are now named. Salt forms already caught via
    parent-ATC before are NOT counted as newly excluded."""
    n = str(name).lower()
    if any(c.startswith(_PRE_REVIEW_KEEP_ATC) for c in atc) or any(kw in n for kw in KEEP_NAME):
        return True                                   # old KEEP-override exempted it
    if (any(s in n for s in _NEW_NAME_STEMS) and not _atc_is_antibiotic(atc)
            and "parent-ATC" not in str(reason)):
        return True                                   # P01 nitroimidazole, unseen before
    return False

APPROVED = DATA / "approved_drugs.csv"
OUT = DATA / "screen_library_clean.csv"
REMOVED = DATA / "screen_library_removed_antibiotics.csv"
AWARE_AUDIT = REFERENCE_DIR / "aware_removals_audit.csv"

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
        print(f"missing {APPROVED} — run `amr-fetch` first")
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
        if _is_aware_antibacterial(row.get("pref_name", "")):
            why.append("aware")
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

    # AWaRe audit: the behavioural DIFF from the team review — drugs that would
    # have LEAKED under the old filter but are now excluded. This is what the
    # review asked to surface (second-line antibacterials in a "non-antibiotic"
    # library), each with its citable AWaRe category. Drugs already caught before
    # (systemic J01/J04) are intentionally excluded from this file.
    newly = removed[removed.apply(
        lambda r: _is_newly_excluded(parse_atc(r.get("atc_classifications")),
                                     r.get("pref_name", ""), r.get("_abx_reason", "")),
        axis=1)].copy()
    if len(newly):
        newly["aware_category"] = newly["pref_name"].apply(aware_category)
        AWARE_AUDIT.parent.mkdir(parents=True, exist_ok=True)
        newly[["molecule_chembl_id", "pref_name", "aware_category",
               "atc_classifications", "_abx_reason"]].sort_values(
            "pref_name").to_csv(AWARE_AUDIT, index=False)
        print(f"AWaRe audit    : {AWARE_AUDIT.relative_to(ROOT)}  "
              f"({len(newly):,} newly-excluded second-line antibacterials)")

    # sanity: known leaks + second-line antibacterials flagged by the AWaRe layer
    # must all be gone from the clean library.
    # NB: FEXINIDAZOLE / BENZNIDAZOLE are antiprotozoal-only and intentionally KEPT.
    leak_pat = ("MOXALACTAM|QUINUPRISTIN|SECNIDAZOLE|NITAZOXANIDE|"
                "TINIDAZOLE|ORNIDAZOLE|METHENAMINE|NITROXOLINE|LINEZOLID|COLISTIN")
    leak_check = keep["pref_name"].astype(str).str.upper().str.contains(leak_pat, na=False)
    print(f"\nSanity — second-line/known-leak antibacterials in clean library: "
          f"{leak_check.sum()} (must be 0)")
    if leak_check.sum():
        print(keep.loc[leak_check, "pref_name"].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
