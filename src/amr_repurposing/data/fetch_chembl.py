"""
Fetch the FDA-approved small-molecule library from ChEMBL.

Writes data/approved_drugs.csv with the columns the screen needs, including
molecule_hierarchy (parent_chembl_id) so antibiotic ATC codes can be propagated
from parent molecules to their salt/solvate forms.

NOTE on indication_class: the ChEMBL /molecule endpoint no longer returns this
legacy field, so it cannot be fetched here. Antibiotic detection therefore relies
on ATC (own + parent-hierarchy), connectivity, and INN name stems — see
antibiotic_filter.py. This script does NOT fetch the separate drug_indication
endpoint because ATC J01/J04 is already the authoritative antibacterial signal.

Run:
    amr-fetch          # or: python -m amr_repurposing.data.fetch_chembl
"""
from __future__ import annotations

import shutil
import time

import httpx
import pandas as pd

from amr_repurposing.config import DATA_DIR as DATA, PROJECT_ROOT as ROOT

OUT = DATA / "approved_drugs.csv"
BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
PAGE = 1000


def fetch_page(client: httpx.Client, offset: int, retries: int = 4) -> dict | None:
    params = {
        "max_phase": 4,
        "molecule_type": "Small molecule",
        "limit": PAGE,
        "offset": offset,
        "format": "json",
    }
    for attempt in range(retries):
        try:
            r = client.get(f"{BASE_URL}/molecule.json", params=params, timeout=90)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"  page offset={offset} failed: {e}")
                return None
            time.sleep(2 ** attempt)
    return None


def main() -> int:
    DATA.mkdir(exist_ok=True)
    records: list[dict] = []
    with httpx.Client() as client:
        first = fetch_page(client, 0)
        if first is None:
            print("ERROR: could not reach ChEMBL")
            return 1
        total = first["page_meta"]["total_count"]
        records.extend(first["molecules"])
        print(f"Total approved small-molecule drugs: {total:,}")
        for offset in range(PAGE, total, PAGE):
            page = fetch_page(client, offset)
            if page:
                records.extend(page["molecules"])
            print(f"  fetched {len(records):,}/{total:,}", end="\r")
    print()

    df = pd.DataFrame(records)
    # flatten canonical_smiles out of molecule_structures
    if "molecule_structures" in df.columns:
        df["canonical_smiles"] = df["molecule_structures"].apply(
            lambda x: x.get("canonical_smiles") if isinstance(x, dict) else None)
    keep = [c for c in (
        "molecule_chembl_id", "pref_name", "max_phase",
        "atc_classifications", "molecule_hierarchy", "molecule_structures",
        "canonical_smiles",
    ) if c in df.columns]
    df = df[keep]

    if OUT.exists():
        bak = OUT.with_suffix(".csv.bak")
        shutil.copy2(OUT, bak)
        print(f"Backed up existing -> {bak.relative_to(ROOT)}")
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df):,} drugs -> {OUT.relative_to(ROOT)}")

    n_atc = df["atc_classifications"].fillna("[]").ne("[]").sum() if \
        "atc_classifications" in df.columns else 0
    n_hier = df["molecule_hierarchy"].notna().sum() if \
        "molecule_hierarchy" in df.columns else 0
    print(f"  rows with ATC codes        : {n_atc:,}")
    print(f"  rows with molecule_hierarchy: {n_hier:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
