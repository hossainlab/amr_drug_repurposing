"""
Fetch + clean + featurize the ESKAPE(+Mtb) bioactivity training set from ChEMBL.

This is the canonical, adapted-from-notebook-01 data step. The original fetch lived
only in ``notebooks/01_data_preprocessing.ipynb``; it is ported here verbatim in its
cleaning/labelling logic so a fresh, reproducible run needs no notebook.

Outputs (all under data/):
  - eskape_raw.csv      raw ChEMBL /activity rows for the 8 ESKAPE(+Mtb) organisms
  - eskape_clean.csv    cleaned + binary-labelled, one row per (molecule, organism)
  - X_ecfp.npy          ECFP4 (2048-bit) fingerprint matrix, row-aligned to y/smiles
  - y_labels.npy        binary activity labels
  - smiles_valid.npy    canonical SMILES aligned to X/y (for ChemBERTa / audit)

Two deliberate additions over the notebook fetch:
  - the ``/activity`` field list now includes ``target_chembl_id`` and
    ``target_pref_name`` so the per-target QSAR layer (screen/target_datasets.py)
    can resolve protein targets — the notebook omitted these.
  - featurization reuses the shared ``features.fingerprints.ecfp4`` (radius 2,
    2048 bits) instead of a private Morgan copy, so every stage is identical.

Run:
    amr-fetch-eskape       # or: python -m amr_repurposing.data.fetch_eskape
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import numpy as np
import pandas as pd

from amr_repurposing.config import DATA_DIR as DATA, PROJECT_ROOT as ROOT, SEED
from amr_repurposing.features.fingerprints import ecfp4

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
PAGE = 1000
MAX_PER_ORG = 20_000  # cap per organism (mirrors notebook 01)
MAX_WORKERS = 5       # concurrent page requests (respects ChEMBL fair-use)

ESKAPE_ORGANISMS = [
    "Staphylococcus aureus",
    "Klebsiella pneumoniae",
    "Acinetobacter baumannii",
    "Pseudomonas aeruginosa",
    "Enterococcus faecium",
    "Enterobacter cloacae",
    "Escherichia coli",
    "Mycobacterium tuberculosis",
]

# NB: target_chembl_id / target_pref_name added over the notebook so the per-target
# QSAR layer can resolve protein targets from this same file.
FIELDS = (
    "molecule_chembl_id,canonical_smiles,standard_type,standard_value,"
    "standard_units,target_organism,target_chembl_id,target_pref_name,"
    "assay_chembl_id,activity_comment,pchembl_value"
)

RAW = DATA / "eskape_raw.csv"
CLEAN = DATA / "eskape_clean.csv"
CACHE = DATA / "_eskape_cache"   # per-organism checkpoints (resumable fetch)


def _slug(organism: str) -> str:
    return organism.lower().replace(" ", "_")


def _get(client: httpx.Client, params: dict, retries: int = 4) -> dict | None:
    for attempt in range(retries):
        try:
            r = client.get(f"{BASE_URL}/activity.json", params=params, timeout=90)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"    request failed (offset={params.get('offset')}): {e}")
                return None
            time.sleep(2 ** attempt)
    return None


def fetch_organism(client: httpx.Client, organism: str) -> pd.DataFrame:
    """Fetch one organism's activities with CONCURRENT pagination (5 workers).

    ChEMBL deep-offset queries are slow one-at-a-time; firing the page requests in
    parallel (like the original notebook's async version) cuts wall-time ~5×. Result
    is checkpointed to CACHE/<slug>.csv so a rerun skips organisms already fetched."""
    cache = CACHE / f"{_slug(organism)}.csv"
    if cache.exists() and cache.stat().st_size > 0:
        df = pd.read_csv(cache)
        print(f"  {organism}: cached ({len(df):,} rows)", flush=True)
        return df

    base = {
        "target_organism": organism,
        "standard_type__in": "MIC,IC50,MBC,Inhibition",
        "standard_value__isnull": False,
        "assay_type__in": "F,B",
        "limit": PAGE,
        "offset": 0,
        "format": "json",
        "fields": FIELDS,
    }
    first = _get(client, base)
    if first is None:
        return pd.DataFrame()
    total = first["page_meta"]["total_count"]
    records = list(first["activities"])
    cap = min(total, MAX_PER_ORG)
    offsets = list(range(PAGE, cap, PAGE))
    print(f"  {organism}: {total:,} total (fetching {cap:,} in {len(offsets) + 1} pages)", flush=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_get, client, {**base, "offset": o}): o for o in offsets}
        for fut in as_completed(futures):
            page = fut.result()
            if page:
                records.extend(page["activities"])

    df = pd.DataFrame(records)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)   # checkpoint so a kill/restart resumes here
    print(f"  {organism}: fetched {len(df):,} rows -> cache", flush=True)
    return df


def fetch_all() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    limits = httpx.Limits(max_connections=MAX_WORKERS, max_keepalive_connections=MAX_WORKERS)
    with httpx.Client(limits=limits) as client:
        for org in ESKAPE_ORGANISMS:
            print(f"Fetching: {org}", flush=True)
            frames.append(fetch_organism(client, org))
            time.sleep(1)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def clean_and_label(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, unit-normalise, and binary-label activities (ported from notebook 01).

    Labels (active=1): MIC<=8, IC50<=10, MBC<=16 (all µM), Inhibition>=50 (%). One
    row per (molecule, organism), keeping the most-potent measurement."""
    df = df.copy()
    df = df.dropna(subset=["canonical_smiles", "standard_value"])
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value"])
    df = df[df["standard_value"] > 0]
    df = df[df["canonical_smiles"].astype(str).str.len() > 5]
    df = df[~df["canonical_smiles"].astype(str).str.contains(r"\*", regex=True)]

    def to_uM(row):
        val = row["standard_value"]
        units = str(row.get("standard_units", "")).lower().strip()
        stype = str(row.get("standard_type", "")).upper()
        if stype == "INHIBITION":
            return val
        if units in ("nm", "nanomolar"):
            return val / 1000
        if units in ("mm", "millimolar"):
            return val * 1000
        return val  # µg/mL ≈ µM approximation common in AMR

    df["value_uM"] = df.apply(to_uM, axis=1)

    def label(row):
        stype = str(row["standard_type"]).upper()
        val = row["value_uM"]
        if stype == "MIC":
            return 1 if val <= 8 else 0
        if stype == "IC50":
            return 1 if val <= 10 else 0
        if stype == "MBC":
            return 1 if val <= 16 else 0
        if stype == "INHIBITION":
            return 1 if val >= 50 else 0
        return np.nan

    df["active"] = df.apply(label, axis=1).astype("float")
    df = df.dropna(subset=["active"])
    df["active"] = df["active"].astype(int)
    df = (df.sort_values("value_uM")
            .drop_duplicates(subset=["molecule_chembl_id", "target_organism"], keep="first"))
    return df.reset_index(drop=True)


def featurize(df_clean: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ECFP4 (radius 2, 2048 bits) for every parseable SMILES, row-aligned to labels.

    Order mirrors the notebook loop so train_mask/test_mask stay valid across runs."""
    fps, labels, smiles = [], [], []
    for _, row in df_clean.iterrows():
        fp = ecfp4(row["canonical_smiles"])
        if fp is not None:
            fps.append(fp)
            labels.append(int(row["active"]))
            smiles.append(row["canonical_smiles"])
    X = np.asarray(fps, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    return X, y, np.asarray(smiles)


def main() -> int:
    np.random.seed(SEED)
    DATA.mkdir(parents=True, exist_ok=True)

    if RAW.exists() and RAW.stat().st_size > 0:
        print(f"Loading cached raw data: {RAW.relative_to(ROOT)}")
        df_raw = pd.read_csv(RAW)
    else:
        print("Fetching ESKAPE(+Mtb) bioactivity from ChEMBL /activity ...")
        t0 = time.time()
        df_raw = fetch_all()
        if df_raw.empty:
            print("ERROR: no records returned from ChEMBL")
            return 1
        df_raw.to_csv(RAW, index=False)
        print(f"Raw fetch done in {(time.time() - t0) / 60:.1f} min "
              f"-> {RAW.relative_to(ROOT)}")
    print(f"Raw shape: {df_raw.shape}")

    df_clean = clean_and_label(df_raw)
    df_clean.to_csv(CLEAN, index=False)
    vc = df_clean["active"].value_counts()
    n = len(df_clean)
    print(f"\nClean dataset: {df_clean.shape} -> {CLEAN.relative_to(ROOT)}")
    print(f"  Active   (1): {vc.get(1, 0):,}  ({vc.get(1, 0) / n * 100:.1f}%)")
    print(f"  Inactive (0): {vc.get(0, 0):,}  ({vc.get(0, 0) / n * 100:.1f}%)")
    print(df_clean.groupby("target_organism")["active"]
                  .agg(["sum", "count"]).rename(columns={"sum": "active", "count": "total"})
                  .assign(pct=lambda x: (x.active / x.total * 100).round(1))
                  .sort_values("total", ascending=False).to_string())

    print("\nGenerating ECFP4 fingerprints (radius 2, 2048 bits) ...")
    X, y, smiles = featurize(df_clean)
    np.save(DATA / "X_ecfp.npy", X)
    np.save(DATA / "y_labels.npy", y)
    np.save(DATA / "smiles_valid.npy", smiles)
    print(f"  Fingerprint matrix : {X.shape}")
    print(f"  Active: {int(y.sum()):,} | Inactive: {int((y == 0).sum()):,}")
    print("  Saved X_ecfp.npy / y_labels.npy / smiles_valid.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
