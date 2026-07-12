"""Score the clean non-antibiotic screen library against every per-target model.

Turns the pooled whole-organism screen into a target-resolved one: each candidate
gets a predicted activity probability for each modelable ESKAPE protein target.
This is the mechanistic evidence layer the team review asked for.

Output:
  data/target_screen.csv   molecule_chembl_id, pref_name, canonical_smiles,
                           one probability column per target slug

Run (after train_target_models.py):
    python scripts/screen_targets.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import joblib

from amr_repurposing.config import (
    DATA_DIR as DATA, PROJECT_ROOT as ROOT, TARGETS_DIR,
    TARGET_MODELS_DIR as MODEL_DIR, ECFP_N_BITS as N_BITS,
)
from amr_repurposing.features.fingerprints import ecfp4

SCREEN_LIB = DATA / "screen_library_clean.csv"
MANIFEST = TARGETS_DIR / "manifest.csv"
OUT = DATA / "target_screen.csv"


def main() -> int:
    if not SCREEN_LIB.exists() or not MANIFEST.exists():
        print("missing screen_library_clean.csv or targets/manifest.csv — "
              "run build_screen_library.py and build_target_datasets.py first")
        return 1

    lib = pd.read_csv(SCREEN_LIB).dropna(subset=["canonical_smiles"]).reset_index(drop=True)
    manifest = pd.read_csv(MANIFEST)
    print(f"Screen library : {len(lib):,} molecules")
    print(f"Targets        : {len(manifest)}")

    # featurize once, reuse across all target models
    fps, keep_idx = [], []
    for i, smi in enumerate(lib["canonical_smiles"]):
        fp = ecfp4(smi)
        if fp is not None:
            fps.append(fp)
            keep_idx.append(i)
    X = np.asarray(fps)
    lib = lib.iloc[keep_idx].reset_index(drop=True)
    print(f"Featurized     : {len(lib):,} (dropped {len(keep_idx) and len(fps) - len(lib) or 0})")

    out = lib[["molecule_chembl_id", "pref_name", "canonical_smiles"]].copy()
    for _, m in manifest.iterrows():
        bundle = joblib.load(MODEL_DIR / f"{m['slug']}.joblib")
        out[m["slug"]] = np.round(bundle["model"].predict_proba(X)[:, 1], 4)

    out.to_csv(OUT, index=False)
    print(f"\nWrote target screen -> {OUT.relative_to(ROOT)}  "
          f"({len(out):,} x {len(manifest)} targets)")

    # top candidate-target pairs (long form), best per target
    slugs = manifest["slug"].tolist()
    long = out.melt(id_vars=["pref_name"], value_vars=slugs,
                    var_name="target", value_name="prob")
    print("\nTop candidate per target:")
    for slug, g in long.groupby("target"):
        top = g.loc[g["prob"].idxmax()]
        tname = manifest.loc[manifest["slug"] == slug, "target"].iloc[0]
        print(f"  {tname[:42]:42s} -> {top['pref_name'][:28]:28s} ({top['prob']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
