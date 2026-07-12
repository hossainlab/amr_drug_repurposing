"""Annotate each repurposing candidate with a likely molecular target (MoA).

Two complementary signals, both from data already in the repo:
  - predicted_target : the target whose known ACTIVE compounds the candidate most
    resembles (max Tanimoto over ECFP4 to the per-target active sets); assigned
    only when that similarity clears SIM_THRESHOLD, else "none".
  - target_score     : the predicted_target's QSAR model probability for the
    candidate (from train_target_models.py), i.e. how the model itself rates it.

The three columns (predicted_target, nn_similarity, target_score) are appended to
data/repurposing_candidates_clean.csv. They are additive — existing figures that
read the file ignore the new columns.

Run (after train_target_models.py):
    python scripts/annotate_targets.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from rdkit.Chem import DataStructs

from amr_repurposing.config import (
    DATA_DIR as DATA, PROJECT_ROOT as ROOT, TARGETS_DIR as TARGET_DIR,
    TARGET_MODELS_DIR as MODEL_DIR,
)
from amr_repurposing.features.fingerprints import ecfp4, ecfp4_bitvect

MANIFEST = TARGET_DIR / "manifest.csv"
CANDIDATES = DATA / "repurposing_candidates_clean.csv"

SIM_THRESHOLD = 0.35   # min Tanimoto to nearest active to assign a target


def morgan(smiles: str):
    """Return (rdkit bit vector, numpy array) for one SMILES, or (None, None)."""
    bv = ecfp4_bitvect(smiles)
    if bv is None:
        return None, None
    return bv, ecfp4(smiles)


def main() -> int:
    if not CANDIDATES.exists() or not MANIFEST.exists():
        print("missing candidates or targets/manifest.csv — run clean_candidates.py "
              "and build_target_datasets.py first")
        return 1

    manifest = pd.read_csv(MANIFEST)

    # pool of known actives: bit vectors tagged with their target name
    active_fps, active_tgt = [], []
    for _, m in manifest.iterrows():
        df = pd.read_csv(TARGET_DIR / f"{m['slug']}.csv")
        for smi in df.loc[df["active"] == 1, "canonical_smiles"]:
            bv, _ = morgan(smi)
            if bv is not None:
                active_fps.append(bv)
                active_tgt.append(m["target"])
    active_tgt = np.array(active_tgt)
    models = {m["target"]: joblib.load(MODEL_DIR / f"{m['slug']}.joblib")["model"]
              for _, m in manifest.iterrows()}
    print(f"Active reference compounds : {len(active_fps):,} across {len(manifest)} targets")

    cand = pd.read_csv(CANDIDATES)
    pred_target, nn_sim, tgt_score = [], [], []
    for smi in cand["smiles"]:
        bv, arr = morgan(smi)
        if bv is None:
            pred_target.append("none"); nn_sim.append(0.0); tgt_score.append(0.0)
            continue
        sims = np.array(DataStructs.BulkTanimotoSimilarity(bv, active_fps))
        j = int(sims.argmax())
        best_sim = float(sims[j])
        if best_sim >= SIM_THRESHOLD:
            tgt = active_tgt[j]
            pred_target.append(tgt)
            nn_sim.append(round(best_sim, 3))
            tgt_score.append(round(float(models[tgt].predict_proba(arr.reshape(1, -1))[0, 1]), 4))
        else:
            pred_target.append("none"); nn_sim.append(round(best_sim, 3)); tgt_score.append(0.0)

    cand["predicted_target"] = pred_target
    cand["nn_similarity"] = nn_sim
    cand["target_score"] = tgt_score
    cand.to_csv(CANDIDATES, index=False)

    assigned = (cand["predicted_target"] != "none").sum()
    print(f"Annotated {len(cand):,} candidates; {assigned:,} assigned a target "
          f"(Tanimoto >= {SIM_THRESHOLD}), {len(cand) - assigned:,} 'none'.")
    print("\nTarget assignment counts:")
    print(cand[cand["predicted_target"] != "none"]["predicted_target"]
          .value_counts().to_string())
    print("\nTop-15 candidates (by ensemble score) with predicted target:")
    cols = ["drug_name", "prob_ensemble", "predicted_target", "nn_similarity", "target_score"]
    print(cand.sort_values("prob_ensemble", ascending=False)[cols].head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
