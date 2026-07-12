"""Train a per-target QSAR model for each modelable ESKAPE protein target.

Reads the datasets from build_target_datasets.py, featurizes with ECFP4 (Morgan
radius=2, 2048 bits — the same representation as the whole-organism model), trains
a seeded RandomForest, and reports out-of-fold ROC-AUC / PRC-AUC from stratified
cross-validation. Small targets (few molecules) are flagged exploratory: their CV
metrics carry wide variance and should be read as hypothesis-generating.

Output:
  data/target_model_metrics.csv        target, slug, n_mols, cv ROC/PRC AUC, exploratory
  checkpoints/target_models/<slug>.joblib   RF fit on all rows (for screening)

Run:
    python scripts/train_target_models.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score

from amr_repurposing.config import (
    DATA_DIR as DATA, PROJECT_ROOT as ROOT, TARGETS_DIR as TARGET_DIR,
    TARGET_MODELS_DIR as MODEL_DIR, SEED, ECFP_N_BITS as N_BITS,
)
from amr_repurposing.features.fingerprints import ecfp4

MANIFEST = TARGET_DIR / "manifest.csv"
METRICS_OUT = DATA / "target_model_metrics.csv"

EXPLORATORY_BELOW = 100   # < this many molecules -> flag as exploratory


def featurize(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for _, r in df.iterrows():
        fp = ecfp4(r["canonical_smiles"])
        if fp is not None:
            X.append(fp)
            y.append(int(r["active"]))
    return np.asarray(X), np.asarray(y)


def main() -> int:
    if not MANIFEST.exists():
        print(f"missing {MANIFEST} — run scripts/build_target_datasets.py first")
        return 1

    manifest = pd.read_csv(MANIFEST)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    rows = []
    for _, m in manifest.iterrows():
        df = pd.read_csv(TARGET_DIR / f"{m['slug']}.csv")
        X, y = featurize(df)
        n_min = int(min(y.sum(), len(y) - y.sum()))       # minority class size
        n_splits = max(2, min(5, n_min))                  # adapt folds to minority
        rf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2,
                                    class_weight="balanced", n_jobs=-1, random_state=SEED)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        oof = cross_val_predict(rf, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        roc = roc_auc_score(y, oof)
        prc = average_precision_score(y, oof)

        rf.fit(X, y)                                      # final fit on all data
        joblib.dump({"model": rf, "target": m["target"], "n_bits": N_BITS},
                    MODEL_DIR / f"{m['slug']}.joblib")

        rows.append(dict(target=m["target"], slug=m["slug"], n_mols=len(y),
                         n_active=int(y.sum()), cv_folds=n_splits,
                         cv_roc_auc=round(roc, 3), cv_prc_auc=round(prc, 3),
                         exploratory=bool(len(y) < EXPLORATORY_BELOW)))
        print(f"  {m['target'][:45]:45s} n={len(y):4d}  ROC-AUC={roc:.3f}  PRC-AUC={prc:.3f}"
              f"{'  [exploratory]' if len(y) < EXPLORATORY_BELOW else ''}")

    metrics = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)
    metrics.to_csv(METRICS_OUT, index=False)
    print(f"\nWrote metrics -> {METRICS_OUT.relative_to(ROOT)}")
    print(f"Wrote {len(rows)} models -> {MODEL_DIR.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
