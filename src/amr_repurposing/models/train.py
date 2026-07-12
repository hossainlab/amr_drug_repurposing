"""
Train the antibacterial-activity ensemble (RF + deep MLP) — canonical, adapted from
notebook 02, with the leaky screen filter removed (scoring is a separate stage that
reads the clean screen library; see screen/score.py).

Pipeline:
  1. load X_ecfp.npy / y_labels.npy + eskape_clean.csv (from amr-fetch-eskape)
  2. molecule-level train/test split (majority-vote label per molecule, stratified) —
     guarantees zero molecule overlap between train and test
  3. Random Forest (300 trees, balanced) + isotonic calibration on a held-out split
  4. deep MLP (AntibacterialMLP) with class-balanced sampling, 60 epochs, best-by-ROC
  5. per-organism RF baselines

Artifacts (data/ and models/):
  - data/train_mask.npy, data/test_mask.npy      molecule-level split masks
  - data/organism_rf_results.csv                 per-pathogen RF metrics
  - models/best_mlp.pt                            best MLP checkpoint (by test ROC-AUC)
  - models/training_history.npz                   loss / val-AUC / val-PRC curves + RF baseline
  - models/rf_calibrated.joblib                   calibrated RF used by the screen scorer

ChemBERTa fine-tuning (notebook 02, supplementary figure S2) is intentionally NOT
ported here — it is optional, GPU-heavy, and not part of the deployed ensemble.

Run:
    amr-train        # or: python -m amr_repurposing.models.train
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from amr_repurposing.config import DATA_DIR as DATA, MODELS_DIR as MODELS, SEED, get_device
from amr_repurposing.models.architecture import AntibacterialMLP

RESULTS_ORG = DATA / "organism_rf_results.csv"
MLP_CKPT = MODELS / "best_mlp.pt"
HISTORY = MODELS / "training_history.npz"
RF_CAL_PATH = MODELS / "rf_calibrated.joblib"

EPOCHS = 60
LOG_EVERY = 5
ORG_MIN_SAMPLES = 300
ORG_MIN_POSITIVE = 60


def _load_aligned():
    """Load fingerprints/labels and the row-aligned clean frame (mirrors notebook 02)."""
    from rdkit import Chem

    X = np.load(DATA / "X_ecfp.npy")
    y = np.load(DATA / "y_labels.npy")
    df = pd.read_csv(DATA / "eskape_clean.csv")
    valid = df["canonical_smiles"].apply(
        lambda s: isinstance(s, str) and Chem.MolFromSmiles(s) is not None)
    df_fp = df[valid].reset_index(drop=True)
    assert len(df_fp) == len(X), f"fingerprint/clean mismatch: X={len(X)} valid={len(df_fp)}"
    return X, y, df_fp


def _molecule_split(df_fp):
    """Molecule-level stratified 80/20 split — zero molecule overlap between sets."""
    from sklearn.model_selection import train_test_split

    mol_label = (df_fp.groupby("molecule_chembl_id")["active"]
                      .agg(lambda x: int(x.mean() >= 0.5)))
    mol_ids = mol_label.index.to_numpy().astype(str)
    mol_y = mol_label.to_numpy().astype(int)
    train_mols, test_mols = train_test_split(
        mol_ids, test_size=0.2, random_state=SEED, stratify=mol_y)
    train_set, test_set = set(train_mols), set(test_mols)
    assert not (train_set & test_set), "molecule overlap between train/test"
    ids = df_fp["molecule_chembl_id"].astype(str)
    return ids.isin(train_set).values, ids.isin(test_set).values, len(mol_ids)


def _train_rf(X_train, y_train, X_test, y_test):
    """RF (300 trees) + isotonic calibration on a held-out slice of the training set."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.model_selection import train_test_split

    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train)
    rf_base = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=SEED)
    rf_base.fit(X_fit, y_fit)
    # FrozenEstimator = the sklearn>=1.6 replacement for the removed cv="prefit":
    # calibrate the already-fit RF on the held-out slice without refitting it.
    rf_cal = CalibratedClassifierCV(FrozenEstimator(rf_base), method="isotonic")
    rf_cal.fit(X_cal, y_cal)

    prob = rf_cal.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, prob)
    prc = average_precision_score(y_test, prob)
    return rf_cal, roc, prc


def _train_mlp(X_train, y_train, X_test, y_test, device):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from sklearn.metrics import roc_auc_score, average_precision_score

    torch.manual_seed(SEED)
    batch = 128 if device.type in ("mps", "cpu") else 256

    class FP(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return self.X[i], self.y[i]

    n_neg, n_pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    sample_w = np.where(y_train == 1, n_neg / n_pos, 1.0)
    sampler = WeightedRandomSampler(sample_w, num_samples=len(y_train), replacement=True)
    train_loader = DataLoader(FP(X_train, y_train), batch_size=batch, sampler=sampler, num_workers=0)
    test_loader = DataLoader(FP(X_test, y_test), batch_size=batch, shuffle=False, num_workers=0)

    model = AntibacterialMLP(input_dim=X_train.shape[1]).to(device)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

    def evaluate():
        model.eval()
        logits, labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                logits.append(model(xb.to(device)).cpu())
                labels.append(yb)
        p = 1 / (1 + np.exp(-torch.cat(logits).numpy()))
        lab = torch.cat(labels).numpy()
        return roc_auc_score(lab, p), average_precision_score(lab, p)

    print(f"MLP parameters: {sum(p.numel() for p in model.parameters()):,}  (device={device})")
    losses, aucs, prcs, best = [], [], [], 0.0
    for epoch in range(EPOCHS):
        model.train()
        ep_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item()
        scheduler.step()
        losses.append(ep_loss / len(train_loader))
        roc, prc = evaluate()
        aucs.append(roc)
        prcs.append(prc)
        marker = ""
        if roc > best:
            best = roc
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(), "best_auc": best}, MLP_CKPT)
            marker = " <- best"
        if (epoch + 1) % LOG_EVERY == 0:
            print(f"  epoch {epoch + 1:3d}/{EPOCHS} | loss {losses[-1]:.4f} | "
                  f"ROC {roc:.4f} | PRC {prc:.4f}{marker}")
    return np.array(losses), np.array(aucs), np.array(prcs), best


def _organism_rf(X, y, organism_arr):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.model_selection import train_test_split

    rows = []
    for org in np.unique(organism_arr):
        mask = organism_arr == org
        Xo, yo = X[mask], y[mask]
        n_pos, n_tot = int(yo.sum()), len(yo)
        short = " ".join(org.split()[:2])
        if n_tot < ORG_MIN_SAMPLES or n_pos < ORG_MIN_POSITIVE or (n_tot - n_pos) < ORG_MIN_POSITIVE:
            print(f"  SKIP {short:<28s} n={n_tot:5,} pos={n_pos:4,} (insufficient)")
            rows.append({"organism": org, "n_total": n_tot, "n_active": n_pos,
                         "roc_auc": None, "prc_auc": None, "status": "skipped"})
            continue
        Xtr, Xte, ytr, yte = train_test_split(Xo, yo, test_size=0.2, random_state=SEED, stratify=yo)
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                    class_weight="balanced", n_jobs=-1, random_state=SEED)
        rf.fit(Xtr, ytr)
        prob = rf.predict_proba(Xte)[:, 1]
        roc = roc_auc_score(yte, prob)
        prc = average_precision_score(yte, prob)
        print(f"  {short:<28s} n={n_tot:5,} pos={n_pos:4,} ROC={roc:.3f} PRC={prc:.3f}")
        rows.append({"organism": org, "n_total": n_tot, "n_active": n_pos,
                     "roc_auc": round(roc, 4), "prc_auc": round(prc, 4), "status": "trained"})
    return pd.DataFrame(rows)


def main() -> int:
    import joblib

    if not (DATA / "X_ecfp.npy").exists():
        print("missing data/X_ecfp.npy — run `amr-fetch-eskape` first")
        return 1
    np.random.seed(SEED)
    device = get_device()

    X, y, df_fp = _load_aligned()
    organism_arr = df_fp["target_organism"].values

    train_mask, test_mask, n_mols = _molecule_split(df_fp)
    np.save(DATA / "train_mask.npy", train_mask)
    np.save(DATA / "test_mask.npy", test_mask)
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    print(f"Molecules total: {n_mols:,} | train rows {X_train.shape[0]:,} | test rows {X_test.shape[0]:,}")
    print(f"Train pos-rate {y_train.mean():.3f} | test pos-rate {y_test.mean():.3f}")

    print("\nTraining Random Forest (300 trees) + isotonic calibration ...")
    t0 = time.time()
    rf_cal, rf_roc, rf_prc = _train_rf(X_train, y_train, X_test, y_test)
    joblib.dump(rf_cal, RF_CAL_PATH)
    print(f"RF done in {time.time() - t0:.0f}s | ROC {rf_roc:.4f} PRC {rf_prc:.4f}"
          f" -> {RF_CAL_PATH.name}")

    print("\nTraining deep MLP ...")
    losses, aucs, prcs, mlp_best = _train_mlp(X_train, y_train, X_test, y_test, device)
    np.savez(HISTORY, train_losses=losses, val_aucs=aucs, val_prcs=prcs,
             baseline_results=np.array({"roc_auc": rf_roc, "prc_auc": rf_prc}))
    print(f"MLP best test ROC-AUC: {mlp_best:.4f} -> {MLP_CKPT.name} / {HISTORY.name}")

    print("\nTraining per-organism RF baselines ...")
    df_org = _organism_rf(X, y, organism_arr)
    df_org.to_csv(RESULTS_ORG, index=False)
    print(f"Organism results -> {RESULTS_ORG.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
