"""Reproduce the held-out evaluation of the deployed ensemble from saved artifacts.

Loads the MLP weights from ``models/best_mlp.pt`` and retrains the RF (seeded, cheap)
on the saved fingerprints/masks, then returns test-set probabilities and metrics.
Results are cached on the module for the session so figures share one evaluation.
"""
from __future__ import annotations

import numpy as np

from amr_repurposing.config import DATA_DIR, MODELS_DIR, SEED, get_device

_MODEL_CACHE: dict = {}


def load_model_eval() -> dict:
    """Return dict: y_test, mlp_prob, mlp_pred, rf_prob, rf, X_train, X_test,
    mlp_roc, mlp_prc, rf_roc, rf_prc. RF is retrained (seeded); MLP is loaded."""
    if _MODEL_CACHE:
        return _MODEL_CACHE

    import torch
    from torch.utils.data import Dataset, DataLoader
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score

    from amr_repurposing.models.architecture import AntibacterialMLP

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = get_device()

    X = np.load(DATA_DIR / "X_ecfp.npy")
    y = np.load(DATA_DIR / "y_labels.npy")
    train_mask = np.load(DATA_DIR / "train_mask.npy")
    test_mask = np.load(DATA_DIR / "test_mask.npy")
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    assert not (set(np.where(train_mask)[0]) & set(np.where(test_mask)[0]))

    rf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2,
                                class_weight="balanced", n_jobs=-1, random_state=SEED)
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)[:, 1]

    class FingerprintDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return self.X[i], self.y[i]

    model = AntibacterialMLP().to(device)
    ckpt = torch.load(MODELS_DIR / "best_mlp.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loader = DataLoader(FingerprintDataset(X_test, y_test), batch_size=512, shuffle=False)
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device)).cpu().numpy()
            probs.extend(1 / (1 + np.exp(-logits)))
    mlp_prob = np.array(probs)

    _MODEL_CACHE.update(dict(
        y_test=y_test, mlp_prob=mlp_prob, mlp_pred=(mlp_prob >= 0.5).astype(int),
        rf_prob=rf_prob, rf=rf, X_train=X_train, X_test=X_test,
        mlp_roc=roc_auc_score(y_test, mlp_prob),
        mlp_prc=average_precision_score(y_test, mlp_prob),
        rf_roc=roc_auc_score(y_test, rf_prob),
        rf_prc=average_precision_score(y_test, rf_prob),
    ))
    return _MODEL_CACHE
