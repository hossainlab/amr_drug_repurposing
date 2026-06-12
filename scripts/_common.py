"""
Shared style + data/model loaders for the figure scripts (fig1.py ... figS5.py).

Every figure script does:  from _common import *
Heavy artifacts (RF retrain, MLP load, fingerprints) are loaded lazily via
load_model_eval() so figures that need no model stay fast.

Data sources
------------
- data/repurposing_candidates_clean.csv : antibiotic-free scored candidates
  (produced by scripts/clean_candidates.py — DO NOT read the leaky raw file)
- data/eskape_clean.csv, X_ecfp.npy, y_labels.npy, train/test masks
- checkpoints/best_mlp.pt, training_history.npz, chembert/*/trainer_state.json
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: scripts save files, never show windows
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
FIG_DIR = PROJECT_ROOT / "figures"
for d in (DATA_DIR, CKPT_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42

# ── Journal-quality matplotlib defaults (Nature / Cell style) ────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "legend.title_fontsize": 6, "legend.frameon": False,
    "legend.borderpad": 0.3, "legend.labelspacing": 0.25,
    "legend.handlelength": 1.5, "legend.handletextpad": 0.4,
    "axes.linewidth": 0.75, "xtick.major.width": 0.75, "ytick.major.width": 0.75,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.minor.visible": False, "ytick.minor.visible": False,
    "lines.linewidth": 1.25, "patch.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# Nature-style colour palette
C = {
    "active": "#2166AC", "inactive": "#B2182B", "mlp": "#1B7837",
    "rf": "#762A83", "ensemble": "#D6604D", "neutral": "#969696",
    "light": "#D1E5F0", "amber": "#F4A582",
}

PANEL_KW = dict(fontsize=9, fontweight="bold", va="top")


def add_panel_label(ax, letter, x=-0.14, y=1.05):
    ax.text(x, y, letter, transform=ax.transAxes, **PANEL_KW)


def save_fig(fig, name):
    for fmt in ("png", "svg"):
        fig.savefig(FIG_DIR / f"{name}.{fmt}", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"  Saved: {name}.png / .svg")


# ── Data loaders ─────────────────────────────────────────────────────────────
def load_eskape_clean() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "eskape_clean.csv")


def load_clean_candidates():
    """Return (df_results, df_druglike) from the antibiotic-free candidate file."""
    path = DATA_DIR / "repurposing_candidates_clean.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run: python scripts/clean_candidates.py")
    df = pd.read_csv(path).sort_values("prob_ensemble", ascending=False).reset_index(drop=True)
    if "drug_like" in df.columns:
        dl = df[df["drug_like"].isin([1, True, "True"])].reset_index(drop=True)
    else:
        dl = df.copy()
    return df, dl


# ── Lazy model evaluation (RF retrain + MLP load) — cached on module ──────────
_MODEL_CACHE = {}


def load_model_eval():
    """
    Reproduce nb02/nb03 held-out evaluation from saved artifacts.
    Returns dict: y_test, mlp_prob, mlp_pred, rf_prob, rf, X_train, X_test,
                  mlp_roc, mlp_prc, rf_roc, rf_prc.
    RF is retrained (cheap, seeded); MLP weights are loaded from checkpoint.
    """
    if _MODEL_CACHE:
        return _MODEL_CACHE

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

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
        def __len__(self): return len(self.y)
        def __getitem__(self, i): return self.X[i], self.y[i]

    class AntibacterialMLP(nn.Module):
        def __init__(self, input_dim=2048, dropout=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout * 0.5),
                nn.Linear(128, 1),
            )
        def forward(self, x): return self.net(x).squeeze(-1)

    model = AntibacterialMLP().to(device)
    ckpt = torch.load(CKPT_DIR / "best_mlp.pt", map_location=device, weights_only=False)
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
