"""Shared plotting style + data loaders for the figure builders.

Every figure module does ``from amr_repurposing.viz.common import *``. Heavy model
evaluation is re-exported from ``amr_repurposing.models.evaluation`` so figures that
need no model stay import-light.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: builders save files, never show windows
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

from amr_repurposing.config import DATA_DIR, FIGURES_DIR, MODELS_DIR, SEED
from amr_repurposing.models.evaluation import load_model_eval  # re-exported for figures

# Legacy aliases some figure builders still reference via `import *`.
FIG_DIR = FIGURES_DIR
CKPT_DIR = MODELS_DIR

warnings.filterwarnings("ignore")

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
    ax.text(x, y, letter.lower(), transform=ax.transAxes, **PANEL_KW)


def save_fig(fig, name):
    for fmt in ("png", "svg"):
        fig.savefig(FIGURES_DIR / f"{name}.{fmt}", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"  Saved: {name}.png / .svg")


# ── Data loaders ─────────────────────────────────────────────────────────────
def load_eskape_clean() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "eskape_clean.csv")


def load_clean_candidates():
    """Return (df_results, df_druglike) from the antibiotic-free candidate file."""
    path = DATA_DIR / "repurposing_candidates_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run: amr-clean-candidates")
    df = pd.read_csv(path).sort_values("prob_ensemble", ascending=False).reset_index(drop=True)
    if "drug_like" in df.columns:
        dl = df[df["drug_like"].isin([1, True, "True"])].reset_index(drop=True)
    else:
        dl = df.copy()
    return df, dl


def load_target_metrics() -> pd.DataFrame:
    """Per-target QSAR cross-validation metrics (amr-train-targets)."""
    path = DATA_DIR / "target_model_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run: amr-train-targets")
    return pd.read_csv(path)


def load_target_screen() -> pd.DataFrame:
    """Screen-library × per-target probability matrix (amr-screen-targets)."""
    path = DATA_DIR / "target_screen.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run: amr-screen-targets")
    return pd.read_csv(path)
