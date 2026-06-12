"""Supplementary Figure S5 — RF vs MLP performance comparison."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import *  # noqa: F403,E402


def main():
    m = load_model_eval()
    models = ["Random Forest", "Deep MLP"]
    roc_scores = [m["rf_roc"], m["mlp_roc"]]
    prc_scores = [m["rf_prc"], m["mlp_prc"]]

    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    x = np.arange(2); w = 0.32
    b1 = ax.bar(x - w / 2, roc_scores, w, color=C["active"], label="ROC-AUC",
                edgecolor="none", zorder=3)
    b2 = ax.bar(x + w / 2, prc_scores, w, color=C["ensemble"], label="PRC-AUC",
                edgecolor="none", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=7)
    ax.set_ylim(0.80, 1.015)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.set_ylabel("Score"); ax.legend(fontsize=6)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002, f"{h:.3f}",
                ha="center", va="bottom", fontsize=5.5)
    add_panel_label(ax, "A", x=-0.18)

    save_fig(fig, "figure_S5_model_comparison")


if __name__ == "__main__":
    main()
