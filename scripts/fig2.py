"""Figure 2 — Model performance (ROC, PR, confusion matrix) on held-out test set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import *  # noqa: F403,E402
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix  # noqa: E402


def main():
    m = load_model_eval()
    y_test, mlp_prob, mlp_pred = m["y_test"], m["mlp_prob"], m["mlp_pred"]
    rf_prob = m["rf_prob"]
    mlp_roc, mlp_prc, rf_roc, rf_prc = m["mlp_roc"], m["mlp_prc"], m["rf_roc"], m["rf_prc"]
    print(f"  MLP ROC={mlp_roc:.4f} PRC={mlp_prc:.4f} | RF ROC={rf_roc:.4f} PRC={rf_prc:.4f}")

    fig = plt.figure(figsize=(7.0, 2.8))
    gs = fig.add_gridspec(1, 3, wspace=0.46, left=0.06, right=0.97)
    ax1, ax2, ax3 = (fig.add_subplot(gs[i]) for i in range(3))

    # Panel A: ROC
    fpr_m, tpr_m, _ = roc_curve(y_test, mlp_prob)
    fpr_r, tpr_r, _ = roc_curve(y_test, rf_prob)
    ax1.fill_between(fpr_m, tpr_m, alpha=0.10, color=C["mlp"], zorder=2)
    ax1.plot(fpr_m, tpr_m, color=C["mlp"], lw=1.25, zorder=3, label=f"MLP  AUC = {mlp_roc:.3f}")
    ax1.plot(fpr_r, tpr_r, color=C["rf"], lw=1.25, ls="--", zorder=3, label=f"RF   AUC = {rf_roc:.3f}")
    ax1.plot([0, 1], [0, 1], color=C["neutral"], lw=0.6, ls=":", alpha=0.8, zorder=1)
    ax1.set_xlabel("False positive rate"); ax1.set_ylabel("True positive rate")
    ax1.set_xlim(-0.02, 1.02); ax1.set_ylim(-0.02, 1.02)
    ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.legend(loc="lower right"); add_panel_label(ax1, "A", x=-0.20)

    # Panel B: PR
    prec_m, rec_m, _ = precision_recall_curve(y_test, mlp_prob)
    prec_r, rec_r, _ = precision_recall_curve(y_test, rf_prob)
    ax2.fill_between(rec_m, prec_m, alpha=0.10, color=C["mlp"], zorder=2)
    ax2.plot(rec_m, prec_m, color=C["mlp"], lw=1.25, zorder=3, label=f"MLP  AUC = {mlp_prc:.3f}")
    ax2.plot(rec_r, prec_r, color=C["rf"], lw=1.25, ls="--", zorder=3, label=f"RF   AUC = {rf_prc:.3f}")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_xlim(-0.02, 1.02); ax2.set_ylim(-0.02, 1.02)
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.legend(loc="lower left"); add_panel_label(ax2, "B", x=-0.20)

    # Panel C: confusion matrix (MLP)
    cm_raw = confusion_matrix(y_test, mlp_pred)
    cm_pct = cm_raw / cm_raw.sum(axis=1, keepdims=True) * 100
    for spine in ax3.spines.values():
        spine.set_visible(True); spine.set_linewidth(0.5)
    im = ax3.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100, aspect="equal")
    ax3.set_xticks([0, 1]); ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["Inactive", "Active"], fontsize=6.5)
    ax3.set_yticklabels(["Inactive", "Active"], fontsize=6.5, rotation=90, va="center")
    ax3.set_xlabel("Predicted label"); ax3.set_ylabel("True label")
    for ii in range(2):
        for jj in range(2):
            col = "white" if cm_pct[ii, jj] > 60 else "black"
            ax3.text(jj, ii - 0.12, f"{cm_pct[ii,jj]:.1f}%", ha="center", va="center",
                     fontsize=7, fontweight="bold", color=col)
            ax3.text(jj, ii + 0.22, f"n = {cm_raw[ii,jj]:,}", ha="center", va="center",
                     fontsize=5.5, color=col, alpha=0.9)
    cb = fig.colorbar(im, ax=ax3, shrink=0.85, pad=0.03)
    cb.set_label("%", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    add_panel_label(ax3, "C", x=-0.24)

    save_fig(fig, "figure_2_model_performance")


if __name__ == "__main__":
    main()
