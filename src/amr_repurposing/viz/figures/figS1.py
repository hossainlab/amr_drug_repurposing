"""Supplementary Figure S1 — MLP training dynamics (loss / val ROC / val PRC)."""
import sys
from pathlib import Path
from amr_repurposing.viz.common import *  # noqa: F403


def main():
    hist_path = CKPT_DIR / "training_history.npz"
    if not hist_path.exists():
        print(f"  {hist_path} not found — run nb02 to save training history.")
        return
    h = np.load(hist_path, allow_pickle=True)
    train_losses = h["train_losses"].tolist()
    val_aucs = h["val_aucs"].tolist()
    val_prcs = h["val_prcs"].tolist()
    baseline = h["baseline_results"].item()
    print(f"  Loaded {len(train_losses)} epochs of history")

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), gridspec_kw={"wspace": 0.40})
    epochs = np.arange(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, color=C["mlp"], lw=1.2, zorder=3)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE loss")
    add_panel_label(axes[0], "A", x=-0.22)

    axes[1].plot(epochs, val_aucs, color=C["active"], lw=1.2, zorder=3)
    axes[1].axhline(baseline["roc_auc"], color=C["rf"], ls="--", lw=0.8,
                    label=f"RF baseline ({baseline['roc_auc']:.3f})")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("ROC-AUC")
    axes[1].legend(fontsize=5.5); add_panel_label(axes[1], "B", x=-0.22)

    axes[2].plot(epochs, val_prcs, color=C["inactive"], lw=1.2, zorder=3)
    axes[2].axhline(baseline["prc_auc"], color=C["rf"], ls="--", lw=0.8,
                    label=f"RF baseline ({baseline['prc_auc']:.3f})")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("PRC-AUC")
    axes[2].legend(fontsize=5.5); add_panel_label(axes[2], "C", x=-0.22)

    save_fig(fig, "figure_S1_mlp_training_curves")


if __name__ == "__main__":
    main()
