"""Model-architecture schematic — deep MLP + RF ensemble (publication figure)."""
import sys
from pathlib import Path
from amr_repurposing.viz.common import *  # noqa: F403
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

# (label, n_units, kind) — kind drives colour
LAYERS = [
    ("ECFP4\n2048-bit", 2048, "input"),
    ("Linear\n1024", 1024, "hidden"),
    ("Linear\n512", 512, "hidden"),
    ("Linear\n256", 256, "hidden"),
    ("Linear\n128", 128, "hidden"),
    ("Linear\n1", 1, "out"),
]
KIND_COLOR = {"input": C["active"], "hidden": C["mlp"], "out": C["ensemble"]}


def main():
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, 10.0); ax.set_ylim(0, 6.0); ax.axis("off")

    # ── MLP funnel (heights scaled by log-units) ─────────────────────────────
    n = len(LAYERS)
    x0, bw, gap = 0.5, 0.95, 0.55
    xs = [x0 + i * (bw + gap) for i in range(n)]
    cy = 4.2
    import numpy as _np
    hmax, hmin = 2.6, 0.55
    logs = [_np.log10(u) for _, u, _ in LAYERS]
    lo, hi = min(logs), max(logs)
    for x, (lab, units, kind), lg in zip(xs, LAYERS, logs):
        bh = hmin + (hmax - hmin) * (lg - lo) / (hi - lo)
        col = KIND_COLOR[kind]
        ax.add_patch(FancyBboxPatch((x, cy - bh / 2), bw, bh,
                                    boxstyle="round,pad=0.02,rounding_size=0.05",
                                    linewidth=1.0, edgecolor=col, facecolor=col, alpha=0.85, zorder=3))
        ax.text(x + bw / 2, cy, lab, ha="center", va="center", color="white",
                fontsize=6.3, fontweight="bold", zorder=4)

    # arrows between layers
    for i in range(n - 1):
        ax.add_patch(FancyArrowPatch((xs[i] + bw, cy), (xs[i + 1], cy), arrowstyle="-|>",
                                     mutation_scale=10, lw=1.2, color=C["neutral"], zorder=2))

    # block annotation under the hidden layers
    hx0, hx1 = xs[1], xs[4] + bw
    ax.annotate("", xy=(hx1, cy - hmax / 2 - 0.25), xytext=(hx0, cy - hmax / 2 - 0.25),
                arrowprops=dict(arrowstyle="-", color=C["neutral"], lw=0.8))
    ax.text((hx0 + hx1) / 2, cy - hmax / 2 - 0.55,
            "each hidden block:  Linear → BatchNorm → ReLU → Dropout (0.3, last 0.15)",
            ha="center", va="top", fontsize=6.0, color="0.3")
    ax.text(xs[0] + bw / 2, cy + hmax / 2 + 0.35, "input", ha="center", fontsize=6.0, color="0.3")
    ax.text(xs[-1] + bw / 2, cy + hmin / 2 + 0.35, "logit", ha="center", fontsize=6.0, color="0.3")
    ax.text(0.1, 5.75, "a", fontsize=9, fontweight="bold", va="top")

    # ── Ensemble combination (panel b): MLP + RF stacked, both feeding ensemble ─
    ax.text(0.1, 2.2, "b", fontsize=9, fontweight="bold", va="top")

    def pill(x, y, w, h, txt, col, fc="white", tc=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    linewidth=1.0, edgecolor=col, facecolor=fc, zorder=3))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=6.4,
                color=tc or col, fontweight="bold", zorder=4)

    pill(0.5, 1.40, 2.3, 0.62, "Deep MLP\n→ sigmoid → p$_{MLP}$", C["mlp"])
    pill(0.5, 0.35, 2.3, 0.62, "Random Forest\n300 trees → p$_{RF}$", C["rf"])
    pill(3.7, 0.85, 2.4, 0.72, "Ensemble\np = (p$_{MLP}$+p$_{RF}$)/2", C["ensemble"],
         fc=C["ensemble"], tc="white")
    pill(6.8, 0.90, 1.7, 0.62, "Ranked\ncandidates", C["active"])

    ax.add_patch(FancyArrowPatch((2.8, 1.71), (3.7, 1.40), arrowstyle="-|>",
                                 mutation_scale=10, lw=1.2, color=C["neutral"], zorder=2))
    ax.add_patch(FancyArrowPatch((2.8, 0.66), (3.7, 1.02), arrowstyle="-|>",
                                 mutation_scale=10, lw=1.2, color=C["neutral"], zorder=2))
    ax.add_patch(FancyArrowPatch((6.1, 1.21), (6.8, 1.21), arrowstyle="-|>",
                                 mutation_scale=10, lw=1.2, color=C["neutral"], zorder=2))

    save_fig(fig, "figure_model_architecture")


if __name__ == "__main__":
    main()
