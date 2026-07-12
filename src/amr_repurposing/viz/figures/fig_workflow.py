"""Workflow schematic — end-to-end repurposing pipeline (publication figure)."""
import sys
from pathlib import Path
from amr_repurposing.viz.common import *  # noqa: F403
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

# Four pipeline phases: (header, header-colour, body lines)
PHASES = [
    ("1  Data", C["active"], [
        "ChEMBL REST API",
        "8 pathogens (ESKAPE + Mtb)",
        "71,623 bioactivity records",
        "MIC / IC50 / MBC / % inhib.",
    ]),
    ("2  Featurise + split", C["mlp"], [
        "Clinical-breakpoint labels",
        "48,153 molecules (43% act.)",
        "ECFP4 2048-bit fingerprints",
        "Molecule-level 80/20 split",
    ]),
    ("3  Model", C["rf"], [
        "Random Forest (300 trees)",
        "Deep MLP (ECFP4)",
        "Ensemble (mean prob.)",
        "ChemBERTa (suppl.)",
    ]),
    ("4  Screen + interpret", C["ensemble"], [
        "2,864 non-antibiotic drugs",
        "ATC + InChIKey + INN filter",
        "Rank by ensemble prob.",
        "SHAP → shortlist",
    ]),
]


def _box(ax, x, y, w, h, header, color, lines):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.02",
                                linewidth=1.0, edgecolor=color, facecolor="white", zorder=3))
    # coloured header strip
    ax.add_patch(FancyBboxPatch((x, y + h - 0.13), w, 0.13,
                                boxstyle="round,pad=0.010,rounding_size=0.02",
                                linewidth=0, facecolor=color, zorder=4))
    ax.text(x + w / 2, y + h - 0.065, header, ha="center", va="center", color="white",
            fontsize=7.0, fontweight="bold", zorder=5)
    for i, ln in enumerate(lines):
        ax.text(x + 0.03, y + h - 0.225 - i * 0.10, f"• {ln}", ha="left", va="center",
                fontsize=5.7, zorder=5)


def main():
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.set_xlim(0, 4.0); ax.set_ylim(0, 1.0); ax.axis("off")

    w, h, gap = 0.925, 0.86, 0.10
    xs = [0.012 + i * (w + gap) for i in range(4)]
    y = 0.07
    for x, (header, color, lines) in zip(xs, PHASES):
        _box(ax, x, y, w, h, header, color, lines)

    # arrows between phases
    for i in range(3):
        x0 = xs[i] + w
        x1 = xs[i + 1]
        ax.add_patch(FancyArrowPatch((x0 + 0.004, y + h / 2), (x1 - 0.004, y + h / 2),
                                     arrowstyle="-|>", mutation_scale=11, lw=1.4,
                                     color=C["neutral"], zorder=2))

    save_fig(fig, "figure_workflow")


if __name__ == "__main__":
    main()
