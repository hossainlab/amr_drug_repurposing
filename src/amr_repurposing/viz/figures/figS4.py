"""Supplementary Figure S4 — ADMET property distributions of top-50 candidates."""
import sys
from pathlib import Path
from amr_repurposing.viz.common import *  # noqa: F403

ADMET_COLS = ["mw", "logp", "tpsa", "hbd", "hba", "rot_bonds", "rings", "qed"]
ADMET_LABS = {
    "mw": "Molecular weight (Da)", "logp": "Calculated LogP", "tpsa": "TPSA (Å²)",
    "hbd": "H-bond donors", "hba": "H-bond acceptors", "rot_bonds": "Rotatable bonds",
    "rings": "Ring count", "qed": "QED score",
}
ADMET_REFS = {"mw": 500, "logp": 5, "tpsa": 140, "hbd": 5, "hba": 10, "rot_bonds": 10}
# Discrete integer-valued counts — bin on whole numbers, not 16 fractional bins.
ADMET_DISCRETE = {"hbd", "hba", "rot_bonds", "rings"}


def main():
    _, df_druglike = load_clean_candidates()
    present = [c for c in ADMET_COLS if c in df_druglike.columns]
    if len(present) < 4:
        print(f"  ADMET columns missing ({present}).")
        return

    top50 = df_druglike.head(50)
    n_col = 4
    n_row = int(np.ceil(len(present) / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(7.0, 2.4 * n_row),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.45})
    axes_flat = axes.flatten()

    for idx, col in enumerate(present):
        ax = axes_flat[idx]
        vals = top50[col].dropna()
        if col in ADMET_DISCRETE:
            lo, hi = int(np.floor(vals.min())), int(np.ceil(vals.max()))
            bins = np.arange(lo - 0.5, hi + 1.5, 1.0)        # one bar per integer
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        else:
            bins = 16
        ax.hist(vals, bins=bins, color=C["active"], edgecolor="white",
                linewidth=0.4, alpha=0.85, zorder=3)
        if col in ADMET_REFS:
            ref = ADMET_REFS[col]
            ax.axvline(ref, color=C["inactive"], ls="--", lw=0.8, zorder=4)
            # Inline label above the plot — same meaning every panel, never on the bars.
            xmax = ax.get_xlim()[1]
            ha = "right" if ref >= 0.92 * xmax else "center"
            ax.annotate(f"Ro5/Veber ≤ {ref}", xy=(ref, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(0, 2), textcoords="offset points", ha=ha, va="bottom",
                        fontsize=5.0, color=C["inactive"])
        ax.set_xlabel(ADMET_LABS.get(col, col), fontsize=6.5)
        ax.set_ylabel("Count", fontsize=6.5)
        ax.tick_params(labelsize=5.5)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        add_panel_label(ax, chr(65 + idx), x=-0.18, y=1.08)

    for idx in range(len(present), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    save_fig(fig, "figure_S4_admet_properties")


if __name__ == "__main__":
    main()
