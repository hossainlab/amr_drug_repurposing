"""Supplementary Figure S4 — ADMET property distributions of top-50 candidates."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import *  # noqa: F403,E402

ADMET_COLS = ["mw", "logp", "tpsa", "hbd", "hba", "rot_bonds", "rings", "qed"]
ADMET_LABS = {
    "mw": "Molecular weight (Da)", "logp": "Calculated LogP", "tpsa": "TPSA (Å²)",
    "hbd": "H-bond donors", "hba": "H-bond acceptors", "rot_bonds": "Rotatable bonds",
    "rings": "Ring count", "qed": "QED score",
}
ADMET_REFS = {"mw": 500, "logp": 5, "tpsa": 140, "hbd": 5, "hba": 10, "rot_bonds": 10}


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
        ax.hist(top50[col].dropna(), bins=16, color=C["active"], edgecolor="white",
                linewidth=0.25, alpha=0.85, zorder=3)
        if col in ADMET_REFS:
            ax.axvline(ADMET_REFS[col], color=C["inactive"], ls="--", lw=0.8,
                       label=f"Ro5/Veber = {ADMET_REFS[col]}", zorder=4)
            ax.legend(fontsize=5.0, handlelength=1.0)
        ax.set_xlabel(ADMET_LABS.get(col, col), fontsize=6.5)
        ax.set_ylabel("Count", fontsize=6.5)
        ax.tick_params(labelsize=5.5)
        add_panel_label(ax, chr(65 + idx), x=-0.18, y=1.08)

    for idx in range(len(present), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    save_fig(fig, "figure_S4_admet_properties")


if __name__ == "__main__":
    main()
