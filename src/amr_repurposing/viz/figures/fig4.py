"""Figure 4 — 2D structures of the top-5 non-antibiotic repurposing candidates."""
import sys
from pathlib import Path
from amr_repurposing.viz.common import *  # noqa: F403

from rdkit import Chem  # noqa: E402
from rdkit.Chem import Draw  # noqa: E402
from rdkit.Chem.Draw import rdMolDraw2D  # noqa: E402
from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def _draw_opts():
    """Crisp, publication-weight drawing options for the molecule grid."""
    o = rdMolDraw2D.MolDrawOptions()
    o.bondLineWidth = 2
    o.legendFontSize = 26
    o.minFontSize = 16
    o.maxFontSize = 32
    o.padding = 0.10
    o.legendFraction = 0.18
    return o


def main():
    _, df_druglike = load_clean_candidates()
    top5 = df_druglike.head(5).copy()

    mols, legends = [], []
    for _, row in top5.iterrows():
        m = Chem.MolFromSmiles(row["smiles"])
        if m is not None:
            mols.append(m)
            legends.append(f"{row['drug_name'].title()}  (p = {row['prob_ensemble']:.3f})")

    if not mols:
        print("No valid molecules in clean candidates.")
        return

    # Render natively at high resolution (no upscaling) — large tiles keep
    # bonds/atoms crisp at 300 DPI instead of interpolating a small raster.
    n, W, H = len(mols), 640, 600

    grid = Draw.MolsToGridImage(mols, molsPerRow=n, subImgSize=(W, H), useSVG=False,
                                legends=legends, drawOptions=_draw_opts())
    out_png = FIG_DIR / "figure_4_molecular_structures.png"
    grid.save(str(out_png))
    print(f"  Saved: figure_4_molecular_structures.png  ({grid.width}x{grid.height} px)")

    svg = Draw.MolsToGridImage(mols, molsPerRow=n, subImgSize=(W, H), useSVG=True,
                               legends=legends, drawOptions=_draw_opts())
    svg_str = svg.data if hasattr(svg, "data") else str(svg)
    (FIG_DIR / "figure_4_molecular_structures.svg").write_text(svg_str)
    print("  Saved: figure_4_molecular_structures.svg")


if __name__ == "__main__":
    main()
