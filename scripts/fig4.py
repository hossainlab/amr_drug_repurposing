"""Figure 4 — 2D structures of the top-5 non-antibiotic repurposing candidates."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import *  # noqa: F403,E402

from rdkit import Chem  # noqa: E402
from rdkit.Chem import Draw  # noqa: E402
from rdkit import RDLogger  # noqa: E402
from PIL import Image  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def main():
    _, df_druglike = load_clean_candidates()
    top5 = df_druglike.head(5).copy()

    mols, legends = [], []
    for _, row in top5.iterrows():
        m = Chem.MolFromSmiles(row["smiles"])
        if m is not None:
            mols.append(m)
            legends.append(f"{row['drug_name'].title()}\np = {row['prob_ensemble']:.3f}")

    if not mols:
        print("No valid molecules in clean candidates.")
        return

    n, W, H = len(mols), 250, 230

    grid = Draw.MolsToGridImage(mols, molsPerRow=n, subImgSize=(W, H),
                                useSVG=False, legends=legends)
    out_png = FIG_DIR / "figure_4_molecular_structures.png"
    scale = 300 / 72
    hires = grid.resize((int(grid.width * scale), int(grid.height * scale)), Image.LANCZOS)
    hires.save(str(out_png))
    print(f"  Saved: figure_4_molecular_structures.png  ({hires.width}x{hires.height} px)")

    svg = Draw.MolsToGridImage(mols, molsPerRow=n, subImgSize=(W, H),
                               useSVG=True, legends=legends)
    svg_str = svg.data if hasattr(svg, "data") else str(svg)
    (FIG_DIR / "figure_4_molecular_structures.svg").write_text(svg_str)
    print("  Saved: figure_4_molecular_structures.svg")


if __name__ == "__main__":
    main()
