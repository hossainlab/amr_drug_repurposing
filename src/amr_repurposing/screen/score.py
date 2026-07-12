"""
Score the non-antibiotic screening library with the trained RF+MLP ensemble.

This replaces notebook 02's in-line screen (which re-ran the OLD leaky J01/J04-only
antibiotic filter). Here the input is the canonical, leak-free
``data/screen_library_clean.csv`` produced by ``amr-build-library`` — the antibiotic
exclusion has already happened, so no molecule can leak back in at scoring time.

For every library molecule: ECFP4 fingerprint (shared featurizer) + RDKit ADMET
descriptors, then MLP probability, calibrated-RF probability, and their mean as the
ensemble score. Output columns match the notebook so the downstream cleaner
(``amr-clean-candidates``) and figures consume it unchanged.

Output:
  - data/repurposing_candidates.csv    ranked by prob_ensemble (descending)

Run:
    amr-score        # or: python -m amr_repurposing.screen.score
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from amr_repurposing.config import DATA_DIR as DATA, MODELS_DIR as MODELS, PROJECT_ROOT as ROOT, get_device
from amr_repurposing.features.fingerprints import ecfp4

LIBRARY = DATA / "screen_library_clean.csv"
MLP_CKPT = MODELS / "best_mlp.pt"
RF_CAL_PATH = MODELS / "rf_calibrated.joblib"
OUT = DATA / "repurposing_candidates.csv"


def compute_admet(smiles: str) -> dict | None:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED, rdMolDescriptors

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    tpsa = Descriptors.TPSA(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    return dict(
        mw=round(mw, 1), logp=round(logp, 2), hbd=hbd, hba=hba,
        tpsa=round(tpsa, 1), rot_bonds=rot,
        rings=rdMolDescriptors.CalcNumRings(mol),
        arom_rings=rdMolDescriptors.CalcNumAromaticRings(mol),
        qed=round(QED.qed(mol), 3),
        lipinski=int(mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10),
        veber=int(tpsa <= 140 and rot <= 10),
        drug_like=int(mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10
                      and tpsa <= 140 and rot <= 10),
    )


def _mlp_scores(X: np.ndarray, device) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader
    from amr_repurposing.models.architecture import AntibacterialMLP

    model = AntibacterialMLP(input_dim=X.shape[1]).to(device)
    ckpt = torch.load(MLP_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in DataLoader(torch.tensor(X, dtype=torch.float32), batch_size=512,
                                shuffle=False, num_workers=0):
            logits = model(batch.to(device)).cpu().numpy().flatten()
            probs.extend(1 / (1 + np.exp(-logits)))
    return np.asarray(probs)


def main() -> int:
    import joblib

    for path, hint in ((LIBRARY, "amr-build-library"), (MLP_CKPT, "amr-train"),
                       (RF_CAL_PATH, "amr-train")):
        if not path.exists():
            print(f"missing {path.name} — run `{hint}` first")
            return 1

    lib = pd.read_csv(LIBRARY)
    smi_col = "canonical_smiles" if "canonical_smiles" in lib.columns else "smiles"
    print(f"Screen library (leak-free): {len(lib):,} molecules  (from {LIBRARY.name})")

    fps, ids, names, smiles, atc, admet = [], [], [], [], [], []
    for _, row in lib.iterrows():
        fp = ecfp4(row[smi_col])
        adm = compute_admet(row[smi_col])
        if fp is None or adm is None:
            continue
        fps.append(fp)
        ids.append(row.get("molecule_chembl_id"))
        names.append(row.get("pref_name", "Unknown"))
        smiles.append(row[smi_col])
        atc.append(str(row.get("atc_classifications", "")))
        admet.append(adm)
    X = np.asarray(fps, dtype=np.float32)
    print(f"Featurized + ADMET-scored : {len(X):,}")

    device = get_device()
    mlp_prob = _mlp_scores(X, device)
    rf_cal = joblib.load(RF_CAL_PATH)
    rf_prob = rf_cal.predict_proba(X)[:, 1]
    ensemble = (mlp_prob + rf_prob) / 2

    df = pd.DataFrame({
        "chembl_id": ids,
        "drug_name": names,
        "smiles": smiles,
        "atc_codes": atc,
        "indication": "",
        "prob_mlp": np.round(mlp_prob, 4),
        "prob_rf": np.round(rf_prob, 4),
        "prob_ensemble": np.round(ensemble, 4),
    }).join(pd.DataFrame(admet)).sort_values("prob_ensemble", ascending=False).reset_index(drop=True)
    df.to_csv(OUT, index=False)

    dl = df[df["drug_like"] == 1]
    print(f"\nScored candidates -> {OUT.relative_to(ROOT)}  ({len(df):,} rows)")
    print(f"  drug-like (Ro5+Veber) : {len(dl):,}")
    print(f"  prob_ensemble >= 0.9  : {int((dl.prob_ensemble >= 0.9).sum())}")
    print(f"  prob_ensemble >= 0.8  : {int((dl.prob_ensemble >= 0.8).sum())}")
    print("\nTop 10 drug-like candidates:")
    print(dl[["drug_name", "prob_ensemble", "qed", "mw", "logp"]].head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
