"""ECFP4 (Morgan) fingerprint featurization — the single shared implementation.

Previously copy-pasted across the training and screening scripts; centralized here
so every stage uses the identical representation (radius 2, 2048 bits).
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger

from amr_repurposing.config import ECFP_N_BITS, ECFP_RADIUS

RDLogger.DisableLog("rdApp.*")


def ecfp4_bitvect(smiles: str):
    """RDKit ExplicitBitVect for one SMILES (for Tanimoto), or None if unparseable."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, ECFP_RADIUS, nBits=ECFP_N_BITS)


def ecfp4(smiles: str) -> np.ndarray | None:
    """Dense float32 ECFP4 vector for one SMILES, or None if unparseable."""
    bv = ecfp4_bitvect(smiles)
    if bv is None:
        return None
    arr = np.zeros(ECFP_N_BITS, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def featurize_frame(df, smiles_col: str = "canonical_smiles", label_col: str | None = None):
    """Featurize a dataframe of SMILES. Returns X (and y if label_col given), dropping
    rows whose SMILES fail to parse."""
    X, y, keep = [], [], []
    for i, row in df.reset_index(drop=True).iterrows():
        fp = ecfp4(row[smiles_col])
        if fp is None:
            continue
        X.append(fp)
        keep.append(i)
        if label_col is not None:
            y.append(int(row[label_col]))
    X = np.asarray(X)
    if label_col is not None:
        return X, np.asarray(y), keep
    return X, keep
