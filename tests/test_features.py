"""ECFP4 featurization contract."""
import numpy as np

from amr_repurposing.config import ECFP_N_BITS
from amr_repurposing.features import ecfp4, ecfp4_bitvect


def test_ecfp4_shape_and_dtype():
    fp = ecfp4("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    assert fp is not None
    assert fp.shape == (ECFP_N_BITS,)
    assert fp.dtype == np.float32
    assert fp.max() == 1 and fp.min() == 0


def test_ecfp4_none_on_invalid():
    assert ecfp4("not_a_smiles!!") is None
    assert ecfp4_bitvect("not_a_smiles!!") is None


def test_ecfp4_deterministic():
    a = ecfp4("c1ccccc1")
    b = ecfp4("c1ccccc1")
    assert np.array_equal(a, b)
