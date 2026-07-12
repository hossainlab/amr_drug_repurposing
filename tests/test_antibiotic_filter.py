"""Guards for the antibiotic filter — especially the team-review fixes.

Covers: nitroimidazole antibacterials excluded, antiprotozoal-only kept, the
J01/J04-defeats-KEEP-override fix, WHO AWaRe matching + stopwords, and the
D06B→D06BB narrowing.
"""
import pytest

from amr_repurposing.filtering import antibiotic_filter as af


# ── nitroimidazoles with a bacterial indication are excluded ─────────────────
@pytest.mark.parametrize("name", ["METRONIDAZOLE", "TINIDAZOLE", "ORNIDAZOLE",
                                  "SECNIDAZOLE", "NIMORAZOLE", "NITAZOXANIDE"])
def test_antibacterial_nitroimidazoles_flagged_by_name(name):
    assert af._name_is_antibiotic(name)


# ── antiprotozoal-only agents are KEPT (scope keeps P01) ─────────────────────
@pytest.mark.parametrize("name", ["BENZNIDAZOLE", "FEXINIDAZOLE"])
def test_antiprotozoal_only_not_flagged(name):
    assert not af._name_is_antibiotic(name)
    assert not af._is_aware_antibacterial(name)


# ── a hard J01/J04 ATC defeats the antifungal KEEP-override ──────────────────
def test_j01_defeats_keep_override_for_dual_coded_nitroimidazole():
    # tinidazole is dual-coded gynae (G01AF, in KEEP_ATC) + antibacterial (J01XD)
    assert af.is_antifungal_or_antiviral(["G01AF21", "P01AB02", "J01XD02"], "TINIDAZOLE") is False


def test_genuine_antifungal_still_kept():
    assert af.is_antifungal_or_antiviral(["J02AC01", "D01AC15"], "FLUCONAZOLE") is True


def test_topical_antiviral_d06bb_kept():
    assert af.is_antifungal_or_antiviral(["D06BB03"], "ACICLOVIR") is True


def test_topical_sulfonamide_d06ba_not_kept():
    # D06BA is antibacterial sulfonamide — must NOT be rescued by the (narrowed) keep set
    assert af.is_antifungal_or_antiviral(["D06BA06", "J01ED07"], "SULFAMERAZINE") is False


# ── WHO AWaRe cross-check ────────────────────────────────────────────────────
@pytest.mark.parametrize("name", ["SECNIDAZOLE", "NITAZOXANIDE", "VANCOMYCIN", "LINEZOLID"])
def test_aware_flags_antibacterials(name):
    assert af._is_aware_antibacterial(name)


@pytest.mark.parametrize("name", ["CILASTATIN", "PROCAINE", "CHLOROPROCAINE"])
def test_aware_stopwords_avoid_false_positives(name):
    # these share a token with an AWaRe combo name but are not antibacterials
    assert not af._is_aware_antibacterial(name)


def test_aware_category_lookup():
    assert af.aware_category("SECNIDAZOLE") == "off-label antibacterial"


# ── connectivity key ─────────────────────────────────────────────────────────
def test_connectivity_key_is_14_chars():
    ck = af.connectivity_key("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    assert ck is not None and len(ck) == 14


def test_connectivity_key_none_on_garbage():
    assert af.connectivity_key("not_a_smiles!!") is None
