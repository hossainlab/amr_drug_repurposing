"""Target dataset helpers + config sanity."""
from amr_repurposing import config
from amr_repurposing.screen.target_datasets import slugify


def test_slugify():
    assert slugify("Enoyl-[acyl-carrier-protein] reductase [NADH]") == \
        "enoyl_acyl_carrier_protein_reductase_nadh"
    assert slugify("DNA gyrase") == "dna_gyrase"
    assert slugify("Beta-lactamase TEM") == "beta_lactamase_tem"


def test_config_paths_are_absolute():
    for p in (config.DATA_DIR, config.MODELS_DIR, config.FIGURES_DIR,
              config.REFERENCE_DIR, config.TARGETS_DIR, config.TARGET_MODELS_DIR):
        assert p.is_absolute()
    # figures live under reports/, models separate from data
    assert config.FIGURES_DIR.parent.name == "reports"
    assert config.SEED == 42
