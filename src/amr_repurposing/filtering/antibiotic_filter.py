"""
Robust antibiotic exclusion for the FDA-approved repurposing library.

Why this exists
---------------
The original screen leaked antibiotics (e.g. MOXALACTAM DISODIUM, QUINUPRISTIN)
into the "FDA-approved non-antibiotic" candidate set. Three independent failures
let them through:

  1. ATC codes in ChEMBL live on the PARENT molecule only. Salt / solvate forms
     (e.g. "MOXALACTAM DISODIUM") carry an empty atc_classifications field, so a
     per-row "exclude J01/J04" check misses them. Because exclusion ran *before*
     salt deduplication, the salt form survived and then became the surviving
     representative of that structure.

  2. indication_class was absent from the cached approved_drugs.csv. The ChEMBL
     /molecule endpoint no longer serves that legacy field at all, so the
     indication-keyword layer can never fire. It is REPLACED here by parent-
     hierarchy ATC propagation (a strictly better antibiotic signal).

  3. The name-stem keyword list missed oxacephems ("moxalactam"/"latamoxef") and
     other non-obvious stems.

This module flags a molecule as an antibiotic if ANY of:
  - its own ATC starts with J01/J04, OR
  - its PARENT molecule's ATC starts with J01/J04 (molecule_hierarchy), OR
  - any salt/solvate sharing its connectivity (InChIKey first 14 chars) is an
    antibiotic, OR
  - its name matches an antibiotic INN stem, OR
  - its name matches the curated WHO AWaRe antibacterial reference
    (data/reference/aware_antibacterials.csv).
The parent-hierarchy + connectivity propagation are what close leak (1), the one
"MOXALACTAM DISODIUM" exploited.

WHO AWaRe layer (added after team review)
-----------------------------------------
Some drugs used clinically for BACTERIAL disease are coded under non-J ATC
classes and so slipped the J01/J04 + stem checks — notably the nitroimidazoles
SECNIDAZOLE (P01AB07) and FEXINIDAZOLE (P01CA03), and NITAZOXANIDE (P01AX11,
off-label antibacterial). Two changes close this:
  - the nitroimidazole class stem "nidazole" (+ "nitazoxanide") is added to
    NAME_KW, catching secnidazole/ornidazole/fexinidazole/tinidazole in one rule;
  - names are cross-checked against the WHO AWaRe 2023 antibacterial list
    (Access/Watch/Reserve) so removals carry a citable justification.
DECISION: nitroimidazoles are dual antibacterial/antiprotozoal. The default scope
keeps antiprotozoals (P01), but per team review + AWaRe this class counts as
ANTIBACTERIAL and is excluded — a deliberate exception to the "keep P01" rule.
The antifungal/antiviral KEEP-override still takes precedence over every layer.

Usage
-----
    from antibiotic_filter import build_antibiotic_index, is_antibiotic_smiles

    idx = build_antibiotic_index(approved_drugs_df)
    leaked = is_antibiotic_smiles(smiles, name, idx)
"""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import inchi
from rdkit import RDLogger

from amr_repurposing.config import REFERENCE_DIR

RDLogger.DisableLog("rdApp.*")  # silence rdkit parse noise

# WHO AWaRe antibacterial reference (curated, committed). Names are cross-checked
# against candidate names to give each removal a citable AWaRe category.
AWARE_CSV = REFERENCE_DIR / "aware_antibacterials.csv"
# qualifier/combo tokens that are not themselves an antibacterial root — never
# match on these. "procaine"/"benzathine" are salt formers of penicillins (also a
# local anaesthetic); "cilastatin" is imipenem's renal-DHP inhibitor, not an
# antibacterial on its own.
AWARE_STOPWORDS = {
    "intravenous", "medocaril", "fosamil", "acid", "sodium", "and",
    "procaine", "benzathine", "cilastatin",
}

# ── ATC prefixes that mark ANTIBACTERIAL / ANTISEPTIC drugs ───────────────────
# Scope = "non-antibiotic" means antibacterials + antiseptics excluded; antivirals
# (J05), antifungals (J02/D01/G01AF/A07AC), and antiprotozoals (P01) are KEPT.
# Antibacterials live in many ATC classes, not just systemic J01 — also topical,
# intestinal, ophthalmic/otic, and antiseptic groups. Prefixes are pinned at the
# subgroup level so antifungal siblings (e.g. G01AF, A07AC) are NOT swept in.
ANTIBIOTIC_ATC = (
    "J01",                                  # antibacterials for systemic use
    "J04",                                  # antimycobacterials (anti-TB/leprosy)
    "A07AA", "A07AB", "A07AX",              # intestinal antibiotics / sulfonamides / other
    "D06A",                                 # antibiotics for topical use
    "D08A",                                 # antiseptics & disinfectants
    "D09AA",                                # medicated dressings with antiinfectives
    "D10AF",                                # anti-infectives for acne
    "G01AA", "G01AB", "G01AC", "G01AX",    # gynaecological antibacterials/antiseptics
    "R02AB",                                # throat antibiotics
    "S01AA", "S01AB", "S01AX",             # ophthalmic antiinfectives (antibacterial)
    "S02AA",                                # otic antiinfectives
    "S03AA",                                # ophthalmological/otological antiinfectives
)

# ── indication_class keywords (only fires if the column is present) ───────────
INDICATION_KW = [
    "antibiotic", "antibacterial", "antimicrobial", "beta-lactam",
    "penicillin", "cephalosporin", "macrolide", "quinolone",
    "aminoglycoside", "antitubercular", "antimycobacterial",
]

# ── INN name stems + explicit compounds (belt-and-suspenders) ────────────────
# NOTE: 'moxalactam'/'latamoxef' (oxacephem) and '-pristin' (streptogramin)
# added — these were the specific compounds that leaked before.
NAME_KW = [
    # beta-lactam stems
    "floxacin", "oxacin", "cillin", "cyclin", "mycin", "penem", "bactam",
    "cef", "ceph", "carbef", "carbaceph", "loracarbef",
    "moxalactam", "latamoxef", "oxacephem",          # oxacephems (leaked before)
    # sulfonamide / antifolate antibacterials — listed explicitly, NOT as a bare
    # "sulfa" stem, because "sulfa" matches the "SULFATE" counterion of unrelated
    # salts (morphine sulfate, barium sulfate, ...). Systemic sulfonamides also
    # carry J01E ATC, so the connectivity layer catches most of them anyway.
    "sulfadiazine", "sulfamethoxazole", "sulfadoxine", "sulfisoxazole",
    "sulfadimethoxine", "sulfamerazine", "trimethoprim",
    # anti-TB
    "isoniazid", "rifamp", "rifabutin", "ethambutol",
    "pretomanid", "delamanid", "bedaquiline",
    # nitroimidazoles WITH a bacterial indication (anaerobes / BV / H. pylori) —
    # listed explicitly, NOT via a bare "nidazole" stem, because that stem also
    # hits antiprotozoal-only agents (benznidazole = Chagas, fexinidazole =
    # sleeping sickness) that scope KEEPS. metronidazole is already listed below.
    # "nitazoxanide" is off-label antibacterial (C. difficile / H. pylori).
    "tinidazole", "ornidazole", "secnidazole", "nimorazole", "nitazoxanide",
    # other systemic antibacterials
    "metronidazole", "vancomycin", "linezolid", "tedizolid", "daptomycin",
    "fosfomycin", "colistin", "polymyxin", "chloramphenicol", "nitrofurantoin",
    "clindamycin", "azithromycin", "clarithromycin", "erythromycin",
    "tobramycin", "gentamicin", "amikacin", "plazomicin", "streptomycin",
    "neomycin", "kanamycin", "netilmicin", "spectinomycin",
    "tetracycline", "doxycycline", "minocycline", "tigecycline", "omadacycline",
    "meropenem", "imipenem", "ertapenem", "aztreonam",
    "clavulan",                                       # clavulanic acid (β-lactam inhibitor)
    "pristin", "quinupristin", "dalfopristin",        # streptogramins (leaked before)
    "oritavancin", "dalbavancin", "telavancin",
    "lefamulin", "fidaxomicin", "retapamulin", "mupirocin", "bacitracin",
    "rifaximin", "nalidixic", "novobiocin", "fusidic",
    # nitrofuran antibacterials (furazolidone slipped through J01-only filtering)
    "furazolidone", "nitrofural", "nitrofurazone", "nifuroxazide", "furaltadone",
    # antibacterial antiseptics / biocides (topical, ophthalmic, oral)
    "clioquinol", "chlorquinaldol", "polihexanide", "polyhexanide",
    "chlorhexidine", "hexamidine", "octenidine", "cetrimide", "cetrimonium",
    "cetylpyridinium", "benzalkonium", "dequalinium", "triclosan",
    "hexachlorophene", "chloroxylenol", "hexetidine", "taurolidine",
    # antiseptic dyes (gentian/crystal violet, acridines)
    "rosanilin", "gentian violet", "methylrosanil", "acriflavin",
    "proflavin", "aminacrine", "ethacridine",
]


# ── Antifungal / antiviral KEEP-override ──────────────────────────────────────
# Scope keeps antifungals & antivirals. Some are mis-shelved under antibacterial
# ATC subgroups (e.g. amphotericin A07AA07, ciclopirox G01AX12, natamycin S01AA10),
# so without this override the broad ATC sweep would wrongly drop them. A drug
# matching an antifungal/antiviral ATC OR name stem is NEVER flagged as antibiotic.
# NOTE: deliberately NOT including antiprotozoal P01 — clioquinol is dual-coded
# antiprotozoal + antibacterial antiseptic and must stay excluded.
# NB: D06BB (topical antivirals: aciclovir/penciclovir) is kept, but NOT the whole
# D06B group — D06BA is topical *sulfonamides* (antibacterial) and D06BX is "other"
# (e.g. metronidazole), which must stay excluded.
KEEP_ATC = ("J02", "D01", "A07AC", "G01AF", "G01AG", "J05", "D06BB")
KEEP_NAME = [
    # azole / polyene / allylamine / other antifungals
    "conazole", "amphotericin", "nystatin", "natamycin", "ciclopirox",
    "terbinafine", "naftifine", "butenafine", "griseofulvin", "flucytosine",
    "tolnaftate", "undecylenic", "tavaborole", "luliconazole", "efinaconazole",
    "tioconazole", "amorolfine", "caspofungin", "micafungin", "anidulafungin",
    # antivirals (subset that may carry odd ATC)
    "ciclovir", "covir", "navir", "buvir", "asvir", "tegravir", "vudine",
    "oseltamivir", "zanamivir", "baloxavir",
]


def is_antifungal_or_antiviral(atc_list: list[str], name: str) -> bool:
    """KEEP-override: True if clearly antifungal/antiviral (never an antibiotic).

    A hard systemic-antibacterial ATC (J01) or antimycobacterial ATC (J04) is
    unambiguous — nothing genuinely antifungal/antiviral is coded there — so it
    defeats the override. This stops nitroimidazoles that are dual-coded gynae
    (G01AF, in KEEP_ATC) *and* antibacterial (J01XD), e.g. TINIDAZOLE/ORNIDAZOLE,
    from being wrongly rescued as antifungals."""
    if any(c.startswith(("J01", "J04")) for c in atc_list):
        return False
    if any(c.startswith(KEEP_ATC) for c in atc_list):
        return True
    n = str(name).lower()
    return any(kw in n for kw in KEEP_NAME)


def parse_atc(val) -> list[str]:
    """Parse the stringified-list atc_classifications field from ChEMBL CSV."""
    if pd.isna(val) or val in ("", "[]"):
        return []
    try:
        parsed = ast.literal_eval(str(val)) if isinstance(val, str) else list(val)
        return [str(x) for x in parsed]
    except Exception:
        return []


def parent_chembl_id(hierarchy_val) -> str | None:
    """Pull parent_chembl_id out of the stringified molecule_hierarchy dict."""
    if hierarchy_val is None or (isinstance(hierarchy_val, float)):
        return None
    try:
        d = (ast.literal_eval(str(hierarchy_val))
             if isinstance(hierarchy_val, str) else dict(hierarchy_val))
        return d.get("parent_chembl_id")
    except Exception:
        return None


def _atc_is_antibiotic(codes: list[str]) -> bool:
    return any(c.startswith(ANTIBIOTIC_ATC) for c in codes)


def connectivity_key(smiles: str) -> str | None:
    """InChIKey first block (14 chars) = connectivity, ignores salt/charge/stereo."""
    if not isinstance(smiles, str) or len(smiles) < 5:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        key = inchi.MolToInchiKey(mol)
    except Exception:
        return None
    return key.split("-")[0] if key else None


def _name_is_antibiotic(name: str) -> bool:
    n = str(name).lower()
    return any(kw in n for kw in NAME_KW)


# ── WHO AWaRe antibacterial cross-check ──────────────────────────────────────
_AWARE_TOKENS: set[str] | None = None      # significant drug-root tokens (len>=5)
_AWARE_CATEGORY: dict[str, str] = {}        # token -> AWaRe category (for audit)


def load_aware_set(path: Path = AWARE_CSV) -> set[str]:
    """Load the curated WHO AWaRe antibacterial reference into a token set.

    Each inn_name is split on '/' and whitespace; component words of length >= 5
    (minus qualifier stopwords) become substring-match tokens. All tokens are
    genuine antibacterial INN roots, so a substring hit reliably marks an
    antibacterial. Cached on the module. Returns an empty set if the file is
    missing (the other filter layers still apply)."""
    global _AWARE_TOKENS
    if _AWARE_TOKENS is not None:
        return _AWARE_TOKENS
    tokens: set[str] = set()
    if path.exists():
        ref = pd.read_csv(path)
        for _, row in ref.iterrows():
            cat = str(row.get("aware_category", ""))
            raw = str(row.get("inn_name", "")).lower().replace("/", " ").replace("-", " ")
            for w in raw.split():
                if len(w) >= 5 and w not in AWARE_STOPWORDS:
                    tokens.add(w)
                    _AWARE_CATEGORY.setdefault(w, cat)
    _AWARE_TOKENS = tokens
    return tokens


def _is_aware_antibacterial(name: str) -> bool:
    n = str(name).lower()
    return any(tok in n for tok in load_aware_set())


def aware_category(name: str) -> str | None:
    """AWaRe category of the first matching token (for the removal audit)."""
    n = str(name).lower()
    for tok in load_aware_set():
        if tok in n:
            return _AWARE_CATEGORY.get(tok)
    return None


def _indication_is_antibiotic(indication) -> bool:
    if indication is None or (isinstance(indication, float)):
        return False
    s = str(indication).lower()
    return any(kw in s for kw in INDICATION_KW)


def build_antibiotic_index(df: pd.DataFrame) -> dict:
    """
    Scan the approved-drug table once and return an index used to flag antibiotics.

    A row is an antibiotic if its own ATC is J01/J04, its PARENT's ATC is J01/J04
    (via molecule_hierarchy), its indication says so (when the column exists), or
    its name matches an INN stem. The connectivity (InChIKey first block) of every
    such row is collected so salt/solvate forms anywhere can be flagged.

    Returns dict with:
      'connectivity'  : set of InChIKey-connectivity blocks of ANY antibiotic form
      'abx_chembl_ids': set of molecule_chembl_id flagged as antibiotic
      'has_indication': whether the indication_class column was usable
      'used_parent'   : whether molecule_hierarchy parent ATC propagation ran
    """
    smiles_col = "canonical_smiles" if "canonical_smiles" in df.columns else None
    if smiles_col is None:
        raise ValueError("approved drugs df has no canonical_smiles column")

    has_ind = "indication_class" in df.columns and df["indication_class"].notna().any()
    has_id = "molecule_chembl_id" in df.columns
    has_hier = "molecule_hierarchy" in df.columns

    # 1) map every molecule_chembl_id -> its own ATC list (for parent lookup)
    id_to_atc: dict[str, list[str]] = {}
    if has_id:
        for _, row in df.iterrows():
            id_to_atc[str(row["molecule_chembl_id"])] = parse_atc(
                row.get("atc_classifications"))

    abx_conn: set[str] = set()
    abx_ids: set[str] = set()
    used_parent = False
    for _, row in df.iterrows():
        atc = parse_atc(row.get("atc_classifications"))
        name = row.get("pref_name", "")

        # KEEP-override wins: antifungals/antivirals are never antibiotics
        if is_antifungal_or_antiviral(atc, name):
            continue

        by_atc = _atc_is_antibiotic(atc)

        by_parent = False
        if not by_atc and has_hier:
            pid = parent_chembl_id(row.get("molecule_hierarchy"))
            if pid and pid in id_to_atc and _atc_is_antibiotic(id_to_atc[pid]):
                by_parent = True
                used_parent = True

        by_name = _name_is_antibiotic(name)
        by_aware = _is_aware_antibacterial(name)
        by_ind = has_ind and _indication_is_antibiotic(row.get("indication_class"))

        if by_atc or by_parent or by_name or by_aware or by_ind:
            ck = connectivity_key(row[smiles_col])
            if ck:
                abx_conn.add(ck)
            if has_id:
                abx_ids.add(str(row["molecule_chembl_id"]))
    return {
        "connectivity": abx_conn,
        "abx_chembl_ids": abx_ids,
        "has_indication": has_ind,
        "used_parent": used_parent,
    }


def is_antibiotic(smiles: str, name: str, index: dict) -> bool:
    """True if this molecule is an antibiotic by connectivity OR name."""
    # KEEP-override by name (no ATC available at candidate level)
    if is_antifungal_or_antiviral([], name):
        return False
    if _name_is_antibiotic(name) or _is_aware_antibacterial(name):
        return True
    ck = connectivity_key(smiles)
    return ck is not None and ck in index["connectivity"]
