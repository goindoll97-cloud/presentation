# -*- coding: utf-8 -*-
"""Study 2 / Step 04A (V4.2): automatically build a larger STRICT broad-salt candidate pool.

V4.2 changes (vs V4.1)
----------------------
- Route D: structure-first discovery. PubChem "mixture/component" links
  (NCBI E-utilities elink, linkname=pccompound_pccompound_mixture) return every
  PubChem compound that contains the parent as a component, independent of naming.
- New SOURCE evidence: PubChem's own parent assignment (PUG-REST cids_type=parent).
  A candidate whose PubChem parent CID equals the reference parent CID is
  source-supported even when its name does not contain the parent name
  (e.g. IUPAC "sodium;...octanoate" titles). RDKit still never creates labels.
- Parent-name evidence also accepts anion forms ("...ic acid" -> "...ate",
  "...ous acid" -> "...ite").
- New pre-freeze SALT-FORM QC gate (downgrade-only): every non-parent component
  must be explainable as a counterion (charged, metal, or complementary
  acid/base partner) or a small neutral solvent; at least one counterion is
  required. Neutral hydrates/co-crystals and multi-drug mixtures go to REVIEW.
- Bug fixes: word-search cap now honours STUDY2_WORD_SEARCH_MAX_CIDS; unified
  synonym window; name queries keep all returned CIDs; all CAS numbers are
  recorded; route priority in balanced selection now actually matches routes;
  removed unused explicit_positive_from_cid(); covalently drawn metals are now
  disconnected before parent-fragment matching.
- Single-parent mode:  python <this file> --parent-cas 335-67-1


NO Claude API is used in this step.

What this script does
---------------------
1. Keeps the existing broad-salt rules/cases as the CORE benchmark.
2. If a Study-1 scope table is available, discovers additional broad-salt parent
   chemicals ONLY from strict rows such as "A and its salts" / "A와 그 염류".
3. Resolves parent identities through exact-CAS-verified PubChem records.
4. Builds candidate substances from PubChem using:
   - explicit salt-name queries (positive candidates),
   - hydrate/solvate/stereochemical-like name queries (harder positives),
   - PubChem 2-D similarity search (hard negative candidates),
   - cross-parent compounds/salts (easy/moderate negatives).
5. Writes a fully auditable candidate pool and a strict auto-selected expansion.

Ground-truth safeguard
----------------------
Reference labels originate from source-supported PubChem identity evidence and
the frozen regulatory parent-and-salts scope. RDKit is used here ONLY as a
pre-freeze structural-consistency QC gate:
- it never flips a MATCH into NO_MATCH or vice versa,
- a source label that conflicts with the parent structure is downgraded to REVIEW,
- only source-supported + structure-consistent rows may enter the automatic set.
This prevents lexical false positives such as hydroxy derivatives, analog salts,
or similarly named compounds from contaminating the benchmark.

Outputs
-------
data/broad_salt_rules_AUTO_EXPANDED.csv
data/broad_salt_validation_cases_AUTO_EXPANDED.csv
intermediate/04A_candidate_pool_all.csv
intermediate/04A_candidate_review_queue.csv
intermediate/04A_structure_qc_audit.csv
intermediate/04A_autobuild_qc.csv
intermediate/04A_autobuild_config.json
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import random
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import pandas as pd

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
except Exception as exc:
    raise ImportError(
        "04A V4 requires RDKit for pre-freeze structural-consistency QC. "
        "Run this script in the same environment used for the deterministic comparator."
    ) from exc

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ENGINE_DATA = ROOT / "engine" / "data"
INTER = ROOT / "intermediate"
CACHE = INTER / "pubchem_candidate_cache"
for p in (DATA, INTER, CACHE):
    p.mkdir(parents=True, exist_ok=True)

GITHUB_RAW_BASE = os.getenv(
    "STUDY2_GITHUB_RAW_BASE",
    "https://raw.githubusercontent.com/goindoll97-cloud/presentation/main/data",
).rstrip("/")
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
USER_AGENT = "study2-identity-benchmark/4.2-structure-first"
PAUSE = float(os.getenv("STUDY2_PUBCHEM_PAUSE_SEC", "0.40"))
# Server-expensive calls (word search, fast* structure search, multi-CID batches,
# E-utilities elink) get a wider gap; PubChem answers these with 503 +
# Retry-After when they arrive back-to-back with other traffic.
HEAVY_PAUSE = float(os.getenv("STUDY2_PUBCHEM_HEAVY_PAUSE_SEC", "2.0"))
PUBCHEM_RETRIES = int(os.getenv("STUDY2_PUBCHEM_RETRIES", "8"))
PUBCHEM_BACKOFF_MAX = float(os.getenv("STUDY2_PUBCHEM_BACKOFF_MAX_SEC", "60"))
_LAST_PUBCHEM_REQUEST_AT = 0.0
TARGET_PARENTS = int(os.getenv("STUDY2_TARGET_PARENTS", "20"))
TARGET_CASES_PER_PARENT = int(os.getenv("STUDY2_TARGET_CASES_PER_PARENT", "30"))
MAX_SIMILAR_HITS = int(os.getenv("STUDY2_MAX_SIMILAR_HITS", "18"))
SIMILARITY_THRESHOLD = int(os.getenv("STUDY2_SIMILARITY_THRESHOLD", "80"))
# Expanded benchmark construction controls. These operate BEFORE any Claude call.
TARGET_MIN_CASES = int(os.getenv("STUDY2_TARGET_MIN_CASES", "180"))
TARGET_MAX_CASES = int(os.getenv("STUDY2_TARGET_MAX_CASES", "240"))
DISCOVERY_PARENT_POOL = int(os.getenv("STUDY2_DISCOVERY_PARENT_POOL", "120"))
MAX_SELECTED_RULES = int(os.getenv("STUDY2_MAX_SELECTED_RULES", "35"))
MIN_STRICT_POS_PER_NEW_PARENT = int(os.getenv("STUDY2_MIN_STRICT_POS_PER_NEW_PARENT", "2"))
WORD_SEARCH_MAX_CIDS = int(os.getenv("STUDY2_WORD_SEARCH_MAX_CIDS", "80"))
MIXTURE_MAX_CIDS = int(os.getenv("STUDY2_MIXTURE_MAX_CIDS", "300"))
SYNONYM_WINDOW = int(os.getenv("STUDY2_SYNONYM_WINDOW", "220"))
SOLVENT_MAX_HEAVY = int(os.getenv("STUDY2_SOLVENT_MAX_HEAVY_ATOMS", "6"))
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_TOOL = os.getenv("STUDY2_NCBI_TOOL", "study2_identity_benchmark")
NCBI_EMAIL = os.getenv("STUDY2_NCBI_EMAIL", "")
TARGET_POSITIVE_FRACTION = float(os.getenv("STUDY2_TARGET_POSITIVE_FRACTION", "0.50"))

CAS_RE = re.compile(r"\b(\d{2,7}-\d{2}-\d)\b")
BROAD_SALT_PATTERNS = [
    re.compile(r"(?:와|과|및)\s*그\s*염류", re.I),
    re.compile(r"(?:와|과|및)\s*그\s*염\b", re.I),
    re.compile(r"\band\s+(?:all\s+)?(?:its|their)\s+salts\b", re.I),
]
# Deliberately NOT accepted as broad-salt scope: a specific sodium/potassium salt,
# "salt with ...", 1:1 salts, or generic appearances of the word salt/염류.
# Those are individual substances, not necessarily parent + all-salts regulatory scopes.

COMMON_SALT_QUERIES = [
    ("sodium", "COMMON_SALT_POSITIVE", "EASY"),
    ("potassium", "COMMON_SALT_POSITIVE", "EASY"),
    ("hydrochloride", "COMMON_SALT_POSITIVE", "EASY"),
    ("hydrobromide", "COMMON_SALT_POSITIVE", "EASY"),
    ("calcium", "UNCOMMON_SALT_POSITIVE", "MODERATE"),
    ("magnesium", "UNCOMMON_SALT_POSITIVE", "MODERATE"),
    ("sulfate", "UNCOMMON_SALT_POSITIVE", "MODERATE"),
    ("nitrate", "UNCOMMON_SALT_POSITIVE", "MODERATE"),
    ("phosphate", "UNCOMMON_SALT_POSITIVE", "MODERATE"),
    ("acetate", "UNCOMMON_SALT_POSITIVE", "MODERATE"),
    ("citrate", "UNCOMMON_SALT_POSITIVE", "MODERATE"),
    ("tartrate", "UNCOMMON_SALT_POSITIVE", "MODERATE"),
    ("maleate", "UNCOMMON_SALT_POSITIVE", "MODERATE"),
    ("fumarate", "UNCOMMON_SALT_POSITIVE", "MODERATE"),
    ("mesylate", "UNCOMMON_SALT_POSITIVE", "MODERATE"),
    ("besylate", "UNCOMMON_SALT_POSITIVE", "MODERATE"),
    ("tosylate", "UNCOMMON_SALT_POSITIVE", "MODERATE"),
]
COMPLEX_QUERIES = [
    ("monohydrate", "COMPLEX_SALT_OR_SOLVATE_POSITIVE", "HARD"),
    ("dihydrate", "COMPLEX_SALT_OR_SOLVATE_POSITIVE", "HARD"),
    ("hydrate", "COMPLEX_SALT_OR_SOLVATE_POSITIVE", "HARD"),
    ("ethanol solvate", "COMPLEX_SALT_OR_SOLVATE_POSITIVE", "HARD"),
    ("isopropanol solvate", "COMPLEX_SALT_OR_SOLVATE_POSITIVE", "HARD"),
]


# Explicit lexical evidence accepted for source-supported salt membership.
SALT_ALIASES = {
    "sodium": ["sodium"], "potassium": ["potassium"],
    "lithium": ["lithium"], "ammonium": ["ammonium"],
    "calcium": ["calcium"], "magnesium": ["magnesium"],
    "hydrochloride": ["hydrochloride"], "hydrobromide": ["hydrobromide"],
    "chloride": ["chloride"], "bromide": ["bromide"], "iodide": ["iodide"],
    "sulfate": ["sulfate", "sulphate"], "bisulfate": ["bisulfate", "hydrogen sulfate"],
    "nitrate": ["nitrate"], "phosphate": ["phosphate"],
    "acetate": ["acetate"], "formate": ["formate"],
    "carbonate": ["carbonate"], "bicarbonate": ["bicarbonate", "hydrogen carbonate"],
    "citrate": ["citrate"], "tartrate": ["tartrate"], "bitartrate": ["bitartrate", "hydrogen tartrate"],
    "maleate": ["maleate"], "fumarate": ["fumarate"],
    "succinate": ["succinate"], "oxalate": ["oxalate"],
    "lactate": ["lactate"], "benzoate": ["benzoate"], "salicylate": ["salicylate"],
    "gluconate": ["gluconate"], "perchlorate": ["perchlorate"],
    "thiocyanate": ["thiocyanate"], "picrate": ["picrate"],
    "mesylate": ["mesylate", "methanesulfonate"],
    "besylate": ["besylate", "benzenesulfonate"],
    "tosylate": ["tosylate", "toluenesulfonate", "p toluenesulfonate"],
}
COMPLEX_FORM_ALIASES = [
    "hydrate", "monohydrate", "dihydrate", "trihydrate",
    "solvate", "ethanol", "isopropanol", "propan-2-ol", "2-propanol",
]

def clean(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    return "" if s.lower() in {"nan", "none", "null", "<na>"} else s


def norm_name(x: str) -> str:
    s = clean(x).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def valid_cas(cas: str) -> bool:
    cas = clean(cas)
    if not CAS_RE.fullmatch(cas):
        return False
    digits = cas.replace("-", "")
    check = int(digits[-1])
    body = digits[:-1][::-1]
    return sum((i + 1) * int(d) for i, d in enumerate(body)) % 10 == check


def extract_cas(values: Iterable[str]) -> list[str]:
    out = []
    for v in values:
        for m in CAS_RE.findall(clean(v)):
            if valid_cas(m) and m not in out:
                out.append(m)
    return out



# ---------------------------------------------------------------------------
# Pre-freeze structural-consistency QC
# ---------------------------------------------------------------------------
_UNCHARGER = rdMolStandardize.Uncharger()
_TAUTOMER = rdMolStandardize.TautomerEnumerator()


def _standardize_qc_mol(mol):
    """Lightweight standardization for identity-QC only.

    The automatic benchmark accepts only an isomeric exact parent-fragment match
    after cleanup/uncharge. Connectivity-only or tautomer/stereo ambiguities are
    sent to REVIEW rather than auto-labeled.
    """
    m = Chem.Mol(mol)
    try:
        m = rdMolStandardize.Cleanup(m)
    except Exception:
        pass
    try:
        m = _UNCHARGER.uncharge(m)
    except Exception:
        pass
    try:
        Chem.SanitizeMol(m)
    except Exception:
        pass
    return m


def _qc_keys(mol) -> tuple[str, str, str]:
    """Return (isomeric, non-isomeric, tautomer-nonisomeric) canonical keys."""
    m = _standardize_qc_mol(mol)
    iso = Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
    noniso = Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)
    taut_noniso = noniso
    try:
        tm = _TAUTOMER.Canonicalize(Chem.Mol(m))
        taut_noniso = Chem.MolToSmiles(tm, canonical=True, isomericSmiles=False)
    except Exception:
        pass
    return iso, noniso, taut_noniso


def structural_parent_relation(parent_smiles: str, candidate_smiles: str) -> tuple[str, str]:
    """Check whether any disconnected candidate fragment is the reference parent.

    Returns:
      EXACT_PARENT_FRAGMENT
          same standardized isomeric parent fragment; suitable for auto-positive QC.
      CONNECTIVITY_ONLY
          same connectivity/tautomeric parent but stereo/representation differs;
          REVIEW only.
      NO_PARENT_FRAGMENT
          no candidate fragment corresponds to the reference parent.
      PARSE_ERROR
          missing/unparseable structure; REVIEW only.
    """
    ps = clean(parent_smiles)
    cs = clean(candidate_smiles)
    if not ps or not cs:
        return "PARSE_ERROR", "missing_parent_or_candidate_smiles"
    pm = Chem.MolFromSmiles(ps)
    cm = Chem.MolFromSmiles(cs)
    if pm is None or cm is None:
        return "PARSE_ERROR", "rdkit_smiles_parse_failed"

    p_iso, p_noniso, p_taut = _qc_keys(pm)
    connectivity_hit = False

    # V4.2: disconnect covalently drawn metals BEFORE fragmenting; V4.1 ran
    # Cleanup per-fragment, so "[Na]OC(=O)R" never exposed the parent fragment.
    try:
        cm = rdMolStandardize.Cleanup(cm)
    except Exception:
        pass
    try:
        frags = Chem.GetMolFrags(cm, asMols=True, sanitizeFrags=True)
    except Exception:
        frags = (cm,)

    for frag in frags:
        c_iso, c_noniso, c_taut = _qc_keys(frag)
        if c_iso == p_iso:
            return "EXACT_PARENT_FRAGMENT", "standardized_isomeric_parent_fragment_present"
        if c_noniso == p_noniso or c_taut == p_taut:
            connectivity_hit = True

    if connectivity_hit:
        return "CONNECTIVITY_ONLY", "parent_connectivity_present_but_stereo_or_representation_differs"
    return "NO_PARENT_FRAGMENT", "no_standardized_parent_fragment_present"


# ---------------------------------------------------------------------------
# Salt-form QC (downgrade-only gate, applied to MATCH candidates after the
# parent-fragment check). It answers: is every other component explainable as a
# counterion or solvent, and is there ionic evidence at all?
# ---------------------------------------------------------------------------
_METALS = {
    "Li", "Na", "K", "Rb", "Cs", "Be", "Mg", "Ca", "Sr", "Ba", "Al", "Zn", "Fe", "Cu",
    "Co", "Ni", "Mn", "Cr", "Ag", "Sn", "Pb", "Bi", "Ti", "Zr", "La", "Ce", "Hg", "Cd",
}
_ACID_PATTERNS = [Chem.MolFromSmarts(s) for s in [
    "[CX3](=O)[OX2H1]",                    # carboxylic acid
    "[SX4](=O)(=O)[OX2H1]",                # sulfonic / sulfuric
    "[SX3](=O)[OX2H1]",                    # sulfinic
    "[PX4](=O)[OX2H1]",                    # phosphonic / phosphoric
    "[NX3H1](S(=O)=O)S(=O)=O",             # sulfonimide (PFAS bis-sulfonimides)
    "[F,Cl,Br,I;D0;+0]",                   # HX drawn as a lone halogen atom
    "[OX2H1]Cl(=O)(=O)=O",                 # perchloric
    "[OX2H1][NX3](=O)=O",                  # nitric (neutral drawing)
    "[OX2H1]c1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]",  # picric
]]
_BASE_PATTERNS = [Chem.MolFromSmarts(s) for s in [
    "[NX3;+0;!$(N-[C,S,P]=[O,S,N]);!$(N-a);!$(N-[N,O,S]);!$(N-C#N)]",  # aliphatic amine / NH3
    "[NX3;H2,H1;+0;$(N-a);!$(N-[C,S]=[O,S])]",                          # aniline-type
    "[nX2;H0;+0]",                                                       # pyridine-type
    "[CX3](=[NX2;+0])[NX3;+0]",                                          # amidine / guanidine
]]


def _has_any(mol, patterns) -> bool:
    return any(p is not None and mol.HasSubstructMatch(p) for p in patterns)


def salt_form_qc(parent_smiles: str, candidate_smiles: str) -> tuple[bool, str, list[str]]:
    """Return (pass, status, counterion_smiles).

    status values:
      SALT_FORM_CONSISTENT / SALT_FORM_CONSISTENT_WITH_SOLVATE -> pass
      SINGLE_COMPONENT        candidate is only the parent (e.g. stereo/isotope form)
      NO_IONIC_EVIDENCE       neutral hydrate / solvate / co-crystal of the parent
      UNEXPLAINED_COMPONENT   another component is neither counterion nor solvent
                              (e.g. multi-drug mixture)
      NO_PARENT_FRAGMENT / PARSE_ERROR
    """
    pm = Chem.MolFromSmiles(clean(parent_smiles)) if clean(parent_smiles) else None
    cm = Chem.MolFromSmiles(clean(candidate_smiles)) if clean(candidate_smiles) else None
    if pm is None or cm is None:
        return False, "PARSE_ERROR", []
    try:
        cm = rdMolStandardize.Cleanup(cm)  # includes metal disconnection; does not uncharge
    except Exception:
        pass
    _, p_noniso, p_taut = _qc_keys(pm)
    p_std = _standardize_qc_mol(pm)
    parent_acidic = _has_any(p_std, _ACID_PATTERNS)
    parent_basic = _has_any(p_std, _BASE_PATTERNS)

    try:
        frags = Chem.GetMolFrags(cm, asMols=True, sanitizeFrags=True)
    except Exception:
        return False, "PARSE_ERROR", []

    parent_frags, others = [], []
    for f in frags:
        _, n, t = _qc_keys(f)
        (parent_frags if (n == p_noniso or t == p_taut) else others).append(f)
    if not parent_frags:
        return False, "NO_PARENT_FRAGMENT", []
    if not others:
        return False, "SINGLE_COMPONENT", []

    ionic = any(a.GetFormalCharge() != 0 for f in parent_frags for a in f.GetAtoms())
    counterions, solvent, unexplained = [], False, False
    for f in others:
        charged = any(a.GetFormalCharge() != 0 for a in f.GetAtoms())
        metal = any(a.GetSymbol() in _METALS for a in f.GetAtoms())
        fs = _standardize_qc_mol(f)
        complementary = (parent_basic and _has_any(fs, _ACID_PATTERNS)) or (
            parent_acidic and _has_any(fs, _BASE_PATTERNS)
        )
        if charged or metal or complementary:
            counterions.append(Chem.MolToSmiles(f))
            ionic = True
        elif f.GetNumHeavyAtoms() <= SOLVENT_MAX_HEAVY:
            solvent = True
        else:
            unexplained = True
    if unexplained:
        return False, "UNEXPLAINED_COMPONENT", counterions
    if not ionic or not counterions:
        return False, "NO_IONIC_EVIDENCE", counterions
    return True, ("SALT_FORM_CONSISTENT_WITH_SOLVATE" if solvent else "SALT_FORM_CONSISTENT"), sorted(set(counterions))


_EASY_COUNTERIONS = {"[Na+]", "[K+]", "Cl", "[Cl-]", "Br", "[Br-]", "[NH4+]", "[Li+]"}


def source_label_passes_structure_qc(label: str, relation: str) -> tuple[bool, str]:
    """QC gate only; does not create or flip a reference label."""
    lab = clean(label).upper()
    if lab == "MATCH":
        if relation == "EXACT_PARENT_FRAGMENT":
            return True, "source_MATCH_and_exact_parent_fragment_concordant"
        if relation == "CONNECTIVITY_ONLY":
            return False, "source_MATCH_but_stereo_or_representation_requires_review"
        return False, "source_MATCH_conflicts_with_parent_structure"
    if lab == "NO_MATCH":
        if relation == "NO_PARENT_FRAGMENT":
            return True, "source_NO_MATCH_and_no_parent_fragment_concordant"
        return False, "source_NO_MATCH_conflicts_with_or_is_ambiguous_by_parent_structure"
    return False, "unsupported_reference_label"


def apply_structure_qc_row(row: dict, reference_smiles: str) -> dict:
    """Attach QC fields and downgrade conflicts to REVIEW_REQUIRED."""
    out = dict(row)
    rel, detail = structural_parent_relation(reference_smiles, out.get("smiles", ""))
    ok, reason = source_label_passes_structure_qc(out.get("reference_membership", ""), rel)
    out["structural_qc_relation"] = rel
    out["structural_qc_detail"] = detail
    out["structural_qc_pass"] = bool(ok)
    out["structural_qc_reason"] = reason
    if clean(out.get("reference_status")) == "AUTO_STRICT" and not ok:
        out["reference_status"] = "REVIEW_REQUIRED"
        out["notes"] = (clean(out.get("notes")) + f" STRUCTURE_QC: {reason}.").strip()
    return out


def apply_structure_qc_dataframe(pool_df: pd.DataFrame, rules_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same pre-freeze QC to positives, negatives, and cross-parent rows."""
    if pool_df is None or pool_df.empty:
        return pool_df
    ref_map = {
        clean(r["rule_id"]): clean(r.get("reference_smiles"))
        for _, r in rules_df.iterrows()
    }
    rows = []
    for _, r in pool_df.iterrows():
        d = r.to_dict()
        rows.append(apply_structure_qc_row(d, ref_map.get(clean(d.get("rule_id")), "")))
    return pd.DataFrame(rows)


def mark_core_duplicates(pool_df: pd.DataFrame, base_cases: pd.DataFrame) -> pd.DataFrame:
    """Exclude expansion rows already represented in the frozen CORE benchmark."""
    if pool_df is None or pool_df.empty:
        return pool_df
    out = pool_df.copy()
    base_keys = set()
    if {"rule_id", "cas"}.issubset(base_cases.columns):
        base_keys = {
            (clean(r), clean(c))
            for r, c in zip(base_cases["rule_id"], base_cases["cas"])
            if clean(r) and clean(c)
        }
    flags = [
        (clean(r), clean(c)) in base_keys
        for r, c in zip(out["rule_id"], out["cas"])
    ]
    out["duplicate_of_core"] = flags
    mask = out["duplicate_of_core"].astype(bool)
    out.loc[mask, "reference_status"] = "CORE_DUPLICATE_EXCLUDED"
    out.loc[mask, "notes"] = (
        out.loc[mask, "notes"].map(clean)
        + " Excluded from expansion because the same rule/CAS is already in CORE."
    ).str.strip()
    return out


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str).fillna("")
    return pd.read_csv(path, dtype=str).fillna("")


def download_if_missing(filename: str, candidates: list[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    target = INTER / "source_snapshots" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return target
    url = f"{GITHUB_RAW_BASE}/{filename}"
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=45) as r:
            b = r.read()
        if b:
            target.write_bytes(b)
            print(f"[SOURCE] downloaded {url}")
            return target
    except Exception as e:
        print(f"[WARN] could not download {url}: {type(e).__name__}: {e}")
    return None



def _is_heavy_request(url: str) -> bool:
    """True for request types that are expensive on the PubChem/NCBI side."""
    if url.startswith(EUTILS):
        return True
    if "name_type=word" in url or "/fast" in url:
        return True
    m = re.search(r"/compound/cid/([^/]+)/", url)
    return bool(m and "," in m.group(1))


def _pubchem_wait_before_request(url: str = "") -> None:
    """Globally pace PUG-REST calls in this single-process script.

    Default interval is 0.40 s (~2.5 req/s), intentionally below PubChem's
    published 5 req/s ceiling because each candidate can trigger several
    dependent requests and dynamic throttling can become stricter under load.
    Heavy requests (see _is_heavy_request) wait HEAVY_PAUSE instead.
    """
    global _LAST_PUBCHEM_REQUEST_AT
    pause = max(PAUSE, HEAVY_PAUSE) if _is_heavy_request(url) else PAUSE
    now = time.monotonic()
    wait = pause - (now - _LAST_PUBCHEM_REQUEST_AT)
    if wait > 0:
        time.sleep(wait)
    _LAST_PUBCHEM_REQUEST_AT = time.monotonic()


def _retry_delay(attempt: int, err: HTTPError | None = None) -> float:
    """Exponential backoff with small jitter; respects Retry-After when present."""
    retry_after = None
    if err is not None:
        try:
            ra = err.headers.get("Retry-After")
            retry_after = float(ra) if ra else None
        except Exception:
            retry_after = None
    base = min(PUBCHEM_BACKOFF_MAX, 2.0 * (2 ** attempt))
    if retry_after is not None:
        base = max(base, retry_after)
    return min(PUBCHEM_BACKOFF_MAX, base + random.uniform(0.0, 0.8))


def _short_url(url: str, limit: int = 140) -> str:
    u = url.replace(PUBCHEM, "PUG").replace(EUTILS, "EUTILS")
    return u if len(u) <= limit else u[: limit - 3] + "..."


def _http_error_detail(e: HTTPError) -> str:
    """Throttling header + first part of the error body, for retry logs."""
    parts = []
    try:
        throttle = e.headers.get("X-Throttling-Control")
        if throttle:
            parts.append(f"throttle=[{throttle}]")
        ra = e.headers.get("Retry-After")
        if ra:
            parts.append(f"retry_after={ra}")
    except Exception:
        pass
    try:
        body = e.read(400).decode("utf-8", "replace")
        body = re.sub(r"<[^>]+>", " ", body)
        body = re.sub(r"\s+", " ", body).strip()[:160]
        if body:
            parts.append(f"body={body!r}")
    except Exception:
        pass
    return "; ".join(parts)


def cached_json(url: str, retries: int = PUBCHEM_RETRIES) -> Optional[dict]:
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    fp = CACHE / f"{key}.json"
    if fp.exists():
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            pass

    last = None
    for attempt in range(max(1, retries)):
        try:
            _pubchem_wait_before_request(url)
            req = Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json",
                },
            )
            with urlopen(req, timeout=90) as r:
                raw = r.read()
                # PubChem notes that an HTML body may occasionally accompany
                # overload responses; reject non-JSON before caching.
                ctype = (r.headers.get("Content-Type") or "").lower()
                if b"<html" in raw[:200].lower() or ("json" not in ctype and raw[:1] not in {b"{", b"["}):
                    raise ValueError("non_json_pubchem_response")
                payload = json.loads(raw.decode("utf-8"))
            fp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return payload

        except HTTPError as e:
            last = e
            if e.code == 404:
                return None
            if e.code in {429, 500, 502, 503, 504}:
                delay = _retry_delay(attempt, e)
                detail = _http_error_detail(e)
                print(
                    f"[PUBCHEM RETRY {attempt+1}/{retries}] HTTP {e.code}; "
                    f"sleep {delay:.1f}s; url={_short_url(url)}"
                    + (f"\n    {detail}" if detail else "")
                )
                time.sleep(delay)
                continue
            return None

        except (URLError, TimeoutError, ValueError) as e:
            last = e
            delay = _retry_delay(attempt, None)
            print(
                f"[PUBCHEM RETRY {attempt+1}/{retries}] {type(e).__name__}: {e}; "
                f"sleep {delay:.1f}s; url={_short_url(url)}"
            )
            time.sleep(delay)

    # IMPORTANT: do not silently interpret a transient PubChem failure as
    # "compound not found".  Stop here so a partially queried candidate universe
    # is never frozen as the benchmark. Successful calls are already cached, so
    # rerunning later resumes economically from the cache.
    raise RuntimeError(
        "PubChem remained unavailable after retries. "
        "No benchmark was frozen from an incomplete query set. "
        f"Last error={last}; URL={url}"
    )


def pubchem_properties_by_name(name: str) -> Optional[dict]:
    props = "Title,SMILES,ConnectivitySMILES,InChIKey,MolecularFormula"
    url = f"{PUBCHEM}/compound/name/{quote(name, safe='')}/property/{props}/JSON"
    j = cached_json(url)
    rows = (((j or {}).get("PropertyTable") or {}).get("Properties") or [])
    return rows[0] if rows else None


def pubchem_properties_by_cid(cid: int | str) -> Optional[dict]:
    props = "Title,SMILES,ConnectivitySMILES,InChIKey,MolecularFormula"
    url = f"{PUBCHEM}/compound/cid/{cid}/property/{props}/JSON"
    j = cached_json(url)
    rows = (((j or {}).get("PropertyTable") or {}).get("Properties") or [])
    return rows[0] if rows else None


def pubchem_synonyms(cid: int | str) -> list[str]:
    url = f"{PUBCHEM}/compound/cid/{cid}/synonyms/JSON"
    j = cached_json(url)
    infos = (((j or {}).get("InformationList") or {}).get("Information") or [])
    return [clean(x) for x in (infos[0].get("Synonym", []) if infos else []) if clean(x)]


PROPS = "Title,SMILES,ConnectivitySMILES,InChIKey,MolecularFormula"


def pubchem_properties_by_name_all(name: str, max_rows: int = 5) -> list[dict]:
    """Name lookup keeping every returned record (V4.1 kept only rows[0])."""
    url = f"{PUBCHEM}/compound/name/{quote(name, safe='')}/property/{PROPS}/JSON"
    j = cached_json(url)
    rows = (((j or {}).get("PropertyTable") or {}).get("Properties") or [])
    return rows[:max_rows]


def pubchem_properties_batch(cids: list[int], chunk: int = 100) -> dict[int, dict]:
    """Batch property fetch (one request per <=100 CIDs) to keep Route D affordable."""
    out: dict[int, dict] = {}
    cids = [int(c) for c in dict.fromkeys(cids)]
    for i in range(0, len(cids), chunk):
        part = cids[i:i + chunk]
        url = f"{PUBCHEM}/compound/cid/{','.join(map(str, part))}/property/{PROPS}/JSON"
        j = cached_json(url)
        for p in (((j or {}).get("PropertyTable") or {}).get("Properties") or []):
            if p.get("CID"):
                out[int(p["CID"])] = p
    return out


def pubchem_parent_cids(cid: int | str) -> list[int]:
    """PubChem's own parent assignment for a salt/mixture record (source evidence)."""
    url = f"{PUBCHEM}/compound/cid/{cid}/cids/JSON?cids_type=parent"
    j = cached_json(url)
    ids = (((j or {}).get("IdentifierList") or {}).get("CID") or [])
    return [int(x) for x in ids if str(x).isdigit()]


def mixture_cids(parent_cid: int | str, max_records: int = MIXTURE_MAX_CIDS) -> tuple[list[int], int]:
    """Route D: every PubChem compound that contains the parent as a component.

    Uses the PubChem 'Mixtures, Components, and Neutralized Forms' relation exposed
    through NCBI E-utilities. Returns (cids_truncated, total_found).
    """
    url = (
        f"{EUTILS}/elink.fcgi?dbfrom=pccompound&db=pccompound&id={parent_cid}"
        f"&linkname=pccompound_pccompound_mixture&retmode=json&tool={quote(NCBI_TOOL)}"
        + (f"&email={quote(NCBI_EMAIL)}" if NCBI_EMAIL else "")
    )
    j = cached_json(url) or {}
    found: list[int] = []
    for ls in j.get("linksets", []) or []:
        for db in ls.get("linksetdbs", []) or []:
            if db.get("linkname") not in (None, "pccompound_pccompound_mixture"):
                continue
            for x in db.get("links", []) or []:
                v = x.get("id") if isinstance(x, dict) else x
                if str(v).isdigit() and int(v) != int(parent_cid):
                    found.append(int(v))
    found = list(dict.fromkeys(found))
    return found[:max_records], len(found)


def resolve_exact_cas(cas: str) -> Optional[dict]:
    if not valid_cas(cas):
        return None
    p = pubchem_properties_by_name(cas)
    if not p or not p.get("CID"):
        return None
    syn = pubchem_synonyms(p["CID"])
    if cas not in extract_cas(syn):
        return None
    return {**p, "synonyms": syn, "cas": cas}


def similar_cids(cid: int | str) -> list[int]:
    url = (
        f"{PUBCHEM}/compound/fastsimilarity_2d/cid/{cid}/cids/JSON"
        f"?Threshold={SIMILARITY_THRESHOLD}&MaxRecords={MAX_SIMILAR_HITS + 1}"
    )
    j = cached_json(url)
    ids = (((j or {}).get("IdentifierList") or {}).get("CID") or [])
    return [int(x) for x in ids if str(x).isdigit() and int(x) != int(cid)][:MAX_SIMILAR_HITS]


def all_cas_from_synonyms(syn: list[str]) -> list[str]:
    return extract_cas(syn)


def first_cas_from_synonyms(syn: list[str]) -> str:
    vals = extract_cas(syn)
    return vals[0] if vals else ""


def parent_name_variants(parent_name: str) -> list[str]:
    """Parent name plus its anion form(s): '...ic acid' -> '...ate', '...ous acid' -> '...ite'.

    Salts of acids are almost always named with the anion, so matching only the
    literal acid name systematically misses them (e.g. sodium perfluorooctanoate).
    """
    base = norm_name(parent_name)
    out = [base] if len(base) >= 3 else []
    m = re.fullmatch(r"(.+?)ic acid", base)
    if m:
        out.append(m.group(1) + "ate")
    m = re.fullmatch(r"(.+?)ous acid", base)
    if m:
        out.append(m.group(1) + "ite")
    return [v for v in dict.fromkeys(out) if len(v) >= 3]


def _name_hits(variant: str, h: str) -> bool:
    return variant == h or h.startswith(variant + " ") or (" " + variant + " ") in (" " + h + " ")


def parent_name_is_explicit(parent_name: str, title: str, synonyms: list[str]) -> bool:
    variants = parent_name_variants(parent_name)
    if not variants:
        return False
    hay = candidate_name_haystack(title, synonyms)
    return any(_name_hits(v, h) for v in variants for h in hay)


def explicit_query_is_supported(query: str, title: str, synonyms: list[str]) -> bool:
    q = norm_name(query)
    hay = candidate_name_haystack(title, synonyms)
    return any(q == h or h.startswith(q + " ") or q in h for h in hay)



def pubchem_word_cids(name: str, max_records: int = WORD_SEARCH_MAX_CIDS) -> list[int]:
    """Return a bounded PubChem word-name search list. Failure is non-fatal."""
    url = f"{PUBCHEM}/compound/name/{quote(name, safe='')}/cids/JSON?name_type=word"
    j = cached_json(url)
    ids = (((j or {}).get("IdentifierList") or {}).get("CID") or [])
    out = []
    for x in ids:
        try:
            cid = int(x)
        except Exception:
            continue
        if cid not in out:
            out.append(cid)
        if len(out) >= max_records:
            break
    return out


def candidate_name_haystack(title: str, synonyms: list[str]) -> list[str]:
    return [norm_name(title)] + [norm_name(x) for x in synonyms[:SYNONYM_WINDOW]]


def salt_token_is_explicit(salt_key: str, title: str, synonyms: list[str]) -> bool:
    aliases = [norm_name(x) for x in SALT_ALIASES.get(salt_key, [salt_key])]
    hay = candidate_name_haystack(title, synonyms)
    return any(any((f" {a} " in f" {h} ") or h.endswith(" " + a) or h.startswith(a + " ") for a in aliases) for h in hay)


def complex_form_is_explicit(title: str, synonyms: list[str]) -> bool:
    hay = candidate_name_haystack(title, synonyms)
    return any(any(tok in h for tok in COMPLEX_FORM_ALIASES) for h in hay)


CATION_SALT_KEYS = {"sodium", "potassium", "lithium", "ammonium", "calcium", "magnesium"}


def query_variants(parent: str, salt_key: str) -> list[str]:
    # Variants are search routes only; final acceptance is based on source evidence.
    aliases = SALT_ALIASES.get(salt_key, [salt_key])
    anion_forms = parent_name_variants(parent)[1:] if salt_key in CATION_SALT_KEYS else []
    out = []
    for a in aliases:
        qs = [f"{parent} {a}", f"{parent} {a} salt", f"{parent}, {a} salt", f"{a} {parent}"]
        qs += [f"{a} {v}" for v in anion_forms]
        for q in qs:
            q = re.sub(r"\s+", " ", q).strip()
            if q not in out:
                out.append(q)
    return out


def source_url_for_cid(cid: int | str) -> str:
    return f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"


def has_strict_broad_salt_phrase(text: str) -> bool:
    t = clean(text)
    return any(p.search(t) for p in BROAD_SALT_PATTERNS)


def is_multicomponent_parent_smiles(smiles: str) -> bool:
    # Dot-disconnected parents are often already salts/complexes.  Auto-discovered
    # parent rules must represent the reference parent itself, not a specific salt.
    return "." in clean(smiles)


def extract_parent_cas_from_broad_scope(text: str, row: pd.Series) -> str:
    """Choose the parent CAS only for a strict parent-and-salts scope.

    Priority: an explicit row parent-CAS field, then a CAS immediately preceding
    the English 'and its/their salts' phrase, then the sole CAS in the source row.
    Rows with multiple unresolved CAS values are not auto-frozen.
    """
    for c in ["reference_parent_cas", "parent_cas"]:
        if c in row.index and valid_cas(clean(row.get(c))):
            return clean(row.get(c))
    t = clean(text)
    m = re.search(r"(\d{2,7}-\d{2}-\d)\s*\)?\s*and\s+(?:all\s+)?(?:its|their)\s+salts\b", t, flags=re.I)
    if m and valid_cas(m.group(1)):
        return m.group(1)
    vals = extract_cas([t])
    return vals[0] if len(vals) == 1 else ""


def strict_scope_row_status(row: pd.Series, text: str) -> tuple[bool, str]:
    typ = clean(row.get("reference_scope_type")) if "reference_scope_type" in row.index else ""
    phrase = has_strict_broad_salt_phrase(text)
    if typ:
        if "BROAD_SALT" not in typ.upper():
            return False, f"reference_scope_type={typ}"
        if not phrase:
            return False, "scope_type_broad_but_no_strict_parent_and_salts_phrase"
    elif not phrase:
        return False, "no_strict_parent_and_salts_phrase"
    return True, "strict_parent_and_salts_scope"


def generic_explicit_salt_relation(parent_query: str, title: str, synonyms: list[str]) -> bool:
    """Accept a generic 'parent ... salt' synonym only when parent and salt occur
    in the SAME PubChem name string.  This lets uncommon salts enter without
    relying on a fixed counterion dictionary.
    """
    variants = parent_name_variants(parent_query)
    if not variants:
        return False
    for raw in [title] + list(synonyms[:SYNONYM_WINDOW]):
        h = norm_name(raw)
        if not h or not any(v in h for v in variants):
            continue
        if re.search(r"\bsalt\b|\bsalts\b", h):
            return True
    return False


def detect_salt_evidence(parent_query: str, title: str, synonyms: list[str]) -> tuple[bool, str]:
    for key in SALT_ALIASES:
        if salt_token_is_explicit(key, title, synonyms):
            return True, key
    if generic_explicit_salt_relation(parent_query, title, synonyms):
        return True, "generic_explicit_salt"
    return False, ""


def classify_positive_difficulty(title: str, synonyms: list[str], salt_key: str) -> tuple[str, str]:
    if complex_form_is_explicit(title, synonyms):
        return "COMPLEX_SALT_OR_SOLVATE_POSITIVE", "HARD"
    easy = {"sodium", "potassium", "hydrochloride", "hydrobromide", "chloride", "bromide"}
    if salt_key in easy:
        return "COMMON_SALT_POSITIVE", "EASY"
    return "UNCOMMON_SALT_POSITIVE", "MODERATE"


def infer_scope_text_col(df: pd.DataFrame) -> str:
    for c in ["source_text", "regulatory_scope_text", "chemical_name", "name", "designation_name"]:
        if c in df.columns:
            return c
    return ""


def infer_designation_col(df: pd.DataFrame) -> str:
    for c in ["designation_id", "designation", "id", "번호", "고유번호"]:
        if c in df.columns:
            return c
    return ""


def infer_name_from_row(r: pd.Series, text: str, cas: str) -> str:
    for c in ["reference_parent_name", "chemical_name", "substance_name", "name", "designation_name"]:
        if c in r.index and clean(r.get(c)):
            v = clean(r.get(c))
            if cas:
                v = v.replace(cas, "")
            return re.sub(r"\s*\([^)]*\)\s*$", "", v).strip(" ,-()")
    if cas and cas in text:
        return text.split(cas)[0].strip(" ,-(")[-120:]
    return ""


def discover_additional_rules(scope: pd.DataFrame, existing_parent_cas: set[str]) -> tuple[list[dict], pd.DataFrame]:
    """Discover ONLY genuine parent + all-salts regulatory scopes.

    V2 was too permissive because any occurrence of 'salt/염류' could enter the
    parent pool.  V3 mirrors Study-1's strict BROAD_SALT phrase logic and rejects
    rows that are themselves specific salts/complexes.
    """
    if scope.empty:
        return [], pd.DataFrame()
    text_col = infer_scope_text_col(scope)
    did_col = infer_designation_col(scope)
    out, audit = [], []
    seen_cas = set(existing_parent_cas)
    for idx, r in scope.iterrows():
        text = clean(r.get(text_col)) if text_col else ""
        did = clean(r.get(did_col)) if did_col else f"AUTO_SCOPE_{idx+1:04d}"
        eligible, reason = strict_scope_row_status(r, text)
        audit_row = {
            "row_index": int(idx), "designation_id": did,
            "reference_scope_type": clean(r.get("reference_scope_type")) if "reference_scope_type" in r.index else "",
            "source_text": text, "strict_scope_phrase": has_strict_broad_salt_phrase(text),
            "decision": "EXCLUDE", "reason": reason, "selected_parent_cas": "", "pubchem_parent_title": "",
        }
        if not eligible:
            # Only write salt-related rejects to keep the audit readable.
            if re.search(r"염|salt", text, flags=re.I):
                audit.append(audit_row)
            continue
        cas = extract_parent_cas_from_broad_scope(text, r)
        if not cas:
            audit_row["reason"] = "parent_cas_not_unambiguously_resolved"
            audit.append(audit_row); continue
        if cas in seen_cas:
            audit_row.update({"reason": "already_in_core_or_previous_rule", "selected_parent_cas": cas})
            audit.append(audit_row); continue
        pc = resolve_exact_cas(cas)
        if not pc:
            audit_row.update({"reason": "exact_cas_pubchem_verification_failed", "selected_parent_cas": cas})
            audit.append(audit_row); continue
        title = clean(pc.get("Title"))
        psmiles = clean(pc.get("SMILES") or pc.get("ConnectivitySMILES"))
        if is_multicomponent_parent_smiles(psmiles):
            audit_row.update({"reason": "candidate_parent_is_dot_disconnected_specific_salt_or_complex", "selected_parent_cas": cas, "pubchem_parent_title": title})
            audit.append(audit_row); continue
        rid = "AUTO_BS_" + re.sub(r"[^A-Za-z0-9]+", "_", did).strip("_")
        out.append({
            "rule_id": rid,
            "designation_id": did,
            "reference_parent_name": title,
            "pubchem_parent_title": title,
            "reference_parent_cas": cas,
            "reference_smiles": psmiles,
            "salt_scope": "all_counterion_salts",
            "automation_status": "verified",
            "source_verified": "true",
            "reference_rule_frozen": "false",
            "regulatory_source": text or "Study-1 strict broad-salt scope row",
            "chemical_identity_source": source_url_for_cid(pc["CID"]),
            "rule_version": "AUTO_DISCOVERED_V3_STRICT_SCOPE",
            "frozen_date": "",
            "notes": "Auto-discovered only from strict Study-1 parent-and-salts scope; parent exact-CAS verified against PubChem.",
            "parent_pubchem_cid": pc["CID"],
            "parent_inchikey": clean(pc.get("InChIKey")),
            "auto_rule_status": "STRICT_BROAD_SALT_PARENT_PENDING_FREEZE",
        })
        seen_cas.add(cas)
        audit_row.update({"decision": "ACCEPT", "reason": "strict_broad_salt_parent_exact_cas_verified", "selected_parent_cas": cas, "pubchem_parent_title": title})
        audit.append(audit_row)
        if len(out) >= max(0, DISCOVERY_PARENT_POOL):
            break
    return out, pd.DataFrame(audit)


def positive_candidates(rule: dict) -> tuple[list[dict], list[dict]]:
    """Find source-supported positive salts using four PubChem discovery routes.

    Discovery routes (never decide the label):
      A  explicit counterion-name queries (incl. anion forms for cation salts)
      B  bounded PubChem word search on the parent name
      C  explicit complex salt/solvate name queries
      D  structure-first: PubChem mixture/component links of the parent CID

    SOURCE evidence for MATCH (at least one required, plus a CAS in the record):
      - PUBCHEM_PARENT_LINK: PubChem assigns the reference parent CID as this
        record's parent (cids_type=parent), or
      - NAME: title/synonym contains the parent (or its anion form) AND salt evidence.

    Pre-freeze QC gates (downgrade-only, never create/flip labels):
      - parent-fragment structural relation must be EXACT_PARENT_FRAGMENT
      - salt-form QC must pass
    """
    parent = clean(rule["reference_parent_name"])
    parent_query = clean(rule.get("pubchem_parent_title")) or parent
    parent_cas = clean(rule["reference_parent_cas"])
    parent_cid = int(rule["parent_pubchem_cid"])
    ref_smiles = clean(rule.get("reference_smiles"))
    candidate_cids: dict[int, set[str]] = {}
    prefetched: dict[int, dict] = {}

    def add(cid, route):
        cid = int(cid)
        if cid != parent_cid:
            candidate_cids.setdefault(cid, set()).add(route)

    # Route A
    for salt_key in SALT_ALIASES:
        for q in query_variants(parent_query, salt_key):
            for p in pubchem_properties_by_name_all(q):
                if p.get("CID"):
                    prefetched[int(p["CID"])] = p
                    add(p["CID"], f"NAME_QUERY:{q}")

    # Route B
    for cid in pubchem_word_cids(parent_query, max_records=WORD_SEARCH_MAX_CIDS):
        add(cid, "PARENT_WORD_SEARCH")

    # Route C
    for salt_key in ["sodium", "potassium", "hydrochloride", "chloride", "sulfate", "tartrate"]:
        for form in ["monohydrate", "dihydrate", "hydrate", "ethanol solvate", "isopropanol solvate"]:
            q = f"{parent_query} {SALT_ALIASES[salt_key][0]} {form}"
            for p in pubchem_properties_by_name_all(q):
                if p.get("CID"):
                    prefetched[int(p["CID"])] = p
                    add(p["CID"], f"COMPLEX_QUERY:{q}")

    # Route D
    mix, mix_total = mixture_cids(parent_cid)
    if mix_total > len(mix):
        print(f"  [WARN] Route D: {mix_total} mixture links for CID {parent_cid}; "
              f"only first {len(mix)} examined (STUDY2_MIXTURE_MAX_CIDS).")
    for cid in mix:
        add(cid, "STRUCTURE_MIXTURE_LINK")

    missing = [c for c in candidate_cids if c not in prefetched]
    prefetched.update(pubchem_properties_batch(missing))

    ok, review = [], []
    for cid, routes in candidate_cids.items():
        p = prefetched.get(cid) or pubchem_properties_by_cid(cid)
        if not p:
            continue
        smiles = clean(p.get("SMILES") or p.get("ConnectivitySMILES"))
        relation, rel_detail = structural_parent_relation(ref_smiles, smiles)
        route_str = ";".join(sorted(routes))

        # Cheap exit for structure-only hits that RDKit cannot reconcile with the
        # parent: keep an audit row, skip the extra synonym/parent requests.
        if routes == {"STRUCTURE_MIXTURE_LINK"} and relation in {"NO_PARENT_FRAGMENT", "PARSE_ERROR"}:
            review.append({
                "rule_id": rule["rule_id"], "designation_id": rule["designation_id"],
                "reference_parent_name": parent, "reference_parent_cas": parent_cas,
                "chemical_name": clean(p.get("Title")) or f"CID {cid}", "cas": "", "smiles": smiles,
                "reference_membership": "MATCH", "challenge_class": "POSSIBLE_SALT_POSITIVE", "difficulty": "HARD",
                "reference_source": source_url_for_cid(cid),
                "reference_evidence": f"PubChem mixture link from parent CID {parent_cid}; RDKit relation={relation}.",
                "candidate_pubchem_cid": cid, "candidate_inchikey": clean(p.get("InChIKey")),
                "generation_route": route_str, "reference_status": "REVIEW_REQUIRED",
                "structural_qc_relation": relation, "structural_qc_detail": rel_detail,
                "structural_qc_pass": False, "structural_qc_reason": "mixture_link_without_rdkit_parent_fragment",
                "review_reason": f"STRUCTURE:{relation}",
                "notes": "Route D hit not reconciled by RDKit; inspect manually.",
            })
            continue

        syn = pubchem_synonyms(cid)
        cas_all = all_cas_from_synonyms(syn)
        cas = cas_all[0] if cas_all else ""
        title = clean(p.get("Title"))
        name_parent_ok = parent_name_is_explicit(parent_query, title, syn)
        salt_ok, salt_key = detect_salt_evidence(parent_query, title, syn)
        name_ok = name_parent_ok and salt_ok
        pubchem_parents = pubchem_parent_cids(cid) if relation != "NO_PARENT_FRAGMENT" else []
        link_ok = parent_cid in pubchem_parents
        source_ok = bool(cas) and (name_ok or link_ok)
        evidence_types = [t for t, f in [("PUBCHEM_PARENT_LINK", link_ok), ("NAME", name_ok)] if f]

        sf_pass, sf_status, counterions = salt_form_qc(ref_smiles, smiles)

        if complex_form_is_explicit(title, syn) or sf_status == "SALT_FORM_CONSISTENT_WITH_SOLVATE":
            cls, diff = "COMPLEX_SALT_OR_SOLVATE_POSITIVE", "HARD"
        elif not name_parent_ok and link_ok:
            # Name does not reveal the parent: the hardest case for a name-based matcher.
            cls, diff = "NAME_OPAQUE_SALT_POSITIVE", "HARD"
        elif salt_ok:
            cls, diff = classify_positive_difficulty(title, syn, salt_key)
        elif counterions and set(counterions) <= _EASY_COUNTERIONS:
            cls, diff = "COMMON_SALT_POSITIVE", "EASY"
        elif link_ok:
            cls, diff = "UNCOMMON_SALT_POSITIVE", "MODERATE"
        else:
            cls, diff = "POSSIBLE_SALT_POSITIVE", "HARD"

        row = {
            "rule_id": rule["rule_id"], "designation_id": rule["designation_id"],
            "reference_parent_name": parent, "reference_parent_cas": parent_cas,
            "chemical_name": title or (syn[0] if syn else f"CID {cid}"), "cas": cas,
            "candidate_cas_all": ";".join(cas_all), "multi_cas_flag": len(cas_all) > 1,
            "smiles": smiles,
            "reference_membership": "MATCH", "challenge_class": cls, "difficulty": diff,
            "reference_source": source_url_for_cid(cid),
            "reference_evidence": (
                f"PubChem CID {cid} (CAS {cas or 'NA'}): "
                f"PubChem parent CID(s)={pubchem_parents or 'NA'} "
                f"{'includes' if link_ok else 'does not include'} reference parent CID {parent_cid}; "
                f"title/synonyms {'contain' if name_parent_ok else 'do not contain'} parent '{parent_query}' "
                f"(or anion form) and {'contain' if salt_ok else 'do not contain'} salt evidence '{salt_key or 'NA'}'."
            ),
            "source_evidence_types": ";".join(evidence_types),
            "pubchem_parent_cids": ";".join(map(str, pubchem_parents)),
            "salt_form_qc_status": sf_status, "salt_form_qc_pass": sf_pass,
            "counterions": ";".join(counterions),
            "candidate_pubchem_cid": cid, "candidate_inchikey": clean(p.get("InChIKey")),
            "generation_route": route_str,
            "reference_status": "AUTO_STRICT" if source_ok else "REVIEW_REQUIRED",
            "notes": "Discovered before model evaluation from PubChem source evidence.",
        }
        if not source_ok:
            row["notes"] += " SOURCE: " + ("no_cas_in_record" if not cas else "no_parent_link_or_name_evidence") + "."
        row = apply_structure_qc_row(row, ref_smiles)
        if row["reference_status"] == "AUTO_STRICT" and not sf_pass:
            row["reference_status"] = "REVIEW_REQUIRED"
            row["notes"] = (clean(row.get("notes")) + f" SALT_FORM_QC: {sf_status}.").strip()
        auto = row["reference_status"] == "AUTO_STRICT" and bool(row.get("structural_qc_pass"))
        row["reference_status"] = "AUTO_STRICT" if auto else "REVIEW_REQUIRED"
        reasons = []
        if not cas:
            reasons.append("NO_CAS_IN_RECORD")
        elif not (name_ok or link_ok):
            reasons.append("NO_SOURCE_EVIDENCE")
        if not row.get("structural_qc_pass"):
            reasons.append(f"STRUCTURE:{relation}")
        if not sf_pass:
            reasons.append(f"SALT_FORM:{sf_status}")
        row["review_reason"] = "" if auto else ";".join(reasons)
        (ok if auto else review).append(row)

    rank = {"HARD": 0, "MODERATE": 1, "EASY": 2}
    ok = sorted(ok, key=lambda x: (rank.get(x.get("difficulty"), 9), str(x.get("cas"))))
    seen, dedup = set(), []
    for x in ok:
        k = (x["rule_id"], x["cas"])
        if x["cas"] and k not in seen:
            seen.add(k); dedup.append(x)
    return dedup, review


def hard_negative_candidates(rule: dict) -> tuple[list[dict], list[dict]]:
    parent = clean(rule["reference_parent_name"])
    parent_query = clean(rule.get("pubchem_parent_title")) or parent
    parent_cid = int(rule["parent_pubchem_cid"])
    ok, review = [], []
    for cid in similar_cids(parent_cid):
        p = pubchem_properties_by_cid(cid)
        if not p:
            continue
        syn = pubchem_synonyms(cid)
        cas = first_cas_from_synonyms(syn)
        title = clean(p.get("Title"))
        if not cas:
            continue
        # Strict auto-negative only when PubChem identifies a distinctly named
        # candidate and nowhere in the inspected name set explicitly calls it the parent.
        parent_explicit = parent_name_is_explicit(parent_query, title, syn)
        distinct_named = bool(title and norm_name(title) != norm_name(parent_query))
        evidence_ok = distinct_named and not parent_explicit
        row = {
            "rule_id": rule["rule_id"], "designation_id": rule["designation_id"],
            "reference_parent_name": parent, "reference_parent_cas": rule["reference_parent_cas"],
            "chemical_name": title or (syn[0] if syn else f"CID {cid}"), "cas": cas,
            "smiles": clean(p.get("SMILES") or p.get("ConnectivitySMILES")),
            "reference_membership": "NO_MATCH", "challenge_class": "CLOSE_ANALOG_NEGATIVE", "difficulty": "HARD",
            "reference_source": source_url_for_cid(cid),
            "reference_evidence": (
                f"CID {cid} returned by PubChem >= {SIMILARITY_THRESHOLD}% 2-D similarity search to parent CID {parent_cid}; "
                "PubChem title/synonyms identify a distinct named compound and do not explicitly identify it as the reference parent."
            ),
            "candidate_pubchem_cid": cid, "candidate_inchikey": clean(p.get("InChIKey")),
            "generation_route": "PUBCHEM_SIMILARITY_HARD_NEGATIVE",
            "reference_status": "AUTO_STRICT" if evidence_ok else "REVIEW_REQUIRED",
            "notes": "Structurally similar PubChem hit used as a hard negative candidate.",
        }
        row = apply_structure_qc_row(row, clean(rule.get("reference_smiles")))
        evidence_ok = bool(evidence_ok and row.get("structural_qc_pass"))
        row["reference_status"] = "AUTO_STRICT" if evidence_ok else "REVIEW_REQUIRED"
        (ok if evidence_ok else review).append(row)
    seen = set(); dedup = []
    for x in ok:
        k = (x["rule_id"], x["cas"])
        if k not in seen:
            seen.add(k); dedup.append(x)
    return dedup, review


def add_cross_parent_negatives(pool: list[dict], rules: list[dict]) -> list[dict]:
    positives_by_rule: dict[str, list[dict]] = {}
    for x in pool:
        if x.get("reference_status") == "AUTO_STRICT" and x.get("reference_membership") == "MATCH":
            positives_by_rule.setdefault(x["rule_id"], []).append(x)
    extra = []
    for r in rules:
        rid = r["rule_id"]
        others = [q for q in rules if q["rule_id"] != rid and q["reference_parent_cas"] != r["reference_parent_cas"]]
        for q in others[:6]:
            # other parent itself = easy negative
            pc = resolve_exact_cas(q["reference_parent_cas"])
            if pc:
                extra.append({
                    "rule_id": rid, "designation_id": r["designation_id"],
                    "reference_parent_name": r["reference_parent_name"], "reference_parent_cas": r["reference_parent_cas"],
                    "chemical_name": clean(pc.get("Title")) or q["reference_parent_name"], "cas": q["reference_parent_cas"],
                    "smiles": clean(pc.get("SMILES") or pc.get("ConnectivitySMILES")),
                    "reference_membership": "NO_MATCH", "challenge_class": "DISTINCT_PARENT_NEGATIVE", "difficulty": "EASY",
                    "reference_source": source_url_for_cid(pc["CID"]),
                    "reference_evidence": f"PubChem exact-CAS record identifies this candidate as distinct parent '{q['reference_parent_name']}' ({q['reference_parent_cas']}), not '{r['reference_parent_name']}'.",
                    "candidate_pubchem_cid": pc["CID"], "candidate_inchikey": clean(pc.get("InChIKey")),
                    "generation_route": "CROSS_PARENT_PARENT_NEGATIVE", "reference_status": "AUTO_STRICT",
                    "notes": "Cross-parent negative constructed from another independently source-verified parent.",
                })
            # another parent's explicit salt = moderate negative
            for x in positives_by_rule.get(q["rule_id"], [])[:1]:
                y = dict(x)
                y.update({
                    "rule_id": rid, "designation_id": r["designation_id"],
                    "reference_parent_name": r["reference_parent_name"], "reference_parent_cas": r["reference_parent_cas"],
                    "reference_membership": "NO_MATCH", "challenge_class": "DISTINCT_PARENT_SALT_NEGATIVE", "difficulty": "MODERATE",
                    "reference_evidence": f"PubChem identifies candidate as a salt/form of distinct parent '{q['reference_parent_name']}', so it is not a salt of '{r['reference_parent_name']}'.",
                    "generation_route": "CROSS_PARENT_SALT_NEGATIVE", "reference_status": "AUTO_STRICT",
                    "notes": "Cross-parent salt negative constructed from another independently source-supported positive.",
                })
                extra.append(y)
    return extra


def select_balanced_additions(pool_df: pd.DataFrame, base_cases: pd.DataFrame, rules_df: pd.DataFrame) -> pd.DataFrame:
    """Select additions without ever filling a missing positive quota with negatives.

    Selection is chemistry/evidence driven and fixed before model evaluation.
    It aims for near 1:1 MATCH/NO_MATCH globally while preferring HARD then MODERATE cases.
    """
    if pool_df.empty:
        return pd.DataFrame(columns=pool_df.columns)
    strict = pool_df[pool_df["reference_status"].eq("AUTO_STRICT")].copy()
    strict = strict.drop_duplicates(subset=["rule_id", "cas", "reference_membership"], keep="first")
    if "duplicate_of_core" in strict.columns:
        strict = strict[~strict["duplicate_of_core"].astype(bool)].copy()
    diff_rank = {"HARD": 0, "MODERATE": 1, "EASY": 2}
    def route_priority(route: str) -> int:
        # generation_route is a ';'-joined set such as "NAME_QUERY:...;STRUCTURE_MIXTURE_LINK".
        r = clean(route)
        order = [
            ("COMPLEX_QUERY:", 0), ("STRUCTURE_MIXTURE_LINK", 1), ("PARENT_WORD_SEARCH", 2),
            ("NAME_QUERY:", 3), ("PUBCHEM_SIMILARITY_HARD_NEGATIVE", 0),
            ("CROSS_PARENT_SALT_NEGATIVE", 1), ("CROSS_PARENT_PARENT_NEGATIVE", 2),
        ]
        hits = [rank for key, rank in order if key in r]
        return min(hits) if hits else 9

    strict["_dr"] = strict["difficulty"].map(diff_rank).fillna(9)
    strict["_rr"] = strict["generation_route"].map(route_priority).fillna(9)

    core_pos = int((base_cases["reference_membership"].astype(str).str.upper() == "MATCH").sum())
    core_neg = int((base_cases["reference_membership"].astype(str).str.upper() == "NO_MATCH").sum())
    core_n = core_pos + core_neg
    max_add = max(0, TARGET_MAX_CASES - core_n)
    min_add = max(0, TARGET_MIN_CASES - core_n)

    selected_idx = []
    pos_added = neg_added = 0
    # Only consider rules exported for the final benchmark. Rich-positive rules sort first.
    for rid in rules_df["rule_id"].map(clean):
        g = strict[strict["rule_id"].eq(rid)]
        if g.empty:
            continue
        core_g = base_cases[base_cases["rule_id"].map(clean).eq(rid)] if "rule_id" in base_cases.columns else pd.DataFrame()
        existing_n = len(core_g)
        capacity = max(0, TARGET_CASES_PER_PARENT - existing_n)
        if capacity <= 0:
            continue
        pos = g[g["reference_membership"].eq("MATCH")].sort_values(["_dr", "_rr"])
        neg = g[g["reference_membership"].eq("NO_MATCH")].sort_values(["_dr", "_rr"])
        pair_n = min(len(pos), len(neg), capacity // 2)
        if pair_n <= 0:
            continue
        chosen = list(pos.head(pair_n).index) + list(neg.head(pair_n).index)
        # If odd capacity remains, choose the label that keeps global fraction closest to 0.50.
        if len(chosen) < capacity and len(chosen) < len(g):
            used = set(chosen)
            pnext = next((i for i in pos.index if i not in used), None)
            nnext = next((i for i in neg.index if i not in used), None)
            total_pos = core_pos + pos_added + pair_n
            total_neg = core_neg + neg_added + pair_n
            if pnext is not None and (nnext is None or total_pos <= total_neg): chosen.append(pnext)
            elif nnext is not None: chosen.append(nnext)
        if len(selected_idx) + len(chosen) > max_add:
            chosen = chosen[:max(0, max_add - len(selected_idx))]
        selected_idx.extend(chosen)
        pos_added += sum(strict.loc[i, "reference_membership"] == "MATCH" for i in chosen)
        neg_added += sum(strict.loc[i, "reference_membership"] == "NO_MATCH" for i in chosen)
        if len(selected_idx) >= max_add:
            break

    out = strict.loc[selected_idx].copy() if selected_idx else pd.DataFrame(columns=strict.columns)
    # Hard safety: never export an expansion with strong negative skew. If the requested
    # minimum size cannot be reached while balanced, 04B will remain blocked rather than
    # paying for Claude on a weak benchmark.
    if len(out):
        final_pos = core_pos + int((out["reference_membership"] == "MATCH").sum())
        final_neg = core_neg + int((out["reference_membership"] == "NO_MATCH").sum())
        frac = final_pos / max(final_pos + final_neg, 1)
        if not (0.40 <= frac <= 0.60):
            print(f"[WARN] balanced expansion could not reach target: projected positive fraction={frac:.3f}")
    if core_n + len(out) < TARGET_MIN_CASES:
        print(f"[WARN] balanced source-supported pool reaches only {core_n + len(out)} cases (< {TARGET_MIN_CASES}). Claude must remain blocked.")
    return out.drop(columns=[c for c in ["_dr", "_rr"] if c in out.columns], errors="ignore")

def run_single_parent(parent_cas: str, parent_name: str = "") -> None:
    """Collect salts for ONE parent without building/balancing the benchmark."""
    cas = clean(parent_cas)
    if not valid_cas(cas):
        raise ValueError(f"invalid CAS: {cas}")
    pc = resolve_exact_cas(cas)
    if not pc:
        raise RuntimeError(f"PubChem exact-CAS verification failed for {cas}")
    psmiles = clean(pc.get("SMILES") or pc.get("ConnectivitySMILES"))
    if is_multicomponent_parent_smiles(psmiles):
        print(f"[WARN] PubChem record for {cas} is itself multi-component: {psmiles}")
    rule = {
        "rule_id": f"SINGLE_{cas}", "designation_id": "SINGLE",
        "reference_parent_name": clean(parent_name) or clean(pc.get("Title")),
        "pubchem_parent_title": clean(pc.get("Title")),
        "reference_parent_cas": cas, "reference_smiles": psmiles,
        "parent_pubchem_cid": pc["CID"], "parent_inchikey": clean(pc.get("InChIKey")),
    }
    print(f"Parent: {rule['pubchem_parent_title']} (CAS {cas}, CID {pc['CID']})")
    print(f"Anion-form name variants: {parent_name_variants(rule['pubchem_parent_title'])}")
    ok, review = positive_candidates(rule)
    ok_df, rev_df = pd.DataFrame(ok), pd.DataFrame(review)
    stem = INTER / f"04A_single_parent_{cas}"
    ok_df.to_csv(f"{stem}_positives.csv", index=False, encoding="utf-8-sig")
    rev_df.to_csv(f"{stem}_review.csv", index=False, encoding="utf-8-sig")

    print(f"\nAUTO_STRICT positives: {len(ok_df)}   REVIEW: {len(rev_df)}")
    if len(ok_df):
        print("\nBy source evidence:")
        print(ok_df["source_evidence_types"].value_counts().to_string())
        print("\nBy challenge class:")
        print(ok_df["challenge_class"].value_counts().to_string())
        struct_only = ok_df["generation_route"].eq("STRUCTURE_MIXTURE_LINK").sum()
        print(f"\nFound ONLY via Route D (structure-first): {struct_only}")
        cols = ["chemical_name", "cas", "counterions", "challenge_class", "source_evidence_types"]
        print("\n" + ok_df[cols].to_string(index=False, max_colwidth=60))
    if len(rev_df):
        reason = rev_df.get("review_reason", pd.Series("", index=rev_df.index)).fillna("")
        reason = reason.where(reason.ne(""), rev_df.get("structural_qc_reason", pd.Series("", index=rev_df.index)))
        print("\nReview reasons:")
        print(reason.value_counts().to_string())
    print(f"\nWritten: {stem}_positives.csv / {stem}_review.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parent-cas", help="single-parent mode: collect salts for this parent CAS only")
    ap.add_argument("--parent-name", default="", help="optional display name for single-parent mode")
    args = ap.parse_args()
    if args.parent_cas:
        run_single_parent(args.parent_cas, args.parent_name)
        return

    print("=" * 94)
    print("04A V4.2 / RATE-SAFE AUTO-BUILD + STRUCTURE-QC BROAD-SALT BENCHMARK (PUBCHEM ONLY; NO CLAUDE)")
    print(f"PubChem pacing: {PAUSE:.2f}s/request (heavy: {max(PAUSE, HEAVY_PAUSE):.2f}s); retries={PUBCHEM_RETRIES}; backoff_max={PUBCHEM_BACKOFF_MAX:.0f}s")
    print("=" * 94)

    rule_path = download_if_missing("broad_salt_rules.csv", [DATA / "broad_salt_rules.csv", ENGINE_DATA / "broad_salt_rules.csv", ROOT / "broad_salt_rules.csv"])
    case_path = download_if_missing("broad_salt_validation_cases.csv", [DATA / "broad_salt_validation_cases.csv", ENGINE_DATA / "broad_salt_validation_cases.csv", ROOT / "broad_salt_validation_cases.csv"])
    if not rule_path or not case_path:
        raise FileNotFoundError("Base broad_salt_rules.csv and broad_salt_validation_cases.csv are required.")
    rules = read_table(rule_path)
    base_cases = read_table(case_path)

    scope_candidates = [INTER / "01_cas_gap_benchmark_full.csv", INTER / "01_cas_gap_benchmark_primary.csv", DATA / "01_cas_gap_benchmark_full.csv"]
    scope_path = next((p for p in scope_candidates if p.exists()), None)
    scope = read_table(scope_path) if scope_path else pd.DataFrame()

    for c in ["rule_id", "designation_id", "reference_parent_name", "reference_parent_cas", "reference_smiles"]:
        if c not in rules.columns: rules[c] = ""
    for c in ["case_id", "rule_id", "reference_membership"]:
        if c not in base_cases.columns: base_cases[c] = ""

    # 1) Resolve CORE parents, but always retain PubChem's title as the search name.
    core_rules = []
    existing_cas = set()
    for _, r in rules.iterrows():
        cas = clean(r.get("reference_parent_cas"))
        if not valid_cas(cas): continue
        pc = resolve_exact_cas(cas)
        if not pc:
            print(f"[WARN] parent exact-CAS verification failed: {cas}"); continue
        d = {k: clean(v) for k, v in r.to_dict().items()}
        d["reference_parent_name"] = clean(d.get("reference_parent_name")) or clean(pc.get("Title"))
        d["pubchem_parent_title"] = clean(pc.get("Title")) or d["reference_parent_name"]
        d["reference_smiles"] = clean(d.get("reference_smiles")) or clean(pc.get("SMILES") or pc.get("ConnectivitySMILES"))
        d["parent_pubchem_cid"] = pc["CID"]; d["parent_inchikey"] = clean(pc.get("InChIKey"))
        d["auto_rule_status"] = "CORE_SOURCE_VERIFIED"
        core_rules.append(d); existing_cas.add(cas)

    # 2) Discover a larger parent pool. We do NOT just take the first 20.
    discovered, scope_audit = discover_additional_rules(scope, existing_cas) if not scope.empty else ([], pd.DataFrame())
    candidate_rules = pd.DataFrame(core_rules + discovered).drop_duplicates(subset=["reference_parent_cas"], keep="first")
    if candidate_rules.empty: raise RuntimeError("No PubChem-verifiable broad-salt parent rules available.")
    print(f"Candidate parent pool: {len(candidate_rules)} (core={len(core_rules)}, discovered={len(discovered)})")

    # 3) Positive discovery FIRST for every candidate parent. New parents are selected
    # by source-supported positive richness, not by downstream model performance.
    pos_pool = []; review = []; pos_count = {}
    for i, rr in candidate_rules.iterrows():
        rule = rr.to_dict()
        print(f"[positive {i+1}/{len(candidate_rules)}] {rule['reference_parent_name']} / PubChem='{rule.get('pubchem_parent_title','')}'")
        p_ok, p_rev = positive_candidates(rule)
        pos_pool.extend(p_ok); review.extend(p_rev); pos_count[rule["rule_id"]] = len(p_ok)

    core_ids = {clean(x.get("rule_id")) for x in core_rules}
    new_df = candidate_rules[~candidate_rules["rule_id"].map(clean).isin(core_ids)].copy()
    new_df["strict_positive_n"] = new_df["rule_id"].map(pos_count).fillna(0).astype(int)
    # Prefer parents with >=2 independently source-supported positive salts; tie-break by designation.
    rich = new_df[new_df["strict_positive_n"] >= MIN_STRICT_POS_PER_NEW_PARENT].sort_values(["strict_positive_n", "designation_id"], ascending=[False, True])
    weak = new_df[new_df["strict_positive_n"] < MIN_STRICT_POS_PER_NEW_PARENT].sort_values(["strict_positive_n", "designation_id"], ascending=[False, True])

    # V3 never pads the benchmark with weak/zero-positive parent rules merely to
    # hit a target count.  If the regulation contains fewer usable broad-salt
    # parents, the readiness gate must fail honestly rather than manufacture diversity.
    selected_new = rich.head(max(0, MAX_SELECTED_RULES - len(core_rules))).copy()
    if len(core_rules) + len(selected_new) < TARGET_PARENTS:
        print(f"[WARN] only {len(core_rules)+len(selected_new)} parent rules have strict positive support; target was {TARGET_PARENTS}. No weak parents were padded in.")
    selected_rules = pd.concat([pd.DataFrame(core_rules), selected_new], ignore_index=True, sort=False)
    selected_rules = selected_rules.drop_duplicates(subset=["reference_parent_cas"], keep="first").head(MAX_SELECTED_RULES).reset_index(drop=True)
    selected_ids = set(selected_rules["rule_id"].map(clean))
    print(f"Selected parent rules for benchmark construction: {len(selected_rules)}")
    print("Strict positive candidates by selected parent:")
    for _, rr in selected_rules.iterrows():
        print(f"  {rr['rule_id']}: {int(pos_count.get(rr['rule_id'],0))}")

    # 4) Generate negatives ONLY for selected parents, then cross-parent negatives.
    pool = [x for x in pos_pool if clean(x.get("rule_id")) in selected_ids]
    for i, rr in selected_rules.iterrows():
        rule = rr.to_dict()
        print(f"[negative {i+1}/{len(selected_rules)}] {rule['reference_parent_name']}")
        n_ok, n_rev = hard_negative_candidates(rule)
        pool.extend(n_ok); review.extend(n_rev)
    pool.extend(add_cross_parent_negatives(pool, selected_rules.to_dict("records")))

    pool_df = pd.DataFrame(pool)
    if pool_df.empty: raise RuntimeError("No candidate pool could be built. Check PubChem/network and parent names.")
    pool_df = pool_df.drop_duplicates(subset=["rule_id", "cas", "reference_membership", "generation_route"], keep="first")

    # Global pre-freeze structural-consistency QC also covers cross-parent rows.
    pool_df = apply_structure_qc_dataframe(pool_df, selected_rules)
    pool_df = mark_core_duplicates(pool_df, base_cases)
    pool_df.insert(0, "auto_candidate_id", [f"AUTOPOOL_{i+1:05d}" for i in range(len(pool_df))])

    # Push all QC-downgraded and CORE-duplicate records into the audit/review stream.
    extra_review = pool_df[
        ~pool_df["reference_status"].eq("AUTO_STRICT")
    ].copy()
    if len(extra_review):
        review.extend(extra_review.to_dict("records"))

    # 5) Strict balanced selection. Missing positives are NEVER replaced by extra negatives.
    selected = select_balanced_additions(pool_df, base_cases, selected_rules)
    if not selected.empty:
        selected = selected.copy().reset_index(drop=True)
        selected["case_id"] = [f"AUTO_BSV_{i+1:05d}" for i in range(len(selected))]
        keep_cols = [
            "case_id", "rule_id", "designation_id", "chemical_name", "cas", "smiles",
            "reference_membership", "reference_source", "notes", "challenge_class", "difficulty",
            "reference_evidence", "generation_route", "reference_status", "candidate_pubchem_cid", "candidate_inchikey",
        ]
        for c in keep_cols:
            if c not in selected.columns: selected[c] = ""
        selected = selected[keep_cols]

    # Export only newly discovered rules that actually appear in selected cases.
    used_rule_ids = set(selected["rule_id"].map(clean)) if len(selected) else set()
    base_rule_ids = set(rules["rule_id"].map(clean))
    new_rules = selected_rules[selected_rules["rule_id"].map(clean).isin(used_rule_ids - base_rule_ids)].copy()
    rule_keep = [
        "rule_id", "designation_id", "reference_parent_name", "reference_parent_cas", "reference_smiles",
        "salt_scope", "automation_status", "source_verified", "reference_rule_frozen", "regulatory_source",
        "chemical_identity_source", "rule_version", "frozen_date", "notes", "parent_pubchem_cid", "parent_inchikey", "auto_rule_status",
    ]
    for c in rule_keep:
        if c not in new_rules.columns: new_rules[c] = ""
    new_rules[rule_keep].to_csv(DATA / "broad_salt_rules_AUTO_EXPANDED.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(DATA / "broad_salt_validation_cases_AUTO_EXPANDED.csv", index=False, encoding="utf-8-sig")
    pool_df.to_csv(INTER / "04A_candidate_pool_all.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(review).drop_duplicates(
        subset=[c for c in ["rule_id", "cas", "reference_membership", "structural_qc_reason"] if c in pd.DataFrame(review).columns],
        keep="first",
    ).to_csv(INTER / "04A_candidate_review_queue.csv", index=False, encoding="utf-8-sig")

    qc_audit_cols = [
        "auto_candidate_id", "rule_id", "reference_parent_name", "reference_parent_cas",
        "chemical_name", "cas", "reference_membership", "challenge_class", "difficulty",
        "reference_status", "structural_qc_relation", "structural_qc_pass",
        "structural_qc_reason", "duplicate_of_core", "generation_route", "reference_source",
    ]
    [c for c in qc_audit_cols if c in pool_df.columns]
    pool_df[[c for c in qc_audit_cols if c in pool_df.columns]].to_csv(
        INTER / "04A_structure_qc_audit.csv", index=False, encoding="utf-8-sig"
    )

    strict_pos_after_qc = (
        pool_df[
            pool_df["reference_status"].eq("AUTO_STRICT")
            & pool_df["reference_membership"].eq("MATCH")
            & ~pool_df.get("duplicate_of_core", pd.Series(False, index=pool_df.index)).astype(bool)
        ]
        .groupby("rule_id")["cas"].nunique()
        .to_dict()
    )
    support_out = candidate_rules.copy()
    support_out["strict_positive_n_lexical_before_qc"] = support_out["rule_id"].map(pos_count).fillna(0).astype(int)
    support_out["strict_positive_n_after_structure_qc"] = support_out["rule_id"].map(strict_pos_after_qc).fillna(0).astype(int)
    support_out.to_csv(INTER / "04A_parent_positive_support.csv", index=False, encoding="utf-8-sig")
    scope_audit.to_csv(INTER / "04A_scope_discovery_audit.csv", index=False, encoding="utf-8-sig")

    core_pos = int((base_cases["reference_membership"].astype(str).str.upper() == "MATCH").sum())
    core_neg = int((base_cases["reference_membership"].astype(str).str.upper() == "NO_MATCH").sum())
    add_pos = int((selected.get("reference_membership", pd.Series(dtype=str)) == "MATCH").sum())
    add_neg = int((selected.get("reference_membership", pd.Series(dtype=str)) == "NO_MATCH").sum())
    combined_n = core_pos + core_neg + add_pos + add_neg
    combined_pos_frac = (core_pos + add_pos) / max(combined_n, 1)
    qc = pd.DataFrame([
        {"metric": "candidate_parent_pool", "value": int(len(candidate_rules))},
        {"metric": "strict_scope_rows_accepted", "value": int((scope_audit.get("decision", pd.Series(dtype=str)) == "ACCEPT").sum())},
        {"metric": "strict_scope_rows_excluded", "value": int((scope_audit.get("decision", pd.Series(dtype=str)) == "EXCLUDE").sum())},
        {"metric": "selected_parent_rules", "value": int(len(selected_rules))},
        {"metric": "new_rules_exported", "value": int(len(new_rules))},
        {"metric": "core_cases", "value": int(len(base_cases))},
        {"metric": "auto_selected_cases", "value": int(len(selected))},
        {"metric": "combined_cases_estimate", "value": int(combined_n)},
        {"metric": "core_match", "value": core_pos}, {"metric": "core_no_match", "value": core_neg},
        {"metric": "auto_match", "value": add_pos}, {"metric": "auto_no_match", "value": add_neg},
        {"metric": "projected_positive_fraction", "value": combined_pos_frac},
        {"metric": "auto_hard", "value": int((selected.get("difficulty", pd.Series(dtype=str)) == "HARD").sum())},
        {"metric": "review_queue", "value": int(len(review))},
        {"metric": "structure_qc_match_conflicts", "value": int(((pool_df["reference_membership"] == "MATCH") & ~pool_df["structural_qc_pass"].astype(bool)).sum())},
        {"metric": "structure_qc_no_match_conflicts", "value": int(((pool_df["reference_membership"] == "NO_MATCH") & ~pool_df["structural_qc_pass"].astype(bool)).sum())},
        {"metric": "core_duplicate_candidates_excluded", "value": int(pool_df.get("duplicate_of_core", pd.Series(False, index=pool_df.index)).astype(bool).sum())},
    ])
    qc.to_csv(INTER / "04A_autobuild_qc.csv", index=False, encoding="utf-8-sig")
    cfg = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_min_cases": TARGET_MIN_CASES, "target_max_cases": TARGET_MAX_CASES,
        "target_parents": TARGET_PARENTS, "discovery_parent_pool": DISCOVERY_PARENT_POOL,
        "min_strict_positive_per_new_parent": MIN_STRICT_POS_PER_NEW_PARENT,
        "ground_truth_guard": "Reference labels come from strict Study-1 broad-salt scope + PubChem source evidence. RDKit is used only as a pre-freeze consistency QC gate; it never flips labels. Structural conflicts are downgraded to REVIEW_REQUIRED and excluded from auto benchmark selection.",
        "scope_guard": "Auto-discovered parent rules require Study-1-consistent parent-and-salts phrase and non-dot-disconnected exact-CAS PubChem parent.",
        "selection_guard": "Parents/candidates selected before model evaluation; no case is chosen based on LLM/RDKit success or failure.",
        "study1_scope_file": str(scope_path) if scope_path else None,
        "base_rule_file": str(rule_path), "base_case_file": str(case_path),
    }
    (INTER / "04A_autobuild_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[04A V4.2 complete]")
    print(qc.to_string(index=False))
    print(f"\nSelected expansion: {DATA / 'broad_salt_validation_cases_AUTO_EXPANDED.csv'}")
    print(f"Positive-support audit: {INTER / '04A_parent_positive_support.csv'}")
    print(f"Scope audit: {INTER / '04A_scope_discovery_audit.csv'}")
    print(f"Review queue: {INTER / '04A_candidate_review_queue.csv'}")
    print(f"Structure QC audit: {INTER / '04A_structure_qc_audit.csv'}")
    print("Next: inspect the structure-QC audit, then run 04B. Do NOT enable Claude unless 04B says ready_for_claude_api=true.")


if __name__ == "__main__":
    main()
