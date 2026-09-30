# -*- coding: utf-8 -*-
"""Frozen salt-aware chemical-identity comparator for Study 2 FAIR V5.

This module is intentionally independent of benchmark labels and compound names.
It compares only candidate/reference structures under a generic parent-and-salts
policy. Freeze this file with 04D_freeze_fair_v5_protocol.py BEFORE evaluating an
independent final holdout.
"""
from __future__ import annotations

from typing import Iterable

try:
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
except Exception:  # pragma: no cover
    Chem = None
    rdMolStandardize = None

ENGINE_VERSION = "V5.1_FAIR_SALT_AWARE_FROZEN_2026-09-30"
POLICY_VERSION = "parent-salt-policy-v5.1"

# Generic chemistry classes only. No benchmark-compound-specific SMARTS or CAS rules.
_ACID_SMARTS = [
    "[CX3](=O)[OX2H1,OX1-]",       # carboxylic acid / carboxylate
    "[SX4](=O)(=O)[OX2H1,OX1-]",  # sulfonic acid / sulfonate
    "[PX4](=O)([OX2H1,OX1-])",    # phosphoric/phosphonic acid family
    "[cX3][OX2H1,OX1-]",          # phenol / phenolate
]
_BASE_SMARTS = [
    "[NX3;H0,H1,H2,H3;+0;!$(N-[C,S,P]=[O,S,N])]",  # neutral non-amide amine
    "[N+;X4;H0,H1,H2,H3,H4]",                   # ammonium / quaternary ammonium
    "[nX2;+0]",                                  # pyridine-like aromatic N
    "[nH+]",                                     # protonated aromatic N
]
# Neutral-drawn inorganic acids as PubChem writes them for salts (HCl -> "Cl",
# nitric acid -> "[N+](=O)(O)[O-]", perchloric acid -> "OCl(=O)(=O)=O", ...).
# Used ONLY to classify counterion fragments, never to classify the parent.
_MINERAL_ACID_SMARTS = [
    "[F,Cl,Br,I;X0,X1;+0]",               # hydrohalic acid drawn as a lone halogen
    "[OX2H1][!#6;!#1]=O",                 # inorganic oxoacid X(=O)OH (S, P, ...)
    "[OX2H1][!#6;!#1;!+0][OX1-]",        # same, charge-separated (RDKit perchloric acid)
    "[OX2H1][N+](=O)[O-]",                # nitric acid, charge-separated
    "[OX2H1][N](=O)=O",                   # nitric acid, pentavalent drawing
]
# Elements that are NOT metals; any other lone atom is treated as a metal atom.
_NONMETALS = {
    "H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar",
    "As", "Se", "Br", "Kr", "Te", "I", "Xe", "At", "Rn",
}
_COMMON_SOLVATES = {
    "O", "CO", "CCO", "CC(C)O", "CC(=O)C", "CC#N", "O=C=O",
    "CS(=O)C", "C1COCCO1",
}


def _mol(smiles: str):
    if Chem is None:
        return None
    s = str(smiles or "").strip()
    if not s:
        return None
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None


def _compile(smarts_list: Iterable[str]):
    out = []
    if Chem is None:
        return out
    for s in smarts_list:
        try:
            q = Chem.MolFromSmarts(s)
        except Exception:
            q = None
        if q is not None:
            out.append(q)
    return out


_ACID_MOLS = _compile(_ACID_SMARTS)
_BASE_MOLS = _compile(_BASE_SMARTS)
_MINERAL_ACID_MOLS = _compile(_MINERAL_ACID_SMARTS)


def _standardize_fragment(mol):
    """Cleanup, normalize, uncharge and canonicalize tautomers for representation invariance."""
    if mol is None or Chem is None:
        return None
    m = Chem.Mol(mol)
    if rdMolStandardize is not None:
        for fn in (rdMolStandardize.Cleanup, rdMolStandardize.Normalize):
            try:
                m = fn(m)
            except Exception:
                pass
        try:
            m = rdMolStandardize.Uncharger().uncharge(m)
        except Exception:
            pass
        try:
            m = rdMolStandardize.TautomerEnumerator().Canonicalize(m)
        except Exception:
            pass
    try:
        Chem.SanitizeMol(m)
    except Exception:
        pass
    return m


def _keys(mol) -> tuple[str, str]:
    m = _standardize_fragment(mol)
    if m is None:
        return "", ""
    try:
        return (
            Chem.MolToSmiles(m, canonical=True, isomericSmiles=True),
            Chem.MolToSmiles(m, canonical=True, isomericSmiles=False),
        )
    except Exception:
        return "", ""


def _contains(mol, patterns) -> bool:
    m = _standardize_fragment(mol)
    if m is None:
        return False
    for q in patterns:
        try:
            if m.HasSubstructMatch(q):
                return True
        except Exception:
            pass
    return False


def _is_acid(mol) -> bool:
    return _contains(mol, _ACID_MOLS)


def _is_base(mol) -> bool:
    return _contains(mol, _BASE_MOLS)


def _net_charge(mol) -> int:
    if mol is None:
        return 0
    try:
        return int(sum(a.GetFormalCharge() for a in mol.GetAtoms()))
    except Exception:
        return 0


def _is_mineral_acid(mol) -> bool:
    """Neutral-drawn inorganic acid fragment (only meaningful for counterions).

    Carbon-containing fragments are excluded so that e.g. a chlorinated organic
    co-component is never mistaken for hydrogen chloride.
    """
    if mol is None or any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
        return False
    for q in _MINERAL_ACID_MOLS:
        try:
            if mol.HasSubstructMatch(q):
                return True
        except Exception:
            pass
    return False


def _is_neutral_metal_atom(mol) -> bool:
    """A lone, uncharged metal atom, e.g. sodium drawn as "[Na]" instead of "[Na+]"."""
    if mol is None or mol.GetNumAtoms() != 1:
        return False
    a = mol.GetAtomWithIdx(0)
    return a.GetFormalCharge() == 0 and a.GetSymbol() not in _NONMETALS


def _is_solvate(mol) -> bool:
    m = _standardize_fragment(mol)
    if m is None:
        return False
    try:
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=False) in _COMMON_SOLVATES
    except Exception:
        return False


def _companion_class(mol, ref_is_acid: bool, ref_is_base: bool) -> tuple[bool, str]:
    """Accept only chemically compatible counterions or common solvates.

    This deliberately rejects an arbitrary charged co-component. A cation is a
    plausible counterion only for an acidic parent; an anion only for a basic
    parent. Neutral-drawn acid/base pairs are handled by generic functional-class
    recognition. If the reference is not recognizably ionizable, extra non-solvate
    fragments remain REVIEW rather than being forced to MATCH.
    """
    if mol is None:
        return False, "INVALID_COMPANION"
    if _is_solvate(mol):
        return True, "COMMON_SOLVATE"

    q = _net_charge(mol)
    if ref_is_acid:
        if q > 0:
            return True, "CATION_COUNTERION"
        if _is_neutral_metal_atom(mol):
            return True, "NEUTRAL_DRAWN_METAL_COUNTERION"
        if _is_base(mol):
            return True, "NEUTRAL_DRAWN_BASE_COUNTERION"
    if ref_is_base:
        if q < 0:
            return True, "ANION_COUNTERION"
        if _is_acid(mol) or _is_mineral_acid(mol):
            return True, "NEUTRAL_DRAWN_ACID_COUNTERION"
    return False, "UNEXPLAINED_OR_INCOMPATIBLE_COMPONENT"


def normalize_isomer_scope(value: str) -> str:
    x = str(value or "").strip().upper()
    aliases = {
        "ALL": "ALL_STEREOISOMERS",
        "ALL_ISOMERS": "ALL_STEREOISOMERS",
        "ALL_STEREOISOMERS": "ALL_STEREOISOMERS",
        "EXACT": "EXACT_STEREO_ONLY",
        "EXACT_ONLY": "EXACT_STEREO_ONLY",
        "EXACT_STEREO_ONLY": "EXACT_STEREO_ONLY",
    }
    return aliases.get(x, "UNSPECIFIED_REVIEW")


def compare_salt_parent_v5(candidate_smiles: str, reference_smiles: str, isomer_scope: str = "") -> tuple[str, str]:
    """Compare one candidate with one frozen reference parent.

    Rules
    -----
    * candidate and reference use the same structure source outside this function;
    * candidate is split into ALL disconnected fragments;
    * charge/protonation and tautomer representation are standardized;
    * one or more parent fragments may occur (stoichiometric salts);
    * every remaining fragment must be a compatible counterion or common solvate;
    * covalent derivatives/analogs without the parent fragment are NO_MATCH;
    * stereo-only differences follow the supplied frozen isomer-scope rule;
    * ambiguous extra components are REVIEW, not MATCH.
    """
    if Chem is None:
        return "REVIEW", "RDKIT_NOT_AVAILABLE"
    cm = _mol(candidate_smiles)
    rm = _mol(reference_smiles)
    if cm is None:
        return "REVIEW", "MISSING_OR_INVALID_CANDIDATE_STRUCTURE"
    if rm is None:
        return "REVIEW", "INVALID_REFERENCE_STRUCTURE"

    try:
        rfrags = list(Chem.GetMolFrags(rm, asMols=True, sanitizeFrags=True))
    except Exception:
        rfrags = [rm]
    if len(rfrags) != 1:
        return "REVIEW", "REFERENCE_MUST_BE_SINGLE_PARENT_COMPONENT"
    rm = rfrags[0]
    ref_iso, ref_conn = _keys(rm)
    if not ref_iso or not ref_conn:
        return "REVIEW", "INVALID_REFERENCE_STRUCTURE"

    try:
        cfrags = list(Chem.GetMolFrags(cm, asMols=True, sanitizeFrags=True))
    except Exception:
        cfrags = [cm]

    exact, conn = [], []
    for i, frag in enumerate(cfrags):
        fi, fc = _keys(frag)
        if fi and fi == ref_iso:
            exact.append(i)
        elif fc and fc == ref_conn:
            conn.append(i)
    if not exact and not conn:
        return "NO_MATCH", "NO_PARENT_FRAGMENT_CONNECTIVITY"

    parent_idx = set(exact) | set(conn)
    ref_is_acid, ref_is_base = _is_acid(rm), _is_base(rm)
    explanations = []
    for i, frag in enumerate(cfrags):
        if i in parent_idx:
            continue
        ok, why = _companion_class(frag, ref_is_acid, ref_is_base)
        explanations.append(why)
        if not ok:
            return "REVIEW", why

    if exact:
        suffix = "PARENT_FRAGMENT_EXACT"
        if explanations:
            suffix += "_WITH_" + "+".join(sorted(set(explanations)))
        return "MATCH", suffix

    scope = normalize_isomer_scope(isomer_scope)
    if scope == "ALL_STEREOISOMERS":
        return "MATCH", "PARENT_CONNECTIVITY_MATCH_ALL_STEREOISOMERS_RULE"
    if scope == "EXACT_STEREO_ONLY":
        return "NO_MATCH", "PARENT_CONNECTIVITY_ONLY_EXACT_STEREO_RULE"
    return "REVIEW", "STEREO_SCOPE_UNSPECIFIED"


def generic_self_tests() -> list[dict]:
    """Benchmark-independent chemistry sanity checks used by the protocol freeze step."""
    tests = [
        ("carboxylate_sodium", "O=C(O)c1ccccc1", "[Na+].[O-]C(=O)c1ccccc1", "MATCH"),
        ("phenolate_potassium", "Oc1ccccc1", "[K+].[O-]c1ccccc1", "MATCH"),
        ("amine_hydrochloride", "Nc1ccccc1", "Nc1ccccc1.[Cl-]", "MATCH"),
        ("neutral_drawn_acid_base_pair", "CC(=O)O", "CC(=O)O.N", "MATCH"),
        ("hydrate_representation", "O=C(O)c1ccccc1", "O.[Na+].[O-]C(=O)c1ccccc1", "MATCH"),
        ("covalent_ester", "O=C(O)c1ccccc1", "COC(=O)c1ccccc1", "NO_MATCH"),
        ("close_phenol_analog", "Oc1ccccc1", "Cc1ccccc1O", "NO_MATCH"),
        ("unrelated_charged_mixture_review", "O=C(O)c1ccccc1", "O=C(O)c1ccccc1.[Cl-]", "REVIEW"),
        # PubChem-style neutral drawings of common counterions
        ("amine_hydrochloride_neutral_drawn", "Nc1ccccc1", "Nc1ccccc1.Cl", "MATCH"),
        ("amine_dihydrobromide_neutral_drawn", "NCCN", "NCCN.Br.Br", "MATCH"),
        ("amine_nitrate_neutral_drawn", "Nc1ccccc1", "Nc1ccccc1.[N+](=O)(O)[O-]", "MATCH"),
        ("amine_perchlorate_neutral_drawn", "Nc1ccccc1", "Nc1ccccc1.OCl(=O)(=O)=O", "MATCH"),
        ("carboxylic_acid_neutral_sodium_atom", "O=C(O)c1ccccc1", "O=C(O)c1ccccc1.[Na]", "MATCH"),
        ("acid_with_neutral_hcl_review", "O=C(O)c1ccccc1", "O=C(O)c1ccccc1.Cl", "REVIEW"),
        ("amine_with_neutral_metal_review", "Nc1ccccc1", "Nc1ccccc1.[Na]", "REVIEW"),
        ("amine_with_chlorinated_organic_review", "Nc1ccccc1", "Nc1ccccc1.ClCCCl", "REVIEW"),
    ]
    out = []
    for name, ref, cand, expected in tests:
        decision, reason = compare_salt_parent_v5(cand, ref, "")
        out.append({"test": name, "expected": expected, "decision": decision, "reason": reason, "pass": decision == expected})
    return out


if __name__ == "__main__":
    rows = generic_self_tests()
    for r in rows:
        print(r)
    if not all(r["pass"] for r in rows):
        raise SystemExit("V5 generic self-test failed")
