# -*- coding: utf-8 -*-
"""Deterministic low-cost gate for Study 2 chemical-group/mixture extension V1.

Design principle: regulatory source first, chemistry second.
- exact CAS in the official Annex-3 member registry -> MATCH
- exact CAS known only under another benchmark group -> NO_MATCH
- otherwise use a conservative chemistry/name fallback and return REVIEW when
  open-world membership cannot be resolved reliably.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent
RULE_FILE = ROOT / "data" / "regulatory_group_rules_v1.csv"


def clean(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def normalize_cas(x) -> str:
    return re.sub(r"\s+", "", clean(x))


def _load_registry(path: Path = RULE_FILE):
    df = pd.read_csv(path, dtype=str).fillna("")
    groups = {}
    cas_to_groups = {}
    for rid, g in df.groupby("rule_id", sort=False):
        first = g.iloc[0]
        members = {normalize_cas(v) for v in g["member_cas"] if normalize_cas(v)}
        groups[rid] = {
            "rule_id": rid,
            "regulatory_id": clean(first["regulatory_id"]),
            "group_name": clean(first["group_name"]),
            "scope_text": clean(first["scope_text"]),
            "threshold_pct": float(first["threshold_pct"]),
            "threshold_operator": clean(first["threshold_operator"]),
            "members": members,
        }
        for cas in members:
            cas_to_groups.setdefault(cas, set()).add(rid)
    return groups, cas_to_groups

GROUPS, CAS_TO_GROUPS = _load_registry()


def get_rule(rule_id: str) -> dict:
    rid = clean(rule_id)
    if rid not in GROUPS:
        raise KeyError(f"Unknown rule_id: {rid}")
    return GROUPS[rid]


def authoritative_members(rule_id: str) -> set[str]:
    return set(get_rule(rule_id)["members"])


def _smiles_has_element(smiles: str, symbol: str) -> Optional[bool]:
    s = clean(smiles)
    if not s:
        return None
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        return any(a.GetSymbol() == symbol for a in m.GetAtoms())
    except Exception:
        pat = re.compile(rf"(?<![A-Za-z]){re.escape(symbol)}(?![a-z])")
        return bool(pat.search(s))


def group_gate(rule_id: str, candidate_cas: str, candidate_name: str = "", candidate_smiles: str = "") -> tuple[str, str]:
    rid = clean(rule_id)
    cas = normalize_cas(candidate_cas)
    name = clean(candidate_name).lower()
    rule = get_rule(rid)

    if cas and cas in rule["members"]:
        return "MATCH", "OFFICIAL_ANNEX3_ENUMERATED_CAS"

    known_elsewhere = CAS_TO_GROUPS.get(cas, set()) - {rid}
    if cas and known_elsewhere:
        return "NO_MATCH", "OFFICIAL_ANNEX3_CROSS_GROUP_CAS"

    if rid == "LEAD_COMPOUNDS":
        has_pb = _smiles_has_element(candidate_smiles, "Pb")
        if has_pb is True:
            return "MATCH", "CHEMISTRY_FALLBACK_CONTAINS_PB"
        if has_pb is False:
            return "NO_MATCH", "CHEMISTRY_FALLBACK_NO_PB"
        return "REVIEW", "OPEN_WORLD_LEAD_MEMBERSHIP_UNRESOLVED"

    if rid == "CRVI_COMPOUNDS":
        has_cr = _smiles_has_element(candidate_smiles, "Cr")
        positive_terms = ("chromate", "dichromate", "chromic acid", "chromium(vi)", "chromium(6+)", "chromium trioxide", "chromyl")
        if any(t in name for t in positive_terms):
            return "MATCH", "CHEMISTRY_NAME_FALLBACK_CRVI"
        if has_cr is False:
            return "NO_MATCH", "CHEMISTRY_FALLBACK_NO_CR"
        return "REVIEW", "OPEN_WORLD_CR_OXIDATION_STATE_UNRESOLVED"

    if rid == "TBT_TRIALKYLTIN":
        if "tributyltin" in name or "trialkyl tin" in name or "trialkyltin" in name:
            return "MATCH", "CHEMISTRY_NAME_FALLBACK_ORGANOTIN"
        has_sn = _smiles_has_element(candidate_smiles, "Sn")
        if has_sn is False:
            return "NO_MATCH", "CHEMISTRY_FALLBACK_NO_SN"
        return "REVIEW", "OPEN_WORLD_ORGANOTIN_MEMBERSHIP_UNRESOLVED"

    if rid == "NONYLPHENOL_NPE":
        if "nonylphenol" in name or "nonylphenyl" in name:
            return "MATCH", "CHEMISTRY_NAME_FALLBACK_NONYLPHENOL"
        return "REVIEW", "OPEN_WORLD_NONYLPHENOL_MEMBERSHIP_UNRESOLVED"

    if rid == "MG_SALTS":
        if "malachite green" in name:
            return "MATCH", "CHEMISTRY_NAME_FALLBACK_MALACHITE_GREEN"
        return "REVIEW", "OPEN_WORLD_MALACHITE_GREEN_SALT_UNRESOLVED"

    return "REVIEW", "NO_DETERMINISTIC_RULE"


def compare_threshold(concentration_pct: float, threshold_pct: float, operator: str) -> bool:
    c = float(concentration_pct); t = float(threshold_pct); op = clean(operator)
    eps = 1e-12
    if op == ">=":
        return c >= t - eps
    if op == ">":
        return c > t + eps
    raise ValueError(f"Unsupported threshold operator: {op}")


def mixture_gate(rule_id: str, component_cas: str, component_name: str, concentration_pct: float,
                 candidate_smiles: str = "") -> tuple[str, str]:
    g, why = group_gate(rule_id, component_cas, component_name, candidate_smiles)
    if g == "NO_MATCH":
        return "NO_MATCH", f"COMPONENT_OUT_OF_SCOPE:{why}"
    if g == "REVIEW":
        return "REVIEW", f"COMPONENT_SCOPE_REVIEW:{why}"
    rule = get_rule(rule_id)
    ok = compare_threshold(float(concentration_pct), rule["threshold_pct"], rule["threshold_operator"])
    if ok:
        return "MATCH", f"IN_SCOPE_AND_THRESHOLD_MET:{rule['threshold_operator']}{rule['threshold_pct']}"
    return "NO_MATCH", f"IN_SCOPE_BUT_THRESHOLD_NOT_MET:{rule['threshold_operator']}{rule['threshold_pct']}"
