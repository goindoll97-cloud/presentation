# -*- coding: utf-8 -*-
"""
Cheminformatics identity-resolution extension for the chemical regulatory pipeline.

Design principles
-----------------
1. LLM parses regulatory scope; it does not make the final legal decision.
2. Direct CAS / numeric thresholds remain deterministic in regulatory_core.py.
3. This module resolves ONLY explicitly validated extended-identity rules.
4. A structure/composition rule is eligible for automatic routing only when:
      validated == True AND automation_approved == True
5. Reaction products, UVCB, and vague derivatives remain REVIEW_REQUIRED unless
   an expert has converted the legal scope into an explicit deterministic rule.
6. A cheminformatics result is a reproducible screening candidate, not a legal opinion.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
except Exception:  # pragma: no cover
    Chem = None
    rdMolStandardize = None


ROOT = Path(__file__).resolve().parent
RULE_SPEC_FILE = os.getenv("REGULATORY_IDENTITY_RULES_FILE", "").strip()
MIXTURE_FILE = os.getenv("COMPANY_MIXTURE_CONSTITUENTS_FILE", "").strip()
VALIDATION_FILE = os.getenv("IDENTITY_RESOLUTION_VALIDATION_FILE", "").strip()

RULE_MODES = {
    "SALT_PARENT",
    "SUBSTRUCTURE",
    "STRUCTURAL_RANGE",
    "MIXTURE_COMPONENT",
    "REACTION_MASS",
    "MANUAL_REVIEW",
}

AUTO_RULE_MODES = {
    "SALT_PARENT",
    "SUBSTRUCTURE",
    "STRUCTURAL_RANGE",
    "MIXTURE_COMPONENT",
    "REACTION_MASS",
}

RULE_SPEC_COLUMNS = [
    "notice",
    "rule_id",
    "designation_id",
    "scope_type",
    "rule_mode",
    "reference_smiles",
    "required_smarts",
    "excluded_smarts",
    "min_carbon",
    "max_carbon",
    "carbon_metric",
    "required_constituent_cas",
    "any_constituent_cas",
    "min_constituent_concentration_pct",
    "total_matched_constituent_min_pct",
    "min_required_constituents",
    "validated",
    "automation_approved",
    "evidence_source",
    "notes",
]

VALIDATION_COLUMNS = [
    "case_id", "notice", "inventory_id", "chemical_name", "cas",
    "concentration_pct", "smiles", "rule_id", "designation_id",
    "expert_identity_membership", "expert_regulated", "evidence_source",
]

MIXTURE_COLUMNS = [
    "inventory_id",
    "product_name",
    "constituent_name",
    "constituent_cas",
    "constituent_concentration_pct",
    "constituent_smiles",
    "source",
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


def as_bool(x) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return clean(x).lower() in {"1", "true", "yes", "y", "validated", "approved", "예"}


def split_values(x) -> list[str]:
    s = clean(x)
    if not s:
        return []
    return [z.strip() for z in re.split(r"[;|]", s) if z.strip()]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _find_input(explicit: str, names: Iterable[str]) -> Optional[Path]:
    cands = []
    if explicit:
        cands.append(Path(explicit))
    for name in names:
        cands.extend([ROOT / name, Path.cwd() / name])
    seen = set()
    for p in cands:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists() and p.is_file():
            return p
    return None


def rule_spec_template() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "notice": "2026-5",
            "rule_id": "",
            "designation_id": "example_designation",
            "scope_type": "BROAD_SALT_SCOPE",
            "rule_mode": "SALT_PARENT",
            "reference_smiles": "O=C(O)c1ccccc1",
            "required_smarts": "",
            "excluded_smarts": "",
            "min_carbon": "",
            "max_carbon": "",
            "carbon_metric": "TOTAL_CARBON",
            "required_constituent_cas": "",
            "any_constituent_cas": "",
            "min_constituent_concentration_pct": "",
            "total_matched_constituent_min_pct": "",
            "min_required_constituents": "",
            "validated": 0,
            "automation_approved": 0,
            "evidence_source": "",
            "notes": "Example only. Set validated/automation_approved only after expert review.",
        },
        {
            "notice": "2026-5",
            "rule_id": "",
            "designation_id": "example_structural_range",
            "scope_type": "STRUCTURAL_RANGE",
            "rule_mode": "STRUCTURAL_RANGE",
            "reference_smiles": "",
            "required_smarts": "[CX4]",
            "excluded_smarts": "",
            "min_carbon": 12,
            "max_carbon": 14,
            "carbon_metric": "LONGEST_CARBON_PATH",
            "required_constituent_cas": "",
            "any_constituent_cas": "",
            "min_constituent_concentration_pct": "",
            "total_matched_constituent_min_pct": "",
            "min_required_constituents": "",
            "validated": 0,
            "automation_approved": 0,
            "evidence_source": "",
            "notes": "SMARTS/carbon rule must be derived from the official scope and validated.",
        },
        {
            "notice": "2026-5",
            "rule_id": "",
            "designation_id": "example_mixture",
            "scope_type": "MIXTURE_OR_REACTION",
            "rule_mode": "MIXTURE_COMPONENT",
            "reference_smiles": "",
            "required_smarts": "",
            "excluded_smarts": "",
            "min_carbon": "",
            "max_carbon": "",
            "carbon_metric": "",
            "required_constituent_cas": "123-45-6",
            "any_constituent_cas": "",
            "min_constituent_concentration_pct": 0.1,
            "total_matched_constituent_min_pct": "",
            "min_required_constituents": 1,
            "validated": 0,
            "automation_approved": 0,
            "evidence_source": "",
            "notes": "Use only for an explicit composition-based regulatory rule, not a generic reaction product.",
        },
    ])


def mixture_template() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "inventory_id": "INV-001",
            "product_name": "Example product",
            "constituent_name": "Example constituent A",
            "constituent_cas": "123-45-6",
            "constituent_concentration_pct": 20.0,
            "constituent_smiles": "",
            "source": "SDS section 3 / supplier composition",
        },
        {
            "inventory_id": "INV-001",
            "product_name": "Example product",
            "constituent_name": "Example constituent B",
            "constituent_cas": "111-22-3",
            "constituent_concentration_pct": 80.0,
            "constituent_smiles": "",
            "source": "SDS section 3 / supplier composition",
        },
    ])


def load_rule_specs() -> tuple[pd.DataFrame, str]:
    p = _find_input(
        RULE_SPEC_FILE,
        ["regulatory_identity_rules.xlsx", "regulatory_identity_rules.csv"],
    )
    if p is None:
        return pd.DataFrame(columns=RULE_SPEC_COLUMNS), "INPUT_REQUIRED"
    df = read_table(p)
    for c in RULE_SPEC_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[RULE_SPEC_COLUMNS].copy()
    df["rule_mode"] = df["rule_mode"].map(lambda x: clean(x).upper())
    df["validated_bool"] = df["validated"].map(as_bool)
    df["automation_approved_bool"] = df["automation_approved"].map(as_bool)
    for c in [
        "min_carbon", "max_carbon", "min_constituent_concentration_pct",
        "total_matched_constituent_min_pct", "min_required_constituents",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["input_source"] = str(p)
    return df, "LOADED"


def validation_template() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "case_id": "VAL-001",
            "notice": "2026-5",
            "inventory_id": "VAL_INV_001",
            "chemical_name": "Sodium benzoate example",
            "cas": "532-32-1",
            "concentration_pct": 5.0,
            "smiles": "[Na+].[O-]C(=O)c1ccccc1",
            "rule_id": "",
            "designation_id": "example_designation",
            "expert_identity_membership": "Y",
            "expert_regulated": "",
            "evidence_source": "Independent expert adjudication / curated reference",
        }
    ])


def load_validation_cases() -> tuple[pd.DataFrame, str]:
    p = _find_input(
        VALIDATION_FILE,
        ["identity_resolution_validation_cases.xlsx", "identity_resolution_validation_cases.csv"],
    )
    if p is None:
        return pd.DataFrame(columns=VALIDATION_COLUMNS), "INPUT_REQUIRED"
    df = read_table(p)
    for c in VALIDATION_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[VALIDATION_COLUMNS].copy()
    df["concentration_pct"] = pd.to_numeric(df["concentration_pct"], errors="coerce")
    df["input_source"] = str(p)
    return df, "LOADED"


def _spec_is_machine_executable(spec: pd.Series) -> tuple[bool, str]:
    mode = clean(spec.get("rule_mode")).upper()
    if mode == "SALT_PARENT":
        return (bool(parent_smiles(spec.get("reference_smiles"))), "REFERENCE_PARENT_SMILES_REQUIRED")
    if mode == "SUBSTRUCTURE":
        required = clean(spec.get("required_smarts"))
        if not required:
            return False, "REQUIRED_SMARTS_MISSING"
        return (_compile_smarts_list(required) is not None and _compile_smarts_list(clean(spec.get("excluded_smarts"))) is not None, "SMARTS_MUST_PARSE")
    if mode == "STRUCTURAL_RANGE":
        lo = pd.to_numeric(pd.Series([spec.get("min_carbon")]), errors="coerce").iloc[0]
        hi = pd.to_numeric(pd.Series([spec.get("max_carbon")]), errors="coerce").iloc[0]
        if not np.isfinite(lo) and not np.isfinite(hi):
            return False, "CARBON_RANGE_REQUIRED"
        req = clean(spec.get("required_smarts"))
        exc = clean(spec.get("excluded_smarts"))
        if req and _compile_smarts_list(req) is None:
            return False, "REQUIRED_SMARTS_INVALID"
        if exc and _compile_smarts_list(exc) is None:
            return False, "EXCLUDED_SMARTS_INVALID"
        return True, "STRUCTURAL_RANGE_EXECUTABLE"
    if mode in {"MIXTURE_COMPONENT", "REACTION_MASS"}:
        targets = set(split_values(spec.get("required_constituent_cas"))) | set(split_values(spec.get("any_constituent_cas")))
        if not targets:
            return False, "CONSTITUENT_CAS_CRITERIA_REQUIRED"
        if mode == "REACTION_MASS":
            min_n = pd.to_numeric(pd.Series([spec.get("min_required_constituents")]), errors="coerce").iloc[0]
            if len(targets) < 2 and not (np.isfinite(min_n) and min_n >= 2):
                return False, "REACTION_MASS_REQUIRES_MULTI_CONSTITUENT_DEFINITION"
        return True, "COMPOSITION_RULE_EXECUTABLE"
    return False, "MANUAL_OR_UNSUPPORTED_RULE_MODE"


def load_mixture_constituents() -> tuple[pd.DataFrame, str]:
    p = _find_input(
        MIXTURE_FILE,
        ["company_mixture_constituents.xlsx", "company_mixture_constituents.csv"],
    )
    if p is None:
        return pd.DataFrame(columns=MIXTURE_COLUMNS), "INPUT_REQUIRED"
    df = read_table(p)
    for c in MIXTURE_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[MIXTURE_COLUMNS].copy()
    df["constituent_cas"] = df["constituent_cas"].map(clean)
    df["constituent_concentration_pct"] = pd.to_numeric(
        df["constituent_concentration_pct"], errors="coerce"
    )
    df["input_source"] = str(p)
    return df, "LOADED"


def _mol(smiles: str):
    if Chem is None:
        return None
    s = clean(smiles)
    if not s:
        return None
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None


def parent_smiles(smiles: str) -> str:
    """RDKit FragmentParent + uncharge; identity candidate only."""
    mol = _mol(smiles)
    if mol is None or rdMolStandardize is None:
        return ""
    try:
        parent = rdMolStandardize.FragmentParent(mol)
        try:
            parent = rdMolStandardize.Uncharger().uncharge(parent)
        except Exception:
            pass
        return Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def canonical_smiles(smiles: str) -> str:
    mol = _mol(smiles)
    if mol is None:
        return ""
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def total_carbon_count(mol) -> int:
    if mol is None:
        return 0
    return int(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6))


def longest_carbon_path(mol) -> int:
    """Carbon-only graph diameter + 1. Useful as a deterministic chain proxy."""
    if mol is None:
        return 0
    carbons = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
    if not carbons:
        return 0
    cset = set(carbons)
    adj = {i: [] for i in carbons}
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in cset and b in cset:
            adj[a].append(b)
            adj[b].append(a)
    best = 1
    for start in carbons:
        dist = {start: 0}
        q = [start]
        for u in q:
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        if dist:
            best = max(best, max(dist.values()) + 1)
    return int(best)


def _compile_smarts_list(text: str):
    pats = []
    for s in split_values(text):
        if Chem is None:
            return None
        try:
            m = Chem.MolFromSmarts(s)
        except Exception:
            m = None
        if m is None:
            return None
        pats.append((s, m))
    return pats


def _smarts_decision(mol, required: str, excluded: str) -> tuple[Optional[bool], str]:
    req = _compile_smarts_list(required)
    exc = _compile_smarts_list(excluded)
    if req is None or exc is None:
        return None, "INVALID_SMARTS_RULE"
    if mol is None:
        return None, "INVALID_OR_MISSING_SMILES"
    for _, patt in req:
        if not mol.HasSubstructMatch(patt):
            return False, "REQUIRED_SMARTS_NOT_FOUND"
    for _, patt in exc:
        if mol.HasSubstructMatch(patt):
            return False, "EXCLUDED_SMARTS_FOUND"
    return True, "SMARTS_RULE_SATISFIED"


def _rule_reference_rows(semantic_ref: pd.DataFrame, spec: pd.Series) -> pd.DataFrame:
    x = semantic_ref.copy()
    rid = clean(spec.get("rule_id"))
    did = clean(spec.get("designation_id"))
    notice = clean(spec.get("notice"))
    if rid and "rule_id" in x.columns:
        x = x[x["rule_id"].astype(str).eq(rid)]
    elif did and "designation_id" in x.columns:
        x = x[x["designation_id"].astype(str).eq(did)]
    else:
        return x.head(0)
    if notice and "notice" in x.columns:
        x = x[x["notice"].astype(str).eq(notice)]
    return x


def _spec_for_notice(specs: pd.DataFrame, notice: str) -> pd.DataFrame:
    if specs is None or not len(specs):
        return pd.DataFrame(columns=list(specs.columns) if isinstance(specs, pd.DataFrame) else RULE_SPEC_COLUMNS)
    n = clean(notice)
    s = specs.copy()
    if "notice" in s.columns and n:
        sn = s["notice"].fillna("").astype(str).str.strip()
        s = s[(sn.eq("")) | (sn.eq(n))].copy()
    return s


def _thresholds_from_rule_rows(rows: pd.DataFrame) -> dict[str, float]:
    out = {}
    mapping = {
        "acute": "acute_threshold_pct",
        "chronic": "chronic_threshold_pct",
        "eco": "eco_threshold_pct",
    }
    for label, col in mapping.items():
        if col in rows.columns:
            vals = pd.to_numeric(rows[col], errors="coerce").dropna()
            out[label] = float(vals.min()) if len(vals) else np.nan
        else:
            out[label] = np.nan
    vals = [v for v in out.values() if np.isfinite(v)]
    out["min"] = min(vals) if vals else np.nan
    return out


def _trigger_categories(conc: float, thresholds: dict[str, float]) -> str:
    if not np.isfinite(conc):
        return ""
    hits = []
    for cat in ["acute", "chronic", "eco"]:
        t = thresholds.get(cat, np.nan)
        if np.isfinite(t) and float(conc) >= float(t):
            hits.append(cat)
    return ";".join(hits)


def _inventory_concentration(inv: pd.Series) -> float:
    try:
        return float(pd.to_numeric(pd.Series([inv.get("concentration_pct")]), errors="coerce").iloc[0])
    except Exception:
        return np.nan


def _evaluate_structure_rule(inv: pd.Series, spec: pd.Series) -> tuple[str, str, float, dict]:
    mode = clean(spec.get("rule_mode")).upper()
    smi = clean(inv.get("smiles"))
    mol = _mol(smi)
    extra = {}

    if Chem is None:
        return "REVIEW_REQUIRED", "RDKIT_NOT_AVAILABLE", np.nan, extra

    if mode == "SALT_PARENT":
        ref = clean(spec.get("reference_smiles"))
        inv_parent = parent_smiles(smi)
        ref_parent = parent_smiles(ref)
        extra.update({"inventory_parent_smiles": inv_parent, "reference_parent_smiles": ref_parent})
        if not inv_parent or not ref_parent:
            return "REVIEW_REQUIRED", "MISSING_OR_INVALID_PARENT_STRUCTURE", np.nan, extra
        return (
            "CHEM_MATCH" if inv_parent == ref_parent else "CHEM_NO_MATCH",
            "PARENT_STRUCTURE_EQUAL" if inv_parent == ref_parent else "PARENT_STRUCTURE_DIFFERENT",
            _inventory_concentration(inv),
            extra,
        )

    if mode in {"SUBSTRUCTURE", "STRUCTURAL_RANGE"}:
        if mode == "SUBSTRUCTURE" and not clean(spec.get("required_smarts")):
            return "REVIEW_REQUIRED", "REQUIRED_SMARTS_MISSING", np.nan, extra
        ok, reason = _smarts_decision(mol, clean(spec.get("required_smarts")), clean(spec.get("excluded_smarts")))
        if ok is None:
            return "REVIEW_REQUIRED", reason, np.nan, extra
        if not ok:
            return "CHEM_NO_MATCH", reason, _inventory_concentration(inv), extra

        if mode == "STRUCTURAL_RANGE":
            metric = clean(spec.get("carbon_metric")).upper() or "TOTAL_CARBON"
            count = longest_carbon_path(mol) if metric == "LONGEST_CARBON_PATH" else total_carbon_count(mol)
            lo = pd.to_numeric(pd.Series([spec.get("min_carbon")]), errors="coerce").iloc[0]
            hi = pd.to_numeric(pd.Series([spec.get("max_carbon")]), errors="coerce").iloc[0]
            extra.update({"carbon_metric": metric, "observed_carbon_metric": count})
            if not np.isfinite(lo) and not np.isfinite(hi):
                return "REVIEW_REQUIRED", "STRUCTURAL_RANGE_BOUNDS_MISSING", np.nan, extra
            if np.isfinite(lo) and count < float(lo):
                return "CHEM_NO_MATCH", "CARBON_RANGE_BELOW_MIN", _inventory_concentration(inv), extra
            if np.isfinite(hi) and count > float(hi):
                return "CHEM_NO_MATCH", "CARBON_RANGE_ABOVE_MAX", _inventory_concentration(inv), extra
            return "CHEM_MATCH", "SMARTS_AND_CARBON_RANGE_SATISFIED", _inventory_concentration(inv), extra

        return "CHEM_MATCH", "SUBSTRUCTURE_RULE_SATISFIED", _inventory_concentration(inv), extra

    return "REVIEW_REQUIRED", "STRUCTURE_MODE_NOT_SUPPORTED", np.nan, extra


def _evaluate_composition_rule(
    inv: pd.Series,
    spec: pd.Series,
    constituents: pd.DataFrame,
) -> tuple[str, str, float, dict]:
    iid = clean(inv.get("inventory_id"))
    g = constituents[constituents["inventory_id"].astype(str).eq(iid)].copy() if len(constituents) else pd.DataFrame()
    if not len(g):
        return "REVIEW_REQUIRED", "MISSING_PRODUCT_COMPOSITION", np.nan, {"matched_constituent_n": 0}

    g["constituent_cas"] = g["constituent_cas"].map(clean)
    g["constituent_concentration_pct"] = pd.to_numeric(g["constituent_concentration_pct"], errors="coerce")
    req = set(split_values(spec.get("required_constituent_cas")))
    anyset = set(split_values(spec.get("any_constituent_cas")))
    min_each = pd.to_numeric(pd.Series([spec.get("min_constituent_concentration_pct")]), errors="coerce").iloc[0]
    min_total = pd.to_numeric(pd.Series([spec.get("total_matched_constituent_min_pct")]), errors="coerce").iloc[0]
    min_n = pd.to_numeric(pd.Series([spec.get("min_required_constituents")]), errors="coerce").iloc[0]

    def qualifying(cas: str) -> pd.DataFrame:
        z = g[g["constituent_cas"].eq(cas)].copy()
        if np.isfinite(min_each):
            z = z[z["constituent_concentration_pct"] >= float(min_each)]
        return z

    missing_req = [cas for cas in req if len(qualifying(cas)) == 0]
    if missing_req:
        return "CHEM_NO_MATCH", "REQUIRED_CONSTITUENT_MISSING", 0.0, {"missing_required_constituent_cas": ";".join(sorted(missing_req))}

    any_hits = []
    if anyset:
        any_hits = [cas for cas in anyset if len(qualifying(cas))]
        if not any_hits:
            return "CHEM_NO_MATCH", "NO_ANY_OF_CONSTITUENTS_PRESENT", 0.0, {}

    target_set = req | anyset
    if target_set:
        matched = g[g["constituent_cas"].isin(target_set)].copy()
        if np.isfinite(min_each):
            matched = matched[matched["constituent_concentration_pct"] >= float(min_each)]
    else:
        return "REVIEW_REQUIRED", "CONSTITUENT_RULE_HAS_NO_CAS_CRITERIA", np.nan, {}

    matched_n = int(matched["constituent_cas"].nunique())
    total_pct = float(matched["constituent_concentration_pct"].sum()) if matched["constituent_concentration_pct"].notna().any() else np.nan

    if np.isfinite(min_n) and matched_n < int(min_n):
        return "CHEM_NO_MATCH", "INSUFFICIENT_REQUIRED_CONSTITUENT_COUNT", total_pct, {"matched_constituent_n": matched_n}
    if np.isfinite(min_total) and (not np.isfinite(total_pct) or total_pct < float(min_total)):
        return "CHEM_NO_MATCH", "MATCHED_CONSTITUENT_TOTAL_BELOW_MIN", total_pct, {"matched_constituent_n": matched_n}

    return "CHEM_MATCH", "COMPOSITION_RULE_SATISFIED", total_pct, {
        "matched_constituent_n": matched_n,
        "matched_constituent_cas": ";".join(sorted(set(matched["constituent_cas"]))),
        "matched_constituent_total_pct": total_pct,
    }


def build_rule_capability(
    semantic_ref: pd.DataFrame,
    scored: pd.DataFrame,
    rule_specs: pd.DataFrame,
    notice: str = "",
) -> pd.DataFrame:
    """Rule-level potential after adding validated cheminformatics/composition rules."""
    if semantic_ref is None or not len(semantic_ref):
        return pd.DataFrame()

    x = semantic_ref.copy()
    if notice and "notice" in x.columns:
        x = x[x["notice"].astype(str).eq(str(notice))].copy()

    ai_cols = [c for c in ["rule_id", "ai_status", "ai_scope_type"] if scored is not None and c in scored.columns]
    if scored is not None and len(scored) and ai_cols:
        ai = scored[ai_cols].drop_duplicates("rule_id")
        x = x.merge(ai, on="rule_id", how="left", suffixes=("", "_ai"))

    specs = _spec_for_notice(rule_specs, notice)
    rows = []

    for _, r in x.iterrows():
        rid = clean(r.get("rule_id"))
        did = clean(r.get("designation_id"))
        has_direct = bool(r.get("has_direct_cas"))

        if has_direct:
            rows.append({
                "rule_id": rid, "designation_id": did,
                "ref_scope_type": clean(r.get("ref_scope_type")),
                "capability_route": "DETERMINISTIC_AUTO_CANDIDATE",
                "capability_reason": "Direct CAS available; existing deterministic engine applies.",
                "rule_mode": "DIRECT_CAS", "validated_rule": True,
                "automation_approved": True, "machine_executable": True,
                "machine_executable_checks": "DIRECT_CAS", "evidence_source": "REGULATORY_MASTER",
            })
            continue

        if len(specs):
            s = specs[(specs["rule_id"].astype(str).eq(rid) & specs["rule_id"].astype(str).ne(""))]
            if not len(s):
                s = specs[specs["designation_id"].astype(str).eq(did)]
        else:
            s = pd.DataFrame()

        if not len(s):
            route = "AI_ASSISTED_PARENT_SCOPE_CANDIDATE" if (
                clean(r.get("ai_status")) == "OK" and clean(r.get("ai_scope_type")) == "BROAD_SALT_SCOPE"
            ) else "REVIEW_REQUIRED"
            rows.append({
                "rule_id": rid, "designation_id": did,
                "ref_scope_type": clean(r.get("ref_scope_type")),
                "capability_route": route,
                "capability_reason": "No validated deterministic extended-identity rule supplied.",
                "rule_mode": "", "validated_rule": False,
                "automation_approved": False, "machine_executable": False,
                "machine_executable_checks": "NO_RULE_SPEC", "evidence_source": "",
            })
            continue

        # Conservative: if any matched rule spec is not approved, do not call it fully automatic.
        executable_flags = s.apply(lambda rr: _spec_is_machine_executable(rr)[0], axis=1).astype(bool)
        executable_reasons = s.apply(lambda rr: _spec_is_machine_executable(rr)[1], axis=1).astype(str)
        approved = s["validated_bool"].astype(bool) & s["automation_approved_bool"].astype(bool) & s["rule_mode"].isin(AUTO_RULE_MODES) & executable_flags
        validated = s["validated_bool"].astype(bool)
        if approved.all() and len(s):
            modes = set(s["rule_mode"])
            if modes.issubset({"MIXTURE_COMPONENT", "REACTION_MASS"}):
                route = "COMPOSITION_AUTO_CANDIDATE"
            else:
                route = "CHEMINFORMATICS_AUTO_CANDIDATE"
            reason = "All applicable extended-identity rule specifications are validated and automation-approved."
        elif validated.any():
            route = "CHEMINFORMATICS_ASSISTED_REVIEW_CANDIDATE"
            reason = "A validated rule specification exists, but automatic use has not been approved for all matched specifications."
        else:
            route = "REVIEW_REQUIRED"
            reason = "Rule specification exists but is not validated."

        rows.append({
            "rule_id": rid, "designation_id": did,
            "ref_scope_type": clean(r.get("ref_scope_type")),
            "capability_route": route,
            "capability_reason": reason,
            "rule_mode": ";".join(sorted(set(s["rule_mode"].astype(str)))),
            "validated_rule": bool(validated.any()),
            "automation_approved": bool(approved.all()) if len(approved) else False,
            "machine_executable": bool(executable_flags.all()) if len(executable_flags) else False,
            "machine_executable_checks": ";".join(sorted(set(executable_reasons))),
            "evidence_source": ";".join(sorted(set(s["evidence_source"].map(clean)) - {""})),
        })

    return pd.DataFrame(rows)


def summarize_capability(capability: pd.DataFrame) -> pd.DataFrame:
    if capability is None or not len(capability):
        return pd.DataFrame()
    x = capability["capability_route"].value_counts().rename_axis("capability_route").reset_index(name="n_rules")
    x["pct_rules"] = x["n_rules"] / len(capability)
    return pd.concat([x, pd.DataFrame([{"capability_route": "TOTAL", "n_rules": len(capability), "pct_rules": 1.0}])], ignore_index=True)


def capability_by_scope(capability: pd.DataFrame) -> pd.DataFrame:
    if capability is None or not len(capability):
        return pd.DataFrame()
    return (
        capability.groupby(["ref_scope_type", "capability_route"], dropna=False)
        .size().reset_index(name="n_rules")
        .merge(capability.groupby("ref_scope_type").size().rename("scope_n"), on="ref_scope_type", how="left")
        .assign(pct_within_scope=lambda d: d["n_rules"] / d["scope_n"])
    )


def evaluate_inventory_extended_scope(
    inventory: pd.DataFrame,
    semantic_ref: pd.DataFrame,
    rule_specs: pd.DataFrame,
    constituents: pd.DataFrame,
    notice: str = "",
) -> pd.DataFrame:
    """Inventory x validated extended-scope rules. Does not duplicate direct-CAS rules."""
    if inventory is None or not len(inventory):
        return pd.DataFrame()
    specs = _spec_for_notice(rule_specs, notice)
    if specs is None or not len(specs):
        return pd.DataFrame()
    specs = specs[specs["validated_bool"].astype(bool)].copy()
    if not len(specs):
        return pd.DataFrame()

    rows = []
    for _, spec in specs.iterrows():
        mode = clean(spec.get("rule_mode")).upper()
        rid = clean(spec.get("rule_id"))
        did = clean(spec.get("designation_id"))
        ref_rows = _rule_reference_rows(semantic_ref, spec)
        thresholds = _thresholds_from_rule_rows(ref_rows)
        scope = clean(spec.get("scope_type")) or (clean(ref_rows.iloc[0].get("ref_scope_type")) if len(ref_rows) else "")

        # Explicitly do not infer generic reaction products / UVCB from SMILES alone.
        if mode == "MANUAL_REVIEW" or mode not in RULE_MODES:
            for _, inv in inventory.iterrows():
                rows.append({
                    "inventory_id": clean(inv.get("inventory_id")), "product_name": clean(inv.get("product_name")),
                    "chemical_name": clean(inv.get("chemical_name")), "inventory_cas": clean(inv.get("cas")),
                    "designation_id": did, "rule_id": rid, "scope_type": scope, "rule_mode": mode,
                    "identity_result": "REVIEW_REQUIRED", "identity_reason": "MANUAL_OR_UNSUPPORTED_RULE_MODE",
                    "automation_eligible": False, "matched_concentration_pct": np.nan,
                    "rule_min_threshold_pct": thresholds["min"], "triggered_categories": "",
                    "screening_status": "REVIEW_REQUIRED", "screening_triggered": None,
                    "evidence_source": clean(spec.get("evidence_source")),
                })
            continue

        for _, inv in inventory.iterrows():
            if mode in {"SALT_PARENT", "SUBSTRUCTURE", "STRUCTURAL_RANGE"}:
                identity, reason, matched_conc, extra = _evaluate_structure_rule(inv, spec)
            elif mode in {"MIXTURE_COMPONENT", "REACTION_MASS"}:
                identity, reason, matched_conc, extra = _evaluate_composition_rule(inv, spec, constituents)
            else:
                identity, reason, matched_conc, extra = "REVIEW_REQUIRED", "UNSUPPORTED_RULE_MODE", np.nan, {}

            approved = bool(spec.get("automation_approved_bool"))
            auto_eligible = approved and identity in {"CHEM_MATCH", "CHEM_NO_MATCH"}
            triggered_categories = _trigger_categories(matched_conc, thresholds) if identity == "CHEM_MATCH" else ""

            if identity == "CHEM_NO_MATCH":
                status, trig = "EXTENDED_IDENTITY_NO_MATCH", False
            elif identity == "REVIEW_REQUIRED":
                status, trig = "REVIEW_REQUIRED", None
            elif not approved:
                status, trig = "REVIEW_REQUIRED_VALIDATED_RULE_NOT_AUTOMATION_APPROVED", None
            elif not np.isfinite(thresholds["min"]):
                status, trig = "REVIEW_MISSING_REGULATORY_THRESHOLD", None
            elif not np.isfinite(matched_conc):
                status, trig = "REVIEW_MISSING_MATCHED_CONCENTRATION", None
            elif float(matched_conc) >= float(thresholds["min"]):
                status, trig = "EXTENDED_IDENTITY_THRESHOLD_TRIGGER_CANDIDATE", True
            else:
                status, trig = "EXTENDED_IDENTITY_BELOW_THRESHOLD", False

            row = {
                "inventory_id": clean(inv.get("inventory_id")),
                "product_name": clean(inv.get("product_name")),
                "chemical_name": clean(inv.get("chemical_name")),
                "inventory_cas": clean(inv.get("cas")),
                "inventory_smiles": clean(inv.get("smiles")),
                "designation_id": did,
                "rule_id": rid,
                "scope_type": scope,
                "rule_mode": mode,
                "identity_result": identity,
                "identity_reason": reason,
                "automation_eligible": auto_eligible,
                "matched_concentration_pct": matched_conc,
                "rule_min_threshold_pct": thresholds["min"],
                "acute_threshold_pct": thresholds["acute"],
                "chronic_threshold_pct": thresholds["chronic"],
                "eco_threshold_pct": thresholds["eco"],
                "triggered_categories": triggered_categories,
                "screening_status": status,
                "screening_triggered": trig,
                "evidence_source": clean(spec.get("evidence_source")),
                "screening_interpretation": "REPRODUCIBLE_SCREENING_CANDIDATE_NOT_LEGAL_OPINION",
            }
            row.update(extra)
            rows.append(row)

    return pd.DataFrame(rows)


def extended_to_core_rows(extended: pd.DataFrame, notice: str) -> pd.DataFrame:
    if extended is None or not len(extended):
        return pd.DataFrame()
    x = extended.copy()
    x["assessment_notice"] = notice
    x["matching_mode"] = "CHEMINFORMATICS_OR_COMPOSITION_RULE"
    x["relation_type"] = x.get("rule_mode", "")
    x["rule_substance_name"] = ""
    x["rule_threshold_pct"] = pd.to_numeric(x.get("rule_min_threshold_pct"), errors="coerce")
    x["source_text"] = ""
    if "cas" not in x.columns:
        x["cas"] = x.get("inventory_cas", "")
    keep = [
        "inventory_id", "product_name", "chemical_name", "cas",
        "assessment_notice", "designation_id", "matching_mode", "relation_type",
        "rule_substance_name", "rule_threshold_pct", "screening_status",
        "screening_triggered", "source_text",
    ]
    for c in keep:
        if c not in x.columns:
            x[c] = ""
    return x[keep].copy()


def extended_review_queue(extended: pd.DataFrame) -> pd.DataFrame:
    if extended is None or not len(extended):
        return pd.DataFrame()
    x = extended[extended["screening_status"].astype(str).str.startswith("REVIEW")].copy()
    if not len(x):
        return x
    x["review_reason"] = x["identity_reason"].astype(str) + ";" + x["screening_status"].astype(str)
    x["recommended_resolution"] = (
        "Verify regulatory scope definition, structure/composition evidence, and enterprise data; "
        "approve the deterministic rule only after expert validation."
    )
    x["review_status"] = "PENDING_HUMAN_REVIEW"
    return x


def summarize_extended_screening(extended: pd.DataFrame) -> pd.DataFrame:
    if extended is None or not len(extended):
        return pd.DataFrame()
    x = (
        extended.groupby(["scope_type", "rule_mode", "screening_status"], dropna=False)
        .size().reset_index(name="n_cases")
    )
    total = len(extended)
    x["pct_all_extended_cases"] = x["n_cases"] / total
    return x


def _yn_value(x):
    s = clean(x).lower()
    if s in {"y", "yes", "1", "true", "예"}:
        return True
    if s in {"n", "no", "0", "false", "아니오"}:
        return False
    return None


def benchmark_identity_resolution(
    validation_cases: pd.DataFrame,
    semantic_ref: pd.DataFrame,
    rule_specs: pd.DataFrame,
    constituents: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Optional independent/expert validation of extended identity membership.

    The benchmark is deliberately selective: REVIEW_REQUIRED is treated as abstention,
    not as a wrong automatic answer. Report both coverage and selective accuracy.
    """
    if validation_cases is None or not len(validation_cases):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    details = []
    for _, case in validation_cases.iterrows():
        gold = _yn_value(case.get("expert_identity_membership"))
        if gold is None:
            continue
        inv = pd.DataFrame([{
            "inventory_id": clean(case.get("inventory_id")) or clean(case.get("case_id")),
            "product_name": "",
            "chemical_name": clean(case.get("chemical_name")),
            "cas": clean(case.get("cas")),
            "concentration_pct": case.get("concentration_pct"),
            "smiles": clean(case.get("smiles")),
        }])
        specs = rule_specs.copy()
        rid, did = clean(case.get("rule_id")), clean(case.get("designation_id"))
        if rid:
            specs = specs[specs["rule_id"].astype(str).eq(rid)].copy()
        elif did:
            specs = specs[specs["designation_id"].astype(str).eq(did)].copy()
        else:
            continue
        notice = clean(case.get("notice"))
        result = evaluate_inventory_extended_scope(inv, semantic_ref, specs, constituents, notice)
        if not len(result):
            pred, status, mode, scope, reason = None, "REVIEW_REQUIRED", "", "", "NO_EVALUABLE_VALIDATED_RULE"
        else:
            rr = result.iloc[0]
            ident = clean(rr.get("identity_result"))
            pred = True if ident == "CHEM_MATCH" else (False if ident == "CHEM_NO_MATCH" else None)
            status = clean(rr.get("screening_status"))
            mode = clean(rr.get("rule_mode"))
            scope = clean(rr.get("scope_type"))
            reason = clean(rr.get("identity_reason"))
        details.append({
            **case.to_dict(),
            "gold_identity_membership": gold,
            "pred_identity_membership": pred,
            "abstained": pred is None,
            "correct_if_decided": (pred == gold) if pred is not None else np.nan,
            "rule_mode_resolved": mode,
            "scope_type_resolved": scope,
            "resolver_status": status,
            "resolver_reason": reason,
        })
    d = pd.DataFrame(details)
    if not len(d):
        return d, pd.DataFrame(), pd.DataFrame()

    def metrics(g):
        decided = g[g["abstained"].eq(False)].copy()
        gold = g["gold_identity_membership"].astype(bool)
        pred_all = g["pred_identity_membership"]
        if len(decided):
            dg = decided["gold_identity_membership"].astype(bool)
            dp = decided["pred_identity_membership"].astype(bool)
            tp = int((dg & dp).sum()); tn = int((~dg & ~dp).sum())
            fp = int((~dg & dp).sum()); fn = int((dg & ~dp).sum())
            acc = float((tp + tn) / len(decided))
        else:
            tp = tn = fp = fn = 0; acc = np.nan
        return pd.Series({
            "n_cases": len(g),
            "n_decided": len(decided),
            "automation_coverage": len(decided) / len(g) if len(g) else np.nan,
            "review_burden": 1 - (len(decided) / len(g)) if len(g) else np.nan,
            "selective_accuracy": acc,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        })

    overall = metrics(d).to_frame().T
    overall.insert(0, "group", "OVERALL")
    by = []
    for (scope, mode), g in d.groupby(["scope_type_resolved", "rule_mode_resolved"], dropna=False):
        m = metrics(g).to_dict(); m.update({"scope_type": scope, "rule_mode": mode}); by.append(m)
    return d, overall, pd.DataFrame(by)


ROUTE_RANK = {
    "DETERMINISTIC_AUTO_CANDIDATE": 1,
    "CHEMINFORMATICS_AUTO_CANDIDATE": 2,
    "COMPOSITION_AUTO_CANDIDATE": 2,
    "AI_ASSISTED_PARENT_SCOPE_CANDIDATE": 3,
    "CHEMINFORMATICS_ASSISTED_REVIEW_CANDIDATE": 4,
    "REVIEW_REQUIRED": 5,
}


def upgrade_rule_routing(base_routing: pd.DataFrame, capability: pd.DataFrame) -> pd.DataFrame:
    if base_routing is None or not len(base_routing):
        if capability is None:
            return pd.DataFrame()
        out = capability.copy()
        out["routing_class"] = out["capability_route"]
        out["routing_reason"] = out["capability_reason"]
        return out
    if capability is None or not len(capability):
        return base_routing.copy()

    c = capability[["rule_id", "capability_route", "capability_reason", "rule_mode", "validated_rule", "automation_approved"]].drop_duplicates("rule_id")
    x = base_routing.merge(c, on="rule_id", how="left")

    def choose(r):
        base = clean(r.get("routing_class")) or "REVIEW_REQUIRED"
        cap = clean(r.get("capability_route"))
        if not cap:
            return base, clean(r.get("routing_reason"))
        # Never downgrade a direct deterministic route; otherwise use the safer/better approved capability.
        if base == "DETERMINISTIC_AUTO_CANDIDATE":
            return base, clean(r.get("routing_reason"))
        if ROUTE_RANK.get(cap, 99) < ROUTE_RANK.get(base, 99):
            return cap, clean(r.get("capability_reason"))
        return base, clean(r.get("routing_reason"))

    chosen = x.apply(choose, axis=1)
    x["routing_class"] = [z[0] for z in chosen]
    x["routing_reason"] = [z[1] for z in chosen]
    return x


def summarize_routing(routing: pd.DataFrame) -> pd.DataFrame:
    if routing is None or not len(routing):
        return pd.DataFrame()
    vc = routing["routing_class"].value_counts().rename_axis("routing_class").reset_index(name="n_rules")
    vc["pct_rules"] = vc["n_rules"] / len(routing)
    return pd.concat([vc, pd.DataFrame([{"routing_class": "TOTAL", "n_rules": len(routing), "pct_rules": 1.0}])], ignore_index=True)


def build_update_routing(update_diff: pd.DataFrame, routing: pd.DataFrame) -> pd.DataFrame:
    if update_diff is None or not len(update_diff):
        return pd.DataFrame()
    changed = update_diff[update_diff["update_change_type"].ne("UNCHANGED")].copy()
    if routing is None or not len(routing):
        changed["automation_route"] = "REVIEW_REQUIRED"
        return changed

    rows = []
    for did, g in routing.groupby("designation_id"):
        routes = [clean(z) for z in g["routing_class"] if clean(z)]
        # Conservative designation-level rule: the least automatable sub-rule controls the designation.
        selected = max(routes, key=lambda z: ROUTE_RANK.get(z, 99)) if routes else "REVIEW_REQUIRED"
        rows.append({
            "designation_id": str(did),
            "automation_route": selected,
            "rule_scope_types": ";".join(sorted(set(g.get("ref_scope_type", pd.Series(dtype=str)).astype(str)))),
            "rule_modes": ";".join(sorted(set(g.get("rule_mode", pd.Series(dtype=str)).astype(str))) - {"", "nan"}),
        })
    r = pd.DataFrame(rows)
    out = changed.merge(r, on="designation_id", how="left")
    out["automation_route"] = out["automation_route"].fillna("DETERMINISTIC_AUTO_CANDIDATE")
    return out


def summarize_update_routing(update_routing: pd.DataFrame) -> pd.DataFrame:
    if update_routing is None or not len(update_routing):
        return pd.DataFrame()
    x = update_routing["automation_route"].value_counts().rename_axis("automation_route").reset_index(name="n_changed_designations")
    x["pct_changed_designations"] = x["n_changed_designations"] / len(update_routing)
    return x
