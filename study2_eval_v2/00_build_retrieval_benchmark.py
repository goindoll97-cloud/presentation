# -*- coding: utf-8 -*-
"""Derive a CAS-only retrieval benchmark from frozen Validation V2.

The frozen 132-row benchmark is pairwise (target, candidate, MATCH/NO_MATCH).
This script does NOT alter that benchmark. It derives a second, retrieval-style
evaluation set in which the query contains CAS identifier(s) only and the GOLD
answer is the regulatory target that the candidate belongs to, or NOT_FOUND.

Important methodological safeguards
-----------------------------------
* Regulatory catalog identity fields are built from target/reference fields,
  never from MATCH candidate fields.
* Named-mixture CAS numbers are NOT inferred from GOLD MATCH rows. The primary
  catalog uses the target's official scope text and target component CAS set.
* A derived NOT_FOUND label is stronger than the original pairwise NO_MATCH and
  therefore requires a separate retrieval-scope researcher review before freeze.
* Target-level scope fields must be unique within each target; row-order fallback
  is prohibited.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
VAL_ROOT = REPO_ROOT / "validation_v2"
VAL_DATA = VAL_ROOT / "data"
VAL_FREEZE = VAL_ROOT / "VALIDATION_V2_FREEZE.json"
MASTER_INPUT = VAL_DATA / "validation_master_INPUT.csv"
MASTER_GOLD = VAL_DATA / "validation_master_GOLD.csv"
GROUP_REFERENCE = VAL_ROOT / "regulatory_group_reference.py"

OUT = ROOT / "data"
OUT.mkdir(parents=True, exist_ok=True)
CATALOG = OUT / "regulatory_catalog.csv"
QUERY_INPUT = OUT / "retrieval_INPUT.csv"
QUERY_GOLD = OUT / "retrieval_GOLD.csv"
EXCLUDED = OUT / "retrieval_EXCLUDED_NO_CAS.csv"
MANIFEST = OUT / "retrieval_build_manifest.json"

CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")


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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm_cas(x) -> str:
    return re.sub(r"\s+", "", clean(x))


def valid_cas_format(x) -> bool:
    return bool(CAS_RE.fullmatch(norm_cas(x)))


def split_parts(x) -> list[str]:
    return [clean(v) for v in clean(x).split("|") if clean(v)]


def load_py(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def unique_value(g: pd.DataFrame, column: str, label: str, allow_blank: bool = False) -> str:
    if column not in g.columns:
        if allow_blank:
            return ""
        raise ValueError(f"Missing target field {column}: {label}")
    vals = sorted({clean(v) for v in g[column].tolist() if clean(v)})
    if not vals and allow_blank:
        return ""
    if len(vals) != 1:
        raise ValueError(
            f"Target field must be unique within {label}: {column} -> {vals}"
        )
    return vals[0]


def verify_validation_freeze() -> dict:
    for p in [VAL_FREEZE, MASTER_INPUT, MASTER_GOLD, GROUP_REFERENCE]:
        if not p.exists():
            raise FileNotFoundError(p)
    freeze = json.loads(VAL_FREEZE.read_text(encoding="utf-8"))
    if freeze.get("pre_freeze_audit_status") != "PASS":
        raise RuntimeError("Validation V2 freeze does not report pre-freeze PASS")
    if int(freeze.get("researcher_signoff_approved", 0)) != 45:
        raise RuntimeError("Validation V2 researcher sign-off is incomplete")
    frozen = freeze.get("sha256", {})
    checks = {}
    for rel, path in [
        ("data/validation_master_INPUT.csv", MASTER_INPUT),
        ("data/validation_master_GOLD.csv", MASTER_GOLD),
        ("regulatory_group_reference.py", GROUP_REFERENCE),
    ]:
        expected = clean(frozen.get(rel))
        actual = sha256(path)
        checks[rel] = bool(expected and expected == actual)
    if not all(checks.values()):
        raise RuntimeError(f"Frozen Validation V2 files changed: {checks}")
    return {"freeze_file": str(VAL_FREEZE), "sha_checks": checks}


def target_id_for_row(r: pd.Series) -> str:
    cat = clean(r.get("category"))
    if cat == "PARENT_SALT":
        cas = norm_cas(r.get("reference_parent_cas"))
        if not cas:
            raise ValueError(f"Missing parent CAS for {clean(r.get('case_id'))}")
        return f"PS::{cas}"
    if cat == "CHEMICAL_GROUP":
        rid = clean(r.get("target_rule_id"))
        if not rid:
            raise ValueError(f"Missing target_rule_id for {clean(r.get('case_id'))}")
        return f"CG::{rid}"
    if cat == "MIXTURE":
        sid = clean(r.get("target_scope_id"))
        if not sid:
            raise ValueError(f"Missing target_scope_id for {clean(r.get('case_id'))}")
        return f"MX::{sid}"
    raise ValueError(f"Unknown category: {cat}")


def query_cas_signature(r: pd.Series) -> tuple[str, str]:
    """Return (signature, mode). Empty signature means CAS-only retrieval impossible."""
    cas = norm_cas(r.get("candidate_cas"))
    if cas and valid_cas_format(cas):
        return cas, "SINGLE_CAS"

    if clean(r.get("category")) == "MIXTURE":
        parts = [norm_cas(v) for v in split_parts(r.get("candidate_components"))]
        if parts and all(valid_cas_format(v) for v in parts):
            return "|".join(sorted(set(parts))), "COMPONENT_CAS_SET"

    return "", "NO_CAS_QUERY"


def build_catalog(merged: pd.DataFrame) -> pd.DataFrame:
    group_ref = load_py(GROUP_REFERENCE, "validation_v2_group_reference")
    official = getattr(group_ref, "OFFICIAL_CAS_BY_RULE", {})
    rows = []

    ps = merged[merged["category"].eq("PARENT_SALT")].copy()
    for parent_cas, g in ps.groupby("reference_parent_cas", sort=True):
        parent_cas = norm_cas(parent_cas)
        scope = unique_value(g, "regulatory_scope_text", f"PARENT_SALT:{parent_cas}")
        iso = unique_value(g, "isomer_scope", f"PARENT_SALT:{parent_cas}")
        rows.append({
            "target_id": f"PS::{parent_cas}",
            "target_type": "PARENT_SALT",
            "regulatory_scope_text": scope,
            "reference_cas_set": parent_cas,
            "official_member_cas_set": "",
            "mixture_component_cas_set": "",
            "isomer_scope": iso,
            "catalog_identity_basis": "FROZEN_TARGET_REFERENCE_FIELDS",
        })

    cg = merged[merged["category"].eq("CHEMICAL_GROUP")].copy()
    for rule_id, g in cg.groupby("target_rule_id", sort=True):
        scope = unique_value(g, "regulatory_scope_text", f"CHEMICAL_GROUP:{rule_id}")
        members = sorted(norm_cas(v) for v in official.get(rule_id, []) if norm_cas(v))
        rows.append({
            "target_id": f"CG::{clean(rule_id)}",
            "target_type": "CHEMICAL_GROUP",
            "regulatory_scope_text": scope,
            "reference_cas_set": "",
            "official_member_cas_set": "|".join(members),
            "mixture_component_cas_set": "",
            "isomer_scope": "",
            "catalog_identity_basis": "FROZEN_REGULATORY_GROUP_REFERENCE",
        })

    mx = merged[merged["category"].eq("MIXTURE")].copy()
    for scope_id, g in mx.groupby("target_scope_id", sort=True):
        scope = unique_value(g, "regulatory_scope_text", f"MIXTURE:{scope_id}")
        target_components = unique_value(
            g, "target_components", f"MIXTURE:{scope_id}", allow_blank=True
        )
        parts = [norm_cas(v) for v in split_parts(target_components)]
        component_cas = sorted(set(v for v in parts if valid_cas_format(v)))
        rows.append({
            "target_id": f"MX::{clean(scope_id)}",
            "target_type": "MIXTURE",
            "regulatory_scope_text": scope,
            "reference_cas_set": "",
            "official_member_cas_set": "",
            "mixture_component_cas_set": "|".join(component_cas),
            "isomer_scope": "",
            "catalog_identity_basis": "FROZEN_TARGET_SCOPE_AND_COMPONENT_FIELDS",
        })

    out = pd.DataFrame(rows).sort_values("target_id").reset_index(drop=True)
    if out["target_id"].duplicated().any():
        raise RuntimeError("Duplicate target_id in regulatory catalog")
    return out


def main() -> None:
    freeze_meta = verify_validation_freeze()

    inp = pd.read_csv(MASTER_INPUT, dtype=str).fillna("")
    gold = pd.read_csv(MASTER_GOLD, dtype=str).fillna("")
    if set(inp["case_id"]) != set(gold["case_id"]):
        raise RuntimeError("Master INPUT/GOLD case IDs differ")

    merged = inp.merge(
        gold,
        on=["case_id", "category"],
        how="inner",
        suffixes=("", "_gold"),
        validate="one_to_one",
    )
    if len(merged) != 132:
        raise RuntimeError(f"Expected frozen 132 pairwise cases, got {len(merged)}")

    merged["target_id"] = merged.apply(target_id_for_row, axis=1)
    sig_mode = merged.apply(query_cas_signature, axis=1)
    merged["cas_inputs"] = [x[0] for x in sig_mode]
    merged["query_mode"] = [x[1] for x in sig_mode]

    catalog = build_catalog(merged)
    catalog.to_csv(CATALOG, index=False, encoding="utf-8-sig")

    excluded = merged[merged["cas_inputs"].eq("")].copy()
    excluded_cols = [
        "case_id", "category", "target_id", "target_scope_id",
        "candidate_cas", "candidate_components", "query_mode", "gold_label",
    ]
    for c in excluded_cols:
        if c not in excluded.columns:
            excluded[c] = ""
    excluded[excluded_cols].to_csv(EXCLUDED, index=False, encoding="utf-8-sig")

    usable = merged[merged["cas_inputs"].ne("")].copy()
    rows_input, rows_gold = [], []
    grouped = list(usable.groupby("cas_inputs", sort=True))
    for i, (signature, g) in enumerate(grouped, start=1):
        qid = f"RQ-{i:04d}"
        match_targets = sorted(set(g.loc[g["gold_label"].eq("MATCH"), "target_id"]))
        if len(match_targets) > 1:
            gold_target = "|".join(match_targets)
            gold_status = "MULTI_TARGET"
            derivation = "MULTIPLE_FROZEN_PAIRWISE_MATCH_TARGETS"
            review_required = "YES"
        elif len(match_targets) == 1:
            gold_target = match_targets[0]
            gold_status = "FOUND"
            derivation = "FROZEN_PAIRWISE_MATCH_TARGET"
            review_required = "NO_REUSED_VERIFIED_MATCH"
        else:
            gold_target = "NOT_FOUND"
            gold_status = "NOT_FOUND"
            derivation = "DERIVED_FROM_ABSENCE_OF_MATCH_IN_29_TARGET_CATALOG"
            review_required = "YES"

        rows_input.append({"query_id": qid, "cas_inputs": signature})
        source_categories = sorted(set(clean(v) for v in g["category"] if clean(v)))
        rows_gold.append({
            "query_id": qid,
            "gold_target_id": gold_target,
            "gold_status": gold_status,
            "source_categories": "|".join(source_categories),
            "source_case_ids": "|".join(sorted(set(clean(v) for v in g["case_id"]))),
            "query_mode": clean(g.iloc[0]["query_mode"]),
            "n_source_pairwise_rows": int(len(g)),
            "gold_derivation": derivation,
            "retrieval_scope_review_required": review_required,
        })

    qin = pd.DataFrame(rows_input)
    qgold = pd.DataFrame(rows_gold)

    forbidden = {
        "category", "target_id", "target_scope_id", "regulatory_scope_text",
        "candidate_name", "gold_label", "gold_target_id", "source_categories",
    }
    leakage = sorted(forbidden.intersection(qin.columns))
    if leakage:
        raise RuntimeError(f"Retrieval INPUT leakage: {leakage}")
    if qin["query_id"].duplicated().any() or qin["cas_inputs"].duplicated().any():
        raise RuntimeError("Retrieval INPUT query IDs/signatures must be unique")
    if set(qin["query_id"]) != set(qgold["query_id"]):
        raise RuntimeError("Retrieval INPUT/GOLD query IDs differ")

    qin.to_csv(QUERY_INPUT, index=False, encoding="utf-8-sig")
    qgold.to_csv(QUERY_GOLD, index=False, encoding="utf-8-sig")

    query_gold_counts = qgold["gold_status"].value_counts().to_dict()
    manifest = {
        "protocol": "VALIDATION_V2_DERIVED_CAS_ONLY_RETRIEVAL_V2",
        "source_validation_frozen_n": int(len(merged)),
        "regulatory_catalog_targets": int(len(catalog)),
        "retrieval_query_n": int(len(qin)),
        "query_gold_status_counts": {k: int(v) for k, v in query_gold_counts.items()},
        "retrieval_scope_review_required_n": int(
            qgold["retrieval_scope_review_required"].eq("YES").sum()
        ),
        "excluded_no_cas_pairwise_rows": int(len(excluded)),
        "excluded_no_cas_unique_targets": int(excluded["target_id"].nunique()) if len(excluded) else 0,
        "input_policy": (
            "Candidate-side input is CAS identifier(s) only. Single substances use one CAS. "
            "Named mixtures without a mixture CAS may use an all-CAS component set. No candidate "
            "names, row-specific target scopes, categories, GOLD labels, or regulatory IDs are "
            "present in retrieval_INPUT.csv."
        ),
        "catalog_policy": (
            "Catalog identity fields are derived only from frozen target/reference fields and the "
            "frozen regulatory-group reference. No mixture CAS is back-filled from a GOLD MATCH "
            "candidate. Named mixtures are represented by scope text and target component CAS set."
        ),
        "gold_review_policy": (
            "FOUND targets reuse frozen verified pairwise MATCH provenance. Derived NOT_FOUND and "
            "any MULTI_TARGET result require separate researcher review against the complete 29-target "
            "catalog before retrieval freeze."
        ),
        "external_data_policy": (
            "At runtime, PubChem may be queried only to obtain SMILES for CAS identifiers. PubChem "
            "titles, synonyms, names, classifications, and regulatory annotations must not be supplied "
            "to either evaluated system."
        ),
        "validation_freeze_verification": freeze_meta,
        "sha256": {
            "validation_master_INPUT.csv": sha256(MASTER_INPUT),
            "validation_master_GOLD.csv": sha256(MASTER_GOLD),
            "regulatory_catalog.csv": sha256(CATALOG),
            "retrieval_INPUT.csv": sha256(QUERY_INPUT),
            "retrieval_GOLD.csv": sha256(QUERY_GOLD),
            "retrieval_EXCLUDED_NO_CAS.csv": sha256(EXCLUDED),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    for p in [CATALOG, QUERY_INPUT, QUERY_GOLD, EXCLUDED, MANIFEST]:
        print(f"[OUT] {p}")


if __name__ == "__main__":
    main()
