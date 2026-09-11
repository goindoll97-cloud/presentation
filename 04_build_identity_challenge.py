# -*- coding: utf-8 -*-
"""Study 2 / Step 04: build a source-supported chemical-identity challenge.

This study is deliberately separate from Study 1 (regulatory-scope interpretation).
It asks a different question:

    After an extended regulatory scope such as "A and its salts" has already
    been identified, can a model/system decide whether an inventory CAS belongs
    to that scope?

The operational reference is source-supported chemical identity evidence from
broad_salt_validation_cases.csv.  It is NOT expert legal gold and it is not a
final compliance determination.

Outputs
-------
intermediate/04_identity_challenge_benchmark.csv
intermediate/04_identity_challenge_rules.csv
intermediate/04_identity_challenge_qc.csv
intermediate/04_identity_challenge_config.json
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INTERMEDIATE = ROOT / "intermediate"
INTERMEDIATE.mkdir(parents=True, exist_ok=True)
ENGINE_DATA = ROOT / "engine" / "data"

RULE_ENV = "STUDY2_BROAD_SALT_RULES_FILE"
VALID_ENV = "STUDY2_BROAD_SALT_VALIDATION_FILE"
SCOPE_ENV = "STUDY2_SCOPE_BENCHMARK_FILE"


def clean(x) -> str:
    if x is None:
        return ""
    try:
        if bool(pd.isna(x)):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    return "" if s.lower() in {"nan", "none", "null", "<na>"} else s


def as_bool(x) -> bool:
    return clean(x).lower() in {"1", "true", "yes", "y", "verified", "frozen", "auto", "automatic"}


def parse_membership(x) -> Optional[bool]:
    s = clean(x).upper()
    if s in {"MATCH", "Y", "YES", "TRUE", "1", "IN", "MEMBER", "POSITIVE"}:
        return True
    if s in {"NO_MATCH", "N", "NO", "FALSE", "0", "OUT", "NON_MEMBER", "NEGATIVE"}:
        return False
    return None


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, dtype=str).fillna("")


def locate(explicit_env: str, candidates: list[Path]) -> Optional[Path]:
    explicit = clean(os.getenv(explicit_env, ""))
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = ROOT / p
        return p if p.exists() else None
    for p in candidates:
        if p.exists():
            return p
    return None


def fallback_from_paper_xlsx(sheet_name: str) -> pd.DataFrame:
    files = sorted(ROOT.glob("PAPER_LLM_RULE_CAS_GAP_RESULTS*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files:
        try:
            xls = pd.ExcelFile(p)
            if sheet_name in xls.sheet_names:
                return pd.read_excel(p, sheet_name=sheet_name)
        except Exception:
            continue
    return pd.DataFrame()


def extract_scope_from_note(note: str, parent_name: str, parent_cas: str) -> str:
    txt = clean(note)
    m = re.search(r"Official scope:\s*(.+?)(?:\.\s*Parent identity|$)", txt, flags=re.I)
    if m:
        return m.group(1).strip()
    return f"{parent_name}({parent_cas}) and its salts"


def main() -> None:
    print("=" * 80)
    print("04 / STUDY 2 - BUILD CHEMICAL-IDENTITY CHALLENGE")
    print("=" * 80)

    rule_path = locate(RULE_ENV, [ENGINE_DATA / "broad_salt_rules.csv", ENGINE_DATA / "broad_salt_rules.xlsx"])
    valid_path = locate(VALID_ENV, [ENGINE_DATA / "broad_salt_validation_cases.csv", ENGINE_DATA / "broad_salt_validation_cases.xlsx"])
    scope_path = locate(SCOPE_ENV, [INTERMEDIATE / "01_cas_gap_benchmark_full.csv", INTERMEDIATE / "01_cas_gap_benchmark_primary.csv"])

    rules = read_table(rule_path) if rule_path else fallback_from_paper_xlsx("Supp_broad_salt_rules")
    cases = read_table(valid_path) if valid_path else fallback_from_paper_xlsx("Supp_broad_salt_validation")
    scope = read_table(scope_path) if scope_path else fallback_from_paper_xlsx("Supp_primary_benchmark")

    if rules.empty:
        raise FileNotFoundError("broad_salt_rules.csv/xlsx not found and no supplementary fallback was available")
    if cases.empty:
        raise FileNotFoundError("broad_salt_validation_cases.csv/xlsx not found and no supplementary fallback was available")

    for col in [
        "rule_id", "designation_id", "reference_parent_name", "reference_parent_cas",
        "reference_smiles", "salt_scope", "automation_status", "source_verified",
        "reference_rule_frozen", "regulatory_source", "chemical_identity_source",
        "rule_version", "frozen_date", "notes",
    ]:
        if col not in rules.columns:
            rules[col] = ""

    rules = rules.copy()
    rules["source_verified_bool"] = rules["source_verified"].map(as_bool)
    rules["reference_rule_frozen_bool"] = rules["reference_rule_frozen"].map(as_bool)
    rules["automation_allowed_bool"] = rules["automation_status"].map(as_bool)
    rules["supported_broad_salt_scope"] = rules["salt_scope"].astype(str).str.lower().isin(
        ["all_counterion_salts", "all_salts", "broad_salt"]
    )
    rules["machine_executable_reference"] = rules["reference_smiles"].map(clean).ne("")
    rules["study2_eligible"] = (
        rules["source_verified_bool"]
        & rules["reference_rule_frozen_bool"]
        & rules["automation_allowed_bool"]
        & rules["supported_broad_salt_scope"]
        & rules["machine_executable_reference"]
    )
    eligible = rules[rules["study2_eligible"]].copy()
    if eligible.empty:
        raise RuntimeError("No source-verified/frozen/machine-executable broad-salt rules are eligible")

    # Exact current regulatory source text from Study 1 when available.
    source_map: dict[str, str] = {}
    if not scope.empty and "designation_id" in scope.columns and "source_text" in scope.columns:
        s = scope.copy()
        if "reference_scope_type" in s.columns:
            broad = s[s["reference_scope_type"].astype(str).eq("BROAD_SALT")]
            if not broad.empty:
                s = broad
        for did, g in s.groupby(s["designation_id"].map(clean)):
            vals = [clean(v) for v in g["source_text"] if clean(v)]
            if vals:
                source_map[did] = vals[0]

    rule_lookup = eligible.set_index(eligible["rule_id"].map(clean), drop=False)
    did_lookup = {clean(r["designation_id"]): r for _, r in eligible.iterrows()}

    required_case_cols = [
        "case_id", "rule_id", "designation_id", "chemical_name", "cas", "smiles",
        "reference_membership", "reference_source", "notes",
    ]
    for col in required_case_cols:
        if col not in cases.columns:
            cases[col] = ""

    rows = []
    excluded = []
    for _, c in cases.iterrows():
        rid = clean(c.get("rule_id"))
        did = clean(c.get("designation_id"))
        rr = None
        if rid and rid in rule_lookup.index:
            rr = rule_lookup.loc[rid]
            if isinstance(rr, pd.DataFrame):
                rr = None
        if rr is None and did in did_lookup:
            rr = did_lookup[did]
        truth = parse_membership(c.get("reference_membership"))
        ref_source = clean(c.get("reference_source"))
        if rr is None or truth is None or not ref_source:
            excluded.append({
                "case_id": clean(c.get("case_id")),
                "reason": "RULE_NOT_ELIGIBLE" if rr is None else ("REFERENCE_LABEL_INVALID" if truth is None else "REFERENCE_SOURCE_MISSING"),
            })
            continue

        parent_name = clean(rr.get("reference_parent_name"))
        parent_cas = clean(rr.get("reference_parent_cas"))
        regulatory_text = source_map.get(clean(rr.get("designation_id")), "")
        source_text_origin = "STUDY1_CURRENT_OFFICIAL_SOURCE_TEXT" if regulatory_text else "FROZEN_RULE_NOTE_SCOPE_FALLBACK"
        if not regulatory_text:
            regulatory_text = extract_scope_from_note(rr.get("notes"), parent_name, parent_cas)

        case_id = clean(c.get("case_id"))
        rows.append({
            "identity_benchmark_id": f"ID2_{case_id}",
            "case_id": case_id,
            "rule_id": clean(rr.get("rule_id")),
            "designation_id": clean(rr.get("designation_id")),
            "regulatory_scope_text": regulatory_text,
            "scope_text_origin": source_text_origin,
            "reference_parent_name": parent_name,
            "reference_parent_cas": parent_cas,
            "reference_parent_smiles": clean(rr.get("reference_smiles")),
            "salt_scope": clean(rr.get("salt_scope")),
            "candidate_name": clean(c.get("chemical_name")),
            "candidate_cas": clean(c.get("cas")),
            "candidate_source_smiles": clean(c.get("smiles")),
            "reference_membership": "MATCH" if truth else "NO_MATCH",
            "reference_membership_bool": bool(truth),
            "reference_source": ref_source,
            "reference_notes": clean(c.get("notes")),
            "reference_type": "SOURCE_SUPPORTED_CHEMICAL_IDENTITY_OPERATIONAL_REFERENCE_NOT_EXPERT_LEGAL_GOLD",
            "case_class": "POSITIVE_MEMBER" if truth else "HARD_NEGATIVE_PAIR",
            "operational_cas_eligible": bool(clean(c.get("cas"))),
            "controlled_structure_eligible": bool(clean(c.get("smiles")) and clean(rr.get("reference_smiles"))),
            "rule_source_verified": bool(rr.get("source_verified_bool")),
            "rule_frozen_before_validation": bool(rr.get("reference_rule_frozen_bool")),
            "regulatory_source": clean(rr.get("regulatory_source")),
            "chemical_identity_source": clean(rr.get("chemical_identity_source")),
            "rule_version": clean(rr.get("rule_version")),
            "rule_frozen_date": clean(rr.get("frozen_date")),
        })

    bench = pd.DataFrame(rows)
    if bench.empty:
        raise RuntimeError("No eligible Study 2 benchmark cases were constructed")
    bench = bench.sort_values(["designation_id", "case_id"], kind="stable").reset_index(drop=True)

    # QC: one frozen rule per benchmark rule_id and no label leakage into model-input columns.
    duplicate_ids = int(bench["identity_benchmark_id"].duplicated().sum())
    label_counts = bench["reference_membership"].value_counts().to_dict()
    by_rule = bench.groupby("designation_id")["reference_membership"].value_counts().unstack(fill_value=0)
    n_missing_struct = int(bench["candidate_source_smiles"].map(clean).eq("").sum())
    n_missing_cas = int(bench["candidate_cas"].map(clean).eq("").sum())

    qc_rows = [
        {"metric": "n_rules_eligible", "value": int(bench["rule_id"].nunique())},
        {"metric": "n_cases", "value": int(len(bench))},
        {"metric": "n_positive", "value": int(label_counts.get("MATCH", 0))},
        {"metric": "n_negative", "value": int(label_counts.get("NO_MATCH", 0))},
        {"metric": "n_missing_candidate_cas", "value": n_missing_cas},
        {"metric": "n_operational_cas_eligible", "value": int(bench["operational_cas_eligible"].astype(bool).sum())},
        {"metric": "n_controlled_structure_eligible", "value": int(bench["controlled_structure_eligible"].astype(bool).sum())},
        {"metric": "n_missing_source_smiles", "value": n_missing_struct},
        {"metric": "n_duplicate_benchmark_ids", "value": duplicate_ids},
        {"metric": "reference_is_expert_legal_gold", "value": False},
        {"metric": "primary_operational_input", "value": "regulatory_scope_text + candidate_CAS_only"},
        {"metric": "controlled_input", "value": "regulatory_scope_text + candidate/reference_SMILES"},
    ]
    qc = pd.DataFrame(qc_rows)

    eligible.to_csv(INTERMEDIATE / "04_identity_challenge_rules.csv", index=False, encoding="utf-8-sig")
    bench.to_csv(INTERMEDIATE / "04_identity_challenge_benchmark.csv", index=False, encoding="utf-8-sig")
    qc.to_csv(INTERMEDIATE / "04_identity_challenge_qc.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(excluded).to_csv(INTERMEDIATE / "04_identity_challenge_excluded.csv", index=False, encoding="utf-8-sig")
    by_rule.reset_index().to_csv(INTERMEDIATE / "04_identity_challenge_by_rule_balance.csv", index=False, encoding="utf-8-sig")

    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rules_file": str(rule_path) if rule_path else "PAPER_XLSX_FALLBACK",
        "validation_file": str(valid_path) if valid_path else "PAPER_XLSX_FALLBACK",
        "scope_file": str(scope_path) if scope_path else "PAPER_XLSX_FALLBACK",
        "study_question": "Can high-capability LLMs resolve CAS-to-broad-salt regulatory identity as reliably as deterministic cheminformatics?",
        "primary_setting": "CAS-only operational",
        "controlled_setting": "same structure information supplied to both LLM and RDKit",
        "reference_definition": "source-supported chemical identity reference; not expert legal gold",
    }
    (INTERMEDIATE / "04_identity_challenge_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Eligible frozen broad-salt rules: {bench['rule_id'].nunique()}")
    print(f"Benchmark cases: {len(bench)} | MATCH={label_counts.get('MATCH',0)} | NO_MATCH={label_counts.get('NO_MATCH',0)}")
    print(f"Operational CAS-eligible: {int(bench['operational_cas_eligible'].astype(bool).sum())} | Controlled structure-eligible: {int(bench['controlled_structure_eligible'].astype(bool).sum())}")
    print(f"Missing CAS: {n_missing_cas} | Missing source SMILES: {n_missing_struct}")
    print("\n[Cases by regulatory scope]")
    print(by_rule.to_string())
    print("\nSaved to: intermediate/")
    print("Next: python 05_compare_claude_rdkit_identity.py")


if __name__ == "__main__":
    main()
