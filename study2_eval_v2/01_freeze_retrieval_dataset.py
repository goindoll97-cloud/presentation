# -*- coding: utf-8 -*-
"""Audit and freeze the derived CAS-only retrieval dataset.

This freezes the retrieval transformation itself, not the later LLM/Hybrid
prompt or runner code. The original Validation V2 freeze remains untouched.

A documented CAS-normalization step is run after derivation and before audit.
This preserves the original frozen source while allowing a source CAS
transcription inconsistency to be corrected transparently in the derived
retrieval representation.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DATA = ROOT / "data"
BUILDER = ROOT / "00_build_retrieval_benchmark.py"
CORRECTOR = ROOT / "00a_apply_cas_corrections.py"
CATALOG = DATA / "regulatory_catalog.csv"
QINPUT = DATA / "retrieval_INPUT.csv"
QGOLD = DATA / "retrieval_GOLD.csv"
EXCLUDED = DATA / "retrieval_EXCLUDED_NO_CAS.csv"
CORRECTION_AUDIT = DATA / "retrieval_CAS_CORRECTIONS.csv"
MANIFEST = DATA / "retrieval_build_manifest.json"
VAL_FREEZE = REPO_ROOT / "validation_v2" / "VALIDATION_V2_FREEZE.json"
OUT = ROOT / "RETRIEVAL_DATASET_FREEZE.json"

CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_cas_checksum(cas: str) -> bool:
    value = re.sub(r"\s+", "", str(cas).strip())
    if not CAS_RE.fullmatch(value):
        return False
    body, check = value.rsplit("-", 1)
    digits = body.replace("-", "")
    checksum = sum((i + 1) * int(d) for i, d in enumerate(reversed(digits))) % 10
    return checksum == int(check)


def main() -> None:
    # Re-derive from the unchanged frozen Validation V2 source, then apply only
    # the explicitly documented retrieval-side CAS normalization.
    subprocess.run([sys.executable, str(BUILDER)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(CORRECTOR)], cwd=ROOT, check=True)

    for p in [
        CATALOG, QINPUT, QGOLD, EXCLUDED, CORRECTION_AUDIT,
        MANIFEST, VAL_FREEZE,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    qin = pd.read_csv(QINPUT, dtype=str).fillna("")
    qgold = pd.read_csv(QGOLD, dtype=str).fillna("")
    catalog = pd.read_csv(CATALOG, dtype=str).fillna("")
    excluded = pd.read_csv(EXCLUDED, dtype=str).fillna("")
    corrections = pd.read_csv(CORRECTION_AUDIT, dtype=str).fillna("")
    build = json.loads(MANIFEST.read_text(encoding="utf-8"))
    validation_freeze = json.loads(VAL_FREEZE.read_text(encoding="utf-8"))

    checks = []

    def add(name: str, ok: bool, details=""):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "details": str(details)})

    add("validation_v2_frozen_132", int(validation_freeze.get("n_total", 0)) == 132,
        validation_freeze.get("n_total"))
    add("validation_v2_audit_pass", validation_freeze.get("pre_freeze_audit_status") == "PASS",
        validation_freeze.get("pre_freeze_audit_status"))
    add("retrieval_input_columns_minimal", list(qin.columns) == ["query_id", "cas_inputs"],
        list(qin.columns))
    add("retrieval_input_query_unique", not qin["query_id"].duplicated().any(), len(qin))
    add("retrieval_input_signature_unique", not qin["cas_inputs"].duplicated().any(), len(qin))
    add("retrieval_input_gold_same_queries", set(qin["query_id"]) == set(qgold["query_id"]),
        f"input={len(qin)}, gold={len(qgold)}")
    add("catalog_target_unique", not catalog["target_id"].duplicated().any(), len(catalog))
    add("catalog_three_target_types",
        set(catalog["target_type"]) == {"PARENT_SALT", "CHEMICAL_GROUP", "MIXTURE"},
        sorted(set(catalog["target_type"])))
    add("catalog_29_targets", len(catalog) == 29, len(catalog))

    bad_cas = []
    multi = 0
    for r in qin.itertuples(index=False):
        parts = [x.strip() for x in str(r.cas_inputs).split("|") if x.strip()]
        if len(parts) > 1:
            multi += 1
        if not parts or any(not valid_cas_checksum(x) for x in parts):
            bad_cas.append({"query_id": r.query_id, "cas_inputs": r.cas_inputs})
    add("all_query_cas_checksum_valid", not bad_cas, bad_cas[:10])

    # The known source inconsistency must be explicitly documented rather than
    # silently repaired.  The original Validation V2 remains frozen unchanged.
    correction_ok = (
        len(corrections) > 0
        and (corrections["original_cas"] == "3084-48-0").any()
        and (corrections["corrected_cas"] == "3084-48-8").any()
    )
    add(
        "documented_source_cas_correction_applied",
        correction_ok,
        corrections[["location", "record_id", "original_cas", "corrected_cas"]].to_dict("records")[:10],
    )
    add(
        "corrected_trihexylphosphine_cas_valid",
        valid_cas_checksum("3084-48-8"),
        "3084-48-8",
    )

    allowed_gold = set(catalog["target_id"]) | {"NOT_FOUND"}
    invalid_targets = sorted(set(qgold["gold_target_id"]) - allowed_gold)
    add("gold_targets_exist_in_catalog_or_not_found", not invalid_targets, invalid_targets)

    multigold = qgold["gold_status"].eq("MULTI_TARGET")
    add("no_multitarget_gold", not multigold.any(),
        qgold.loc[multigold, ["query_id", "gold_target_id"]].to_dict("records"))

    leaked_tokens = []
    forbidden_names = {
        "category", "target", "scope", "candidate_name", "gold", "label",
        "regulatory", "reference_parent", "challenge", "difficulty",
    }
    for c in qin.columns:
        cl = c.lower()
        if any(t in cl for t in forbidden_names):
            leaked_tokens.append(c)
    add("no_target_or_label_columns_in_input", not leaked_tokens, leaked_tokens)

    add("retrieval_queries_nontrivial", len(qin) >= 100, len(qin))
    add("cas_set_queries_present_for_named_mixtures", multi > 0, f"multi-CAS queries={multi}")
    add("cas_unavailable_cases_documented", len(excluded) > 0, len(excluded))

    n_fail = sum(x["status"] == "FAIL" for x in checks)
    if n_fail:
        print(pd.DataFrame(checks).to_string(index=False))
        raise RuntimeError(f"Retrieval dataset audit failed: {n_fail} FAIL item(s)")

    freeze = {
        "protocol": "STUDY2_RETRIEVAL_DATASET_V1_1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_validation_protocol": validation_freeze.get("protocol"),
        "source_validation_frozen_at_utc": validation_freeze.get("frozen_at_utc"),
        "source_validation_n_pairwise": int(validation_freeze.get("n_total", 0)),
        "retrieval_query_n": int(len(qin)),
        "regulatory_catalog_target_n": int(len(catalog)),
        "query_mode_counts": {
            str(k): int(v)
            for k, v in qgold["query_mode"].value_counts().to_dict().items()
        },
        "gold_status_counts": {
            str(k): int(v)
            for k, v in qgold["gold_status"].value_counts().to_dict().items()
        },
        "catalog_type_counts": {
            str(k): int(v)
            for k, v in catalog["target_type"].value_counts().to_dict().items()
        },
        "excluded_no_cas_pairwise_rows": int(len(excluded)),
        "excluded_no_cas_unique_targets": int(
            build.get("excluded_no_cas_unique_targets", 0)
        ),
        "candidate_input_policy": (
            "retrieval_INPUT.csv contains only an opaque query_id and CAS identifier(s). "
            "No candidate names, target scope, category, or GOLD information is supplied."
        ),
        "mixture_input_policy": (
            "When a named mixture has no single mixture CAS but all components have CAS numbers, "
            "the query is the unordered component-CAS set. Mixtures with neither a mixture CAS "
            "nor an all-CAS component set are excluded from the primary CAS-only retrieval analysis."
        ),
        "external_data_policy": (
            "Both LLM-only and Hybrid may use the same PubChem-derived SMILES only. "
            "PubChem names, titles, synonyms, classifications, and regulatory annotations "
            "must not be exposed to the evaluated systems."
        ),
        "cas_normalization_policy": build.get("cas_correction_policy", ""),
        "cas_corrections": build.get("cas_corrections", []),
        "checks": checks,
        "sha256": {
            "00_build_retrieval_benchmark.py": sha256(BUILDER),
            "00a_apply_cas_corrections.py": sha256(CORRECTOR),
            "retrieval_build_manifest.json": sha256(MANIFEST),
            "regulatory_catalog.csv": sha256(CATALOG),
            "retrieval_INPUT.csv": sha256(QINPUT),
            "retrieval_GOLD.csv": sha256(QGOLD),
            "retrieval_EXCLUDED_NO_CAS.csv": sha256(EXCLUDED),
            "retrieval_CAS_CORRECTIONS.csv": sha256(CORRECTION_AUDIT),
            "../validation_v2/VALIDATION_V2_FREEZE.json": sha256(VAL_FREEZE),
        },
    }
    OUT.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")

    print(pd.DataFrame(checks).to_string(index=False))
    print(json.dumps(freeze, ensure_ascii=False, indent=2))
    print(f"[FROZEN] {OUT}")


if __name__ == "__main__":
    main()
