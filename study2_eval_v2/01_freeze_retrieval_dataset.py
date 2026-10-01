# -*- coding: utf-8 -*-
"""Audit and freeze the derived CAS-only retrieval dataset.

The original Validation V2 freeze remains untouched. This step rebuilds the
retrieval derivative, applies independently verified official mixture-CAS
references, applies documented CAS corrections, refreshes the human review
table, and refuses to freeze until every retrieval-scope review-required GOLD
row has explicit researcher approval.
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
MIXTURE_REF_APPLIER = ROOT / "00c_apply_mixture_regulatory_reference.py"
MIXTURE_REFERENCE = ROOT / "reference" / "mixture_regulatory_reference.csv"
MIXTURE_REFERENCE_AUDIT = DATA / "mixture_regulatory_reference_applied.csv"
CORRECTOR = ROOT / "00a_apply_cas_corrections.py"
GOLD_OVERRIDER = ROOT / "00b_apply_gold_overrides.py"
REVIEW_PREP = ROOT / "01a_prepare_retrieval_gold_review.py"
REVIEW_SIGNOFF = ROOT / "01b_retrieval_gold_signoff.py"
CATALOG = DATA / "regulatory_catalog.csv"
QINPUT = DATA / "retrieval_INPUT.csv"
QGOLD = DATA / "retrieval_GOLD.csv"
EXCLUDED = DATA / "retrieval_EXCLUDED_NO_CAS.csv"
CORRECTIONS = DATA / "retrieval_CAS_CORRECTIONS.csv"
MANIFEST = DATA / "retrieval_build_manifest.json"
REVIEW = ROOT / "review" / "retrieval_gold_manual_review.csv"
GOLD_OVERRIDES = ROOT / "review" / "retrieval_gold_overrides.csv"
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
    for script in [
        BUILDER, MIXTURE_REF_APPLIER, CORRECTOR,
        GOLD_OVERRIDER, REVIEW_PREP, REVIEW_SIGNOFF,
    ]:
        if not script.exists():
            raise FileNotFoundError(script)
    if not MIXTURE_REFERENCE.exists():
        raise FileNotFoundError(MIXTURE_REFERENCE)

    subprocess.run([sys.executable, str(BUILDER)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(MIXTURE_REF_APPLIER)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(CORRECTOR)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(GOLD_OVERRIDER)], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(REVIEW_PREP)], cwd=ROOT, check=True)

    for p in [
        CATALOG, QINPUT, QGOLD, EXCLUDED, CORRECTIONS, MANIFEST, REVIEW,
        GOLD_OVERRIDES, MIXTURE_REFERENCE, MIXTURE_REFERENCE_AUDIT, VAL_FREEZE,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    qin = pd.read_csv(QINPUT, dtype=str).fillna("")
    qgold = pd.read_csv(QGOLD, dtype=str).fillna("")
    catalog = pd.read_csv(CATALOG, dtype=str).fillna("")
    excluded = pd.read_csv(EXCLUDED, dtype=str).fillna("")
    review = pd.read_csv(REVIEW, dtype=str).fillna("")
    mix_ref = pd.read_csv(MIXTURE_REFERENCE, dtype=str).fillna("")
    mix_audit = pd.read_csv(MIXTURE_REFERENCE_AUDIT, dtype=str).fillna("")
    build = json.loads(MANIFEST.read_text(encoding="utf-8"))
    validation_freeze = json.loads(VAL_FREEZE.read_text(encoding="utf-8"))

    checks = []

    def add(name: str, ok: bool, details=""):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "details": str(details)})

    add("validation_v2_frozen_132", int(validation_freeze.get("n_total", 0)) == 132,
        validation_freeze.get("n_total"))
    add("validation_v2_audit_pass", validation_freeze.get("pre_freeze_audit_status") == "PASS",
        validation_freeze.get("pre_freeze_audit_status"))
    add("retrieval_input_columns_minimal", list(qin.columns) == ["query_id", "cas_inputs"], list(qin.columns))
    add("retrieval_input_query_unique", not qin["query_id"].duplicated().any(), len(qin))
    add("retrieval_input_signature_unique", not qin["cas_inputs"].duplicated().any(), len(qin))
    add("retrieval_input_gold_same_queries", set(qin["query_id"]) == set(qgold["query_id"]),
        f"input={len(qin)}, gold={len(qgold)}")
    add("catalog_target_unique", not catalog["target_id"].duplicated().any(), len(catalog))
    add("catalog_three_target_types",
        set(catalog["target_type"]) == {"PARENT_SALT", "CHEMICAL_GROUP", "MIXTURE"},
        sorted(set(catalog["target_type"])))
    add("catalog_29_targets", len(catalog) == 29, len(catalog))

    # Mixture CAS values may enter the catalog only through the independent
    # official regulatory reference file; never by back-filling a GOLD MATCH row.
    ref_pairs = {
        (str(r.target_id).strip(), str(r.official_mixture_cas).strip())
        for r in mix_ref.itertuples(index=False)
        if str(r.target_id).strip() and str(r.official_mixture_cas).strip()
    }
    catalog_pairs = {
        (str(r.target_id).strip(), str(getattr(r, "official_mixture_cas", "")).strip())
        for r in catalog.itertuples(index=False)
        if str(getattr(r, "official_mixture_cas", "")).strip()
    }
    add(
        "mixture_official_cas_from_independent_reference",
        bool(ref_pairs) and catalog_pairs == ref_pairs,
        f"reference={sorted(ref_pairs)}, catalog={sorted(catalog_pairs)}",
    )
    basis_ok = True
    if catalog_pairs:
        for tid, _cas in catalog_pairs:
            row = catalog[catalog["target_id"].eq(tid)]
            if len(row) != 1 or row.iloc[0].get("official_mixture_cas_basis", "") != "INDEPENDENT_OFFICIAL_REGULATORY_REFERENCE":
                basis_ok = False
                break
    add(
        "mixture_official_cas_basis_independent",
        basis_ok and len(mix_audit) == len(ref_pairs),
        mix_audit[["target_id", "official_mixture_cas", "source_name"]].to_dict("records") if len(mix_audit) else [],
    )

    bad_cas, multi = [], 0
    for r in qin.itertuples(index=False):
        parts = [x.strip() for x in str(r.cas_inputs).split("|") if x.strip()]
        if len(parts) > 1:
            multi += 1
        if not parts or any(not valid_cas_checksum(x) for x in parts):
            bad_cas.append({"query_id": r.query_id, "cas_inputs": r.cas_inputs})
    add("all_query_cas_checksum_valid", not bad_cas, bad_cas[:10])

    allowed_gold = set(catalog["target_id"]) | {"NOT_FOUND"}
    invalid_targets = sorted(set(qgold["gold_target_id"]) - allowed_gold)
    add("gold_targets_exist_in_catalog_or_not_found", not invalid_targets, invalid_targets)
    multigold = qgold["gold_status"].eq("MULTI_TARGET")
    add("no_multitarget_gold", not multigold.any(),
        qgold.loc[multigold, ["query_id", "gold_target_id"]].to_dict("records"))

    leaked_tokens = []
    for c in qin.columns:
        cl = c.lower()
        if any(t in cl for t in {"category", "target", "scope", "candidate_name", "gold", "label", "regulatory", "challenge"}):
            leaked_tokens.append(c)
    add("no_target_or_label_columns_in_input", not leaked_tokens, leaked_tokens)
    add("retrieval_queries_nontrivial", len(qin) >= 100, len(qin))
    add("cas_set_queries_present_for_named_mixtures", multi > 0, f"multi-CAS queries={multi}")
    add("cas_unavailable_cases_documented", len(excluded) > 0, len(excluded))

    required = review[review["review_required"].astype(str).str.upper().eq("YES")].copy()
    approved = required["researcher_status"].astype(str).str.upper().eq("APPROVED")
    rejected = required["researcher_status"].astype(str).str.upper().eq("REJECTED")
    pending = required["researcher_status"].astype(str).str.upper().eq("PENDING")
    expected_required = set(qgold.loc[
        qgold["retrieval_scope_review_required"].astype(str).str.upper().eq("YES"), "query_id"
    ])
    add("retrieval_gold_review_query_coverage",
        set(required["query_id"]) == expected_required,
        f"required_review={len(required)}, expected={len(expected_required)}")
    add("retrieval_gold_researcher_signoff_complete",
        bool(len(required) > 0 and approved.all() and not rejected.any() and not pending.any()),
        f"approved={int(approved.sum())}, pending={int(pending.sum())}, rejected={int(rejected.sum())}")

    conflict = required.apply(
        lambda r: bool(r["catalog_exact_hit_precheck"])
        and r["derived_gold_target_id"] not in r["catalog_exact_hit_precheck"].split("|"),
        axis=1,
    ) if len(required) else pd.Series(dtype=bool)
    unresolved = required[conflict & ~(
        required["researcher_status"].str.upper().eq("APPROVED")
        & required["researcher_note"].str.startswith("CONFLICT_RESOLVED:")
    )] if len(required) else required
    add("exact_hit_conflicts_resolved", unresolved.empty,
        unresolved[["query_id", "cas_inputs", "derived_gold_target_id",
                    "catalog_exact_hit_precheck"]].to_dict("records")[:10])

    ps = catalog[catalog["target_type"].eq("PARENT_SALT")]
    add("parent_salt_isomer_scope_explicit",
        bool(len(ps) and ps["isomer_scope"].astype(str).str.len().gt(0).all()),
        sorted(set(ps["isomer_scope"])))
    add("catalog_identity_basis_present",
        catalog["catalog_identity_basis"].astype(str).str.len().gt(0).all(),
        sorted(set(catalog["catalog_identity_basis"])))

    n_fail = sum(x["status"] == "FAIL" for x in checks)
    print(pd.DataFrame(checks).to_string(index=False))
    if n_fail:
        print("\n[BLOCKED] Retrieval dataset is not frozen.")
        print(f"Review file: {REVIEW}")
        print("After manual inspection, run: python 01b_retrieval_gold_signoff.py --approve")
        raise RuntimeError(f"Retrieval dataset audit failed: {n_fail} FAIL item(s)")

    freeze = {
        "protocol": "STUDY2_RETRIEVAL_DATASET_V2_1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_validation_protocol": validation_freeze.get("protocol"),
        "source_validation_frozen_at_utc": validation_freeze.get("frozen_at_utc"),
        "source_validation_n_pairwise": int(validation_freeze.get("n_total", 0)),
        "retrieval_query_n": int(len(qin)),
        "regulatory_catalog_target_n": int(len(catalog)),
        "query_mode_counts": {str(k): int(v) for k, v in qgold["query_mode"].value_counts().to_dict().items()},
        "gold_status_counts": {str(k): int(v) for k, v in qgold["gold_status"].value_counts().to_dict().items()},
        "catalog_type_counts": {str(k): int(v) for k, v in catalog["target_type"].value_counts().to_dict().items()},
        "retrieval_scope_review_required": int(len(required)),
        "retrieval_scope_review_approved": int(approved.sum()),
        "exact_hit_conflicts_resolved_by_note": int(conflict.sum()) if len(required) else 0,
        "gold_overrides": build.get("gold_overrides", []),
        "excluded_no_cas_pairwise_rows": int(len(excluded)),
        "excluded_no_cas_unique_targets": int(build.get("excluded_no_cas_unique_targets", 0)),
        "candidate_input_policy": (
            "retrieval_INPUT.csv contains only an opaque query_id and CAS identifier(s). No candidate "
            "names, row-specific target scope, category, or GOLD information is supplied."
        ),
        "catalog_policy": (
            "Catalog target identity is built independently of retrieval GOLD. Named-mixture component "
            "CAS sets come from frozen target scope fields; an official mixture CAS is included only "
            "when independently verified in reference/mixture_regulatory_reference.csv from an official "
            "regulatory source. No mixture CAS is inferred from a GOLD MATCH candidate."
        ),
        "gold_review_policy": (
            "Every derived NOT_FOUND or MULTI_TARGET retrieval claim requires explicit researcher "
            "review against all frozen catalog targets before freeze."
        ),
        "external_data_policy": (
            "Both LLM-only and Hybrid may use the same PubChem-derived SMILES only. PubChem names, "
            "titles, synonyms, classifications, and regulatory annotations are not model inputs."
        ),
        "checks": checks,
        "sha256": {
            "00_build_retrieval_benchmark.py": sha256(BUILDER),
            "00c_apply_mixture_regulatory_reference.py": sha256(MIXTURE_REF_APPLIER),
            "reference/mixture_regulatory_reference.csv": sha256(MIXTURE_REFERENCE),
            "data/mixture_regulatory_reference_applied.csv": sha256(MIXTURE_REFERENCE_AUDIT),
            "00a_apply_cas_corrections.py": sha256(CORRECTOR),
            "00b_apply_gold_overrides.py": sha256(GOLD_OVERRIDER),
            "review/retrieval_gold_overrides.csv": sha256(GOLD_OVERRIDES),
            "01a_prepare_retrieval_gold_review.py": sha256(REVIEW_PREP),
            "01b_retrieval_gold_signoff.py": sha256(REVIEW_SIGNOFF),
            "retrieval_build_manifest.json": sha256(MANIFEST),
            "regulatory_catalog.csv": sha256(CATALOG),
            "retrieval_INPUT.csv": sha256(QINPUT),
            "retrieval_GOLD.csv": sha256(QGOLD),
            "retrieval_EXCLUDED_NO_CAS.csv": sha256(EXCLUDED),
            "retrieval_CAS_CORRECTIONS.csv": sha256(CORRECTIONS),
            "review/retrieval_gold_manual_review.csv": sha256(REVIEW),
            "../validation_v2/VALIDATION_V2_FREEZE.json": sha256(VAL_FREEZE),
        },
    }
    OUT.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(freeze, ensure_ascii=False, indent=2))
    print(f"[FROZEN] {OUT}")


if __name__ == "__main__":
    main()
