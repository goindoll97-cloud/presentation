# -*- coding: utf-8 -*-
"""Freeze Validation V2 after construction, source verification, audit, and sign-off."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd

from validation_common import DATA, ROOT, SEEDS, SOURCES, sha256

ALLOW_SMALL = os.getenv(
    "VALIDATION_V2_ALLOW_SMALL_FREEZE", "0"
).strip().lower() in {"1", "true", "yes", "on"}

MIN_COUNTS = {"PARENT_SALT": 72, "CHEMICAL_GROUP": 30, "MIXTURE": 30}
MIN_UNIQUE_MIXTURE_TARGETS = 15
EXPECTED_GROUP_RULES = {
    "MG_SALTS",
    "TBT_TRIALKYLTIN",
    "NONYLPHENOL_NPE",
    "LEAD_COMPOUNDS",
    "CRVI_COMPOUNDS",
}

AUDIT_JSON = ROOT / "VALIDATION_V2_AUDIT.json"
MANUAL_REVIEW = SOURCES / "validation_manual_source_review.csv"
SOURCE_EVIDENCE = SEEDS / "source_verification_evidence_v2.csv"

FILES = [
    ROOT / "validation_common.py",
    ROOT / "regulatory_group_reference.py",
    ROOT / "01_build_parent_salt_validation.py",
    ROOT / "02_build_chemical_group_validation.py",
    ROOT / "03_build_mixture_validation.py",
    ROOT / "04_merge_validation_master.py",
    ROOT / "04b_audit_validation_v2.py",
    ROOT / "04c_author_signoff.py",
    SEEDS / "chemical_group_candidates_seed.csv",
    SEEDS / "mixture_candidates_seed.csv",
    SEEDS / "mixture_candidates_seed_additional.csv",
    SOURCE_EVIDENCE,
    DATA / "validation_parent_salt_INPUT.csv",
    DATA / "validation_parent_salt_GOLD.csv",
    DATA / "validation_chemical_group_INPUT.csv",
    DATA / "validation_chemical_group_GOLD.csv",
    DATA / "validation_mixture_INPUT.csv",
    DATA / "validation_mixture_GOLD.csv",
    DATA / "validation_master_INPUT.csv",
    DATA / "validation_master_GOLD.csv",
    AUDIT_JSON,
    MANUAL_REVIEW,
]


def main() -> None:
    missing = [str(p) for p in FILES if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing files before freeze:\n" + "\n".join(missing)
        )

    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    if audit.get("overall_status") != "PASS":
        raise RuntimeError(
            "Validation V2 pre-freeze audit has not passed. Run "
            "04b_audit_validation_v2.py and resolve all FAIL items."
        )

    review = pd.read_csv(MANUAL_REVIEW, dtype=str).fillna("")
    if "source_verification_status" not in review.columns:
        raise RuntimeError("Source-verification status missing from review file.")
    if not review["source_verification_status"].eq("VERIFIED").all():
        raise RuntimeError("Source verification is incomplete.")
    if "author_signoff_status" not in review.columns:
        raise RuntimeError("Researcher sign-off status missing from review file.")
    signoff = review["author_signoff_status"].astype(str).str.upper()
    if not signoff.eq("APPROVED").all():
        raise RuntimeError(
            "Researcher sign-off is incomplete. Inspect the review table and "
            "use 04c_author_signoff.py --approve only after acceptance."
        )

    inp = pd.read_csv(DATA / "validation_master_INPUT.csv", dtype=str).fillna("")
    gold = pd.read_csv(DATA / "validation_master_GOLD.csv", dtype=str).fillna("")
    if set(inp["case_id"]) != set(gold["case_id"]):
        raise ValueError("Master INPUT/GOLD case IDs differ")
    if inp["case_id"].duplicated().any() or gold["case_id"].duplicated().any():
        raise ValueError("Duplicate case_id in master validation data")

    counts = gold["category"].value_counts().to_dict()
    too_small = {
        k: (int(counts.get(k, 0)), v)
        for k, v in MIN_COUNTS.items()
        if int(counts.get(k, 0)) < v
    }

    mix_inp = inp[inp["category"].eq("MIXTURE")].copy()
    grp_inp = inp[inp["category"].eq("CHEMICAL_GROUP")].copy()
    grp_gold = gold[gold["category"].eq("CHEMICAL_GROUP")].copy()

    unique_mix_targets = (
        int(mix_inp["target_scope_id"].nunique())
        if "target_scope_id" in mix_inp.columns else 0
    )
    group_rules = set(grp_inp["target_rule_id"]) if "target_rule_id" in grp_inp.columns else set()

    diversity_failures = {}
    if unique_mix_targets < MIN_UNIQUE_MIXTURE_TARGETS:
        diversity_failures["unique_mixture_targets"] = (
            unique_mix_targets, MIN_UNIQUE_MIXTURE_TARGETS
        )
    if group_rules != EXPECTED_GROUP_RULES:
        diversity_failures["chemical_group_rules"] = (
            sorted(group_rules), sorted(EXPECTED_GROUP_RULES)
        )

    if "target_rule_id" in grp_inp.columns:
        cg = grp_inp[["case_id", "target_rule_id"]].merge(
            grp_gold[["case_id", "gold_label"]], on="case_id", how="inner"
        )
        group_balance = {}
        balance_ok = True
        for rule_id in EXPECTED_GROUP_RULES:
            sub = cg[cg["target_rule_id"].eq(rule_id)]
            vc = sub["gold_label"].value_counts().to_dict()
            group_balance[rule_id] = {
                "n": int(len(sub)),
                "MATCH": int(vc.get("MATCH", 0)),
                "NO_MATCH": int(vc.get("NO_MATCH", 0)),
            }
            if len(sub) != 6 or vc.get("MATCH", 0) != 3 or vc.get("NO_MATCH", 0) != 3:
                balance_ok = False
        if not balance_ok:
            diversity_failures["chemical_group_balance"] = group_balance
    else:
        group_balance = {}

    if (too_small or diversity_failures) and not ALLOW_SMALL:
        raise RuntimeError(
            "Validation V2 is not ready for final paper-facing freeze. "
            f"row_count_failures={too_small}; "
            f"diversity_failures={diversity_failures}."
        )

    labels = gold["gold_label"].value_counts().to_dict()
    freeze = {
        "protocol": "VALIDATION_V2_THREE_CATEGORY",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "allow_small_freeze": ALLOW_SMALL,
        "pre_freeze_audit_status": audit.get("overall_status"),
        "source_verification_items": int(len(review)),
        "source_verification_verified": int(
            review["source_verification_status"].eq("VERIFIED").sum()
        ),
        "researcher_signoff_approved": int(signoff.eq("APPROVED").sum()),
        "n_total": int(len(gold)),
        "category_counts": {k: int(v) for k, v in counts.items()},
        "label_counts": {k: int(v) for k, v in labels.items()},
        "minimum_category_counts": MIN_COUNTS,
        "diversity": {
            "unique_mixture_targets": unique_mix_targets,
            "minimum_unique_mixture_targets": MIN_UNIQUE_MIXTURE_TARGETS,
            "chemical_group_rules": sorted(group_rules),
            "required_chemical_group_rules": sorted(EXPECTED_GROUP_RULES),
            "chemical_group_balance": group_balance,
        },
        "design_notes": {
            "PARENT_SALT": (
                "Controlled carry-over from V6; not an independent holdout."
            ),
            "CHEMICAL_GROUP": (
                "Balanced five-group challenge set: six cases per group, "
                "three MATCH and three NO_MATCH. MATCH includes both direct "
                "Appendix-3 enumeration and open-set generic-scope cases."
            ),
            "MIXTURE": (
                "Thirty classification cases based on fifteen unique real "
                "mixture identities; each target has one exact MATCH and one "
                "cross-mixture hard negative."
            ),
            "source_verification": (
                "AI-assisted source verification is explicitly recorded in "
                "source_verification_evidence_v2.csv and is not represented "
                "as human review."
            ),
            "researcher_signoff": (
                "A separate explicit researcher sign-off is required after "
                "inspection of the evidence table."
            ),
            "scope_semantics": (
                "CHEMICAL_GROUP MATCH denotes membership in the chemical/"
                "generic identity scope. It does not by itself assert that "
                "every use or concentration is legally restricted."
            ),
            "leakage_control": (
                "INPUT and GOLD files are physically separated before evaluation."
            ),
        },
        "sha256": {
            str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
            for p in FILES
        },
    }

    out = ROOT / "VALIDATION_V2_FREEZE.json"
    out.write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(freeze, ensure_ascii=False, indent=2))
    print(f"[FROZEN] {out}")


if __name__ == "__main__":
    main()
