# -*- coding: utf-8 -*-
"""Freeze Validation V2 after all three category datasets are curated.

Paper-facing freeze requirements intentionally check both row counts and dataset
DIVERSITY. In particular, mixture cross-pairing must not inflate the apparent
sample size: at least 15 unique target mixture identities are required.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd

from validation_common import DATA, ROOT, SEEDS, sha256

ALLOW_SMALL = os.getenv("VALIDATION_V2_ALLOW_SMALL_FREEZE", "0").strip().lower() in {"1", "true", "yes", "on"}
MIN_COUNTS = {"PARENT_SALT": 72, "CHEMICAL_GROUP": 20, "MIXTURE": 20}
MIN_UNIQUE_MIXTURE_TARGETS = 15
MIN_CHEMICAL_GROUP_RULES = 3

FILES = [
    ROOT / "validation_common.py",
    ROOT / "regulatory_group_reference.py",
    ROOT / "01_build_parent_salt_validation.py",
    ROOT / "02_build_chemical_group_validation.py",
    ROOT / "03_build_mixture_validation.py",
    ROOT / "04_merge_validation_master.py",
    SEEDS / "chemical_group_candidates_seed.csv",
    SEEDS / "mixture_candidates_seed.csv",
    SEEDS / "mixture_candidates_seed_additional.csv",
    DATA / "validation_parent_salt_INPUT.csv",
    DATA / "validation_parent_salt_GOLD.csv",
    DATA / "validation_chemical_group_INPUT.csv",
    DATA / "validation_chemical_group_GOLD.csv",
    DATA / "validation_mixture_INPUT.csv",
    DATA / "validation_mixture_GOLD.csv",
    DATA / "validation_master_INPUT.csv",
    DATA / "validation_master_GOLD.csv",
]


def main() -> None:
    missing = [str(p) for p in FILES if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing files before freeze:\n" + "\n".join(missing))

    inp = pd.read_csv(DATA / "validation_master_INPUT.csv", dtype=str).fillna("")
    gold = pd.read_csv(DATA / "validation_master_GOLD.csv", dtype=str).fillna("")
    if set(inp["case_id"]) != set(gold["case_id"]):
        raise ValueError("Master INPUT/GOLD case IDs differ")
    if inp["case_id"].duplicated().any() or gold["case_id"].duplicated().any():
        raise ValueError("Duplicate case_id in master validation data")

    counts = gold["category"].value_counts().to_dict()
    too_small = {k: (int(counts.get(k, 0)), v) for k, v in MIN_COUNTS.items() if int(counts.get(k, 0)) < v}

    mix_inp = inp[inp["category"].eq("MIXTURE")].copy()
    grp_inp = inp[inp["category"].eq("CHEMICAL_GROUP")].copy()
    unique_mix_targets = int(mix_inp["target_scope_id"].nunique()) if "target_scope_id" in mix_inp.columns else 0
    unique_group_rules = int(grp_inp["target_rule_id"].nunique()) if "target_rule_id" in grp_inp.columns else 0

    diversity_failures = {}
    if unique_mix_targets < MIN_UNIQUE_MIXTURE_TARGETS:
        diversity_failures["unique_mixture_targets"] = (unique_mix_targets, MIN_UNIQUE_MIXTURE_TARGETS)
    if unique_group_rules < MIN_CHEMICAL_GROUP_RULES:
        diversity_failures["chemical_group_rules"] = (unique_group_rules, MIN_CHEMICAL_GROUP_RULES)

    if (too_small or diversity_failures) and not ALLOW_SMALL:
        raise RuntimeError(
            "Validation V2 is not ready for final paper-facing freeze. "
            f"row_count_failures={too_small}; diversity_failures={diversity_failures}. "
            "Expand curated seeds first. For development testing only, set "
            "VALIDATION_V2_ALLOW_SMALL_FREEZE=1."
        )

    labels = gold["gold_label"].value_counts().to_dict()
    freeze = {
        "protocol": "VALIDATION_V2_THREE_CATEGORY",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "allow_small_freeze": ALLOW_SMALL,
        "n_total": int(len(gold)),
        "category_counts": {k: int(v) for k, v in counts.items()},
        "label_counts": {k: int(v) for k, v in labels.items()},
        "minimum_category_counts": MIN_COUNTS,
        "diversity": {
            "unique_mixture_targets": unique_mix_targets,
            "minimum_unique_mixture_targets": MIN_UNIQUE_MIXTURE_TARGETS,
            "chemical_group_rules": unique_group_rules,
            "minimum_chemical_group_rules": MIN_CHEMICAL_GROUP_RULES,
        },
        "design_notes": {
            "PARENT_SALT": "Controlled carry-over from V6; not an independent holdout.",
            "CHEMICAL_GROUP": "External-DB open-set candidates; exact target-rule CAS hits excluded from MATCH seed cases.",
            "MIXTURE": "Real named chemical-mixture records with exact-composition matches and cross-mixture hard negatives.",
            "mixture_effective_n": "Report unique mixture identities separately from pairwise case rows.",
            "leakage_control": "INPUT and GOLD files are physically separated before evaluation.",
        },
        "sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in FILES},
    }
    out = ROOT / "VALIDATION_V2_FREEZE.json"
    out.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(freeze, ensure_ascii=False, indent=2))
    print(f"[FROZEN] {out}")


if __name__ == "__main__":
    main()
