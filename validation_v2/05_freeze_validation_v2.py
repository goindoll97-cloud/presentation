# -*- coding: utf-8 -*-
"""Freeze Validation V2 after all three category datasets are curated.

By default this script refuses to freeze a paper-facing benchmark when the new
CHEMICAL_GROUP or MIXTURE subsets contain fewer than 20 approved cases. For
pipeline testing only, set VALIDATION_V2_ALLOW_SMALL_FREEZE=1.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd

from validation_common import DATA, ROOT, SEEDS, SOURCES, sha256

ALLOW_SMALL = os.getenv("VALIDATION_V2_ALLOW_SMALL_FREEZE", "0").strip().lower() in {"1", "true", "yes", "on"}
MIN_COUNTS = {"PARENT_SALT": 72, "CHEMICAL_GROUP": 20, "MIXTURE": 20}

FILES = [
    ROOT / "validation_common.py",
    ROOT / "01_build_parent_salt_validation.py",
    ROOT / "02_build_chemical_group_validation.py",
    ROOT / "03_build_mixture_validation.py",
    ROOT / "04_merge_validation_master.py",
    SEEDS / "chemical_group_candidates_seed.csv",
    SEEDS / "mixture_candidates_seed.csv",
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
    if too_small and not ALLOW_SMALL:
        raise RuntimeError(
            "Validation V2 is not large enough for final freeze. "
            f"Observed/minimum={too_small}. Expand curated seeds first. "
            "For development testing only, set VALIDATION_V2_ALLOW_SMALL_FREEZE=1."
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
        "design_notes": {
            "PARENT_SALT": "Controlled carry-over from V6; not an independent holdout.",
            "CHEMICAL_GROUP": "External-DB open-set candidates; official-list exact hits excluded from starter MATCH cases.",
            "MIXTURE": "Real named mixture records, including CAS-less mixture identities and composition hard negatives.",
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
