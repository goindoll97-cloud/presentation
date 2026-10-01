# -*- coding: utf-8 -*-
"""Merge Parent/Salt, Chemical-group, and Mixture validation files.

INPUT and GOLD remain physically separate. The master INPUT intentionally does
not contain gold_label, gold_reason, curation_status, or source-audit columns.
"""
from __future__ import annotations

import pandas as pd

from validation_common import DATA, SOURCES, write_csv

CATEGORIES = [
    ("PARENT_SALT", "validation_parent_salt_INPUT.csv", "validation_parent_salt_GOLD.csv"),
    ("CHEMICAL_GROUP", "validation_chemical_group_INPUT.csv", "validation_chemical_group_GOLD.csv"),
    ("MIXTURE", "validation_mixture_INPUT.csv", "validation_mixture_GOLD.csv"),
]


def main() -> None:
    inputs, golds, summaries = [], [], []
    all_ids = []
    for category, ifile, gfile in CATEGORIES:
        ip, gp = DATA / ifile, DATA / gfile
        if not ip.exists() or not gp.exists():
            raise FileNotFoundError(f"Missing category files for {category}: {ip}, {gp}")
        a = pd.read_csv(ip, dtype=str).fillna("")
        b = pd.read_csv(gp, dtype=str).fillna("")
        if "case_id" not in a.columns or "case_id" not in b.columns:
            raise ValueError(f"Missing case_id in {category}")
        if set(a["case_id"]) != set(b["case_id"]):
            raise ValueError(f"INPUT/GOLD case mismatch in {category}")
        if not a["category"].eq(category).all() or not b["category"].eq(category).all():
            raise ValueError(f"Category value mismatch in {category}")
        inputs.append(a); golds.append(b); all_ids.extend(a["case_id"].tolist())
        counts = b["gold_label"].value_counts().to_dict() if "gold_label" in b.columns else {}
        summaries.append({
            "category": category,
            "n_cases": len(a),
            "n_match": int(counts.get("MATCH", 0)),
            "n_no_match": int(counts.get("NO_MATCH", 0)),
            "n_review": int(counts.get("REVIEW", 0)),
        })

    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate case_id across validation categories")

    master_input = pd.concat(inputs, ignore_index=True, sort=False).fillna("")
    master_gold = pd.concat(golds, ignore_index=True, sort=False).fillna("")

    forbidden = {"gold_label", "gold_reason", "curation_status", "official_list_hit"}
    leakage = forbidden.intersection(master_input.columns)
    if leakage:
        raise ValueError(f"Potential label/audit leakage in master INPUT: {sorted(leakage)}")

    write_csv(master_input, DATA / "validation_master_INPUT.csv")
    write_csv(master_gold, DATA / "validation_master_GOLD.csv")
    write_csv(pd.DataFrame(summaries), SOURCES / "validation_category_summary.csv")

    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"[OK] Master validation cases: {len(master_input)}")
    print(f"[OUT] {DATA / 'validation_master_INPUT.csv'}")
    print(f"[OUT] {DATA / 'validation_master_GOLD.csv'}")


if __name__ == "__main__":
    main()
