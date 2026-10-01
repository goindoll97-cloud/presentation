# -*- coding: utf-8 -*-
"""Build the Parent/Salt validation subset from the frozen V6 72-case benchmark.

Important
---------
These 72 cases are reused from Study 2 V6 and are therefore a controlled
carry-over subset, not a newly independent holdout. The provenance is written
explicitly into GOLD and manifest files.
"""
from __future__ import annotations

import pandas as pd

from validation_common import REPO_ROOT, SOURCES, split_input_gold, write_csv

SRC = REPO_ROOT / "data" / "identity_e2e_v6_72.csv"


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"Missing source benchmark: {SRC}")
    df = pd.read_csv(SRC, dtype=str).fillna("")
    required = [
        "case_id", "regulatory_scope_text", "reference_parent_name", "reference_parent_cas",
        "candidate_name", "candidate_cas", "reference_membership", "challenge_class",
        "difficulty", "isomer_scope",
    ]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing V6 columns: {miss}")
    if len(df) != 72:
        raise ValueError(f"Expected 72 V6 cases, found {len(df)}")

    out = pd.DataFrame({
        "case_id": [f"PS-{i:03d}" for i in range(1, len(df) + 1)],
        "source_case_id": df["case_id"],
        "category": "PARENT_SALT",
        "regulatory_scope_text": df["regulatory_scope_text"],
        "reference_parent_name": df["reference_parent_name"],
        "reference_parent_cas": df["reference_parent_cas"],
        "candidate_name": df["candidate_name"],
        "candidate_cas": df["candidate_cas"],
        "challenge_class": df["challenge_class"],
        "difficulty": df["difficulty"],
        "isomer_scope": df["isomer_scope"],
        "gold_label": df["reference_membership"].astype(str).str.upper(),
        "gold_basis": "REUSED_STUDY2_V6_CONTROLLED_BENCHMARK",
        "curation_status": "APPROVED",
        "independent_holdout": "NO",
        "provenance_note": "Same 72 chemical cases as Study 2 V6; retained as controlled parent/salt subset.",
    })

    input_cols = [
        "case_id", "category", "regulatory_scope_text", "reference_parent_name",
        "reference_parent_cas", "candidate_name", "candidate_cas", "challenge_class",
        "difficulty", "isomer_scope",
    ]
    gold_cols = [
        "case_id", "category", "gold_label", "gold_basis", "curation_status",
        "independent_holdout", "source_case_id", "provenance_note",
    ]
    ip, gp = split_input_gold(out, input_cols, gold_cols, "validation_parent_salt")

    manifest = pd.DataFrame([{
        "category": "PARENT_SALT",
        "n_cases": len(out),
        "input_file": ip.name,
        "gold_file": gp.name,
        "source_file": str(SRC.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_type": "EXISTING_FROZEN_BENCHMARK",
        "independent_holdout": "NO",
        "note": "Controlled carry-over subset; do not describe as a new independent validation set.",
    }])
    write_csv(manifest, SOURCES / "parent_salt_manifest.csv")
    print(f"[OK] Parent/Salt: {len(out)} cases")
    print(f"[OUT] {ip}")
    print(f"[OUT] {gp}")


if __name__ == "__main__":
    main()
