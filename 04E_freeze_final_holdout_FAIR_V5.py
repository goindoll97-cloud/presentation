# -*- coding: utf-8 -*-
"""Freeze the exact FAIR-V5 final holdout bytes BEFORE any Step-05 evaluation.

Required order
--------------
1. Freeze code/protocol with 04D.
2. Independently construct/curate the final holdout without using V5 predictions.
3. Point IDENTITY_V5_FINAL_HOLDOUT_FILE to that CSV and run THIS script.
4. Only then run Step 05 (dry run first, then paid execution).

The freeze stores SHA256 for the holdout, protocol freeze, and Step 05. Step 05
rechecks all three before any confirmatory evaluation.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
INTER = ROOT / "intermediate"
PROTOCOL_FREEZE = INTER / "04D_fair_v5_protocol_freeze.json"
DEV_BENCHMARK = INTER / "04_identity_challenge_benchmark.csv"
STEP05 = ROOT / "05_compare_identity_SHARED_DB_FAIR_V5.py"
OUT = INTER / "04E_fair_v5_holdout_freeze.json"


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


def main() -> None:
    if not PROTOCOL_FREEZE.exists():
        raise RuntimeError("Run 04D_freeze_fair_v5_protocol.py BEFORE creating/freezing the final holdout")
    if not STEP05.exists():
        raise FileNotFoundError(STEP05)

    raw = os.getenv("IDENTITY_V5_FINAL_HOLDOUT_FILE", "").strip()
    if not raw:
        raise RuntimeError("Set IDENTITY_V5_FINAL_HOLDOUT_FILE to the independently curated final holdout CSV")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p, dtype=str).fillna("")
    required = [
        "case_id", "rule_id", "candidate_cas", "reference_parent_smiles",
        "reference_membership", "reference_source",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Final holdout missing columns: {missing}")
    if df.empty:
        raise RuntimeError("Final holdout is empty")
    if df["case_id"].map(clean).duplicated().any():
        raise RuntimeError("Final holdout case_id must be unique")
    if not df["candidate_cas"].map(clean).ne("").all():
        raise RuntimeError("Every final-holdout row requires candidate_cas for the shared exact-CAS lookup")
    if not df["reference_source"].map(clean).ne("").all():
        raise RuntimeError("Every final-holdout row requires an independent reference_source")
    labels = set(df["reference_membership"].map(lambda x: clean(x).upper()))
    if not labels.issubset({"MATCH", "NO_MATCH"}):
        raise RuntimeError(f"reference_membership must be MATCH/NO_MATCH only; found {sorted(labels)}")

    hold_cas = set(df["candidate_cas"].map(clean))
    overlap = []
    if DEV_BENCHMARK.exists():
        dev = pd.read_csv(DEV_BENCHMARK, dtype=str).fillna("")
        dev_cas = {clean(x) for x in dev.get("candidate_cas", pd.Series(dtype=str)) if clean(x)}
        overlap = sorted(hold_cas & dev_cas)
        if overlap:
            raise RuntimeError(f"Final holdout candidate-CAS overlaps development data: {overlap}")

    protocol = json.loads(PROTOCOL_FREEZE.read_text(encoding="utf-8"))
    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": "STUDY2_FAIR_V5_FINAL_HOLDOUT",
        "holdout_file_name": p.name,
        "holdout_sha256": sha256(p),
        "n_rows": int(len(df)),
        "n_unique_candidate_cas": int(df["candidate_cas"].map(clean).nunique()),
        "label_counts": {str(k): int(v) for k, v in df["reference_membership"].map(lambda x: clean(x).upper()).value_counts().items()},
        "development_candidate_cas_overlap_n": int(len(overlap)),
        "development_candidate_cas_overlap": overlap,
        "protocol_frozen_at_utc": protocol.get("frozen_at_utc", ""),
        "protocol_freeze_sha256": sha256(PROTOCOL_FREEZE),
        "step05_sha256": sha256(STEP05),
        "statement": "Exact holdout bytes frozen before any FAIR-V5 Step-05 evaluation.",
    }
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\n[FROZEN FINAL HOLDOUT] {OUT}")


if __name__ == "__main__":
    main()
