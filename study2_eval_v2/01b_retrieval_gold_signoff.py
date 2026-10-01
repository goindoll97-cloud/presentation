# -*- coding: utf-8 -*-
"""Explicit researcher sign-off for derived retrieval GOLD review rows.

Run only after manually inspecting review/retrieval_gold_manual_review.csv and
regulatory_catalog.csv. This script never changes retrieval_GOLD.csv; it records
human approval of the stronger retrieval-scope NOT_FOUND claims.

Bulk approval is blocked when a review-required row has an exact catalog hit on a
target other than its derived GOLD. Such a row may indicate that the derived
NOT_FOUND label is wrong. Resolve it either by
  * adding a GOLD correction to review/retrieval_gold_overrides.csv and rerunning 01, or
  * approving that single row with a written justification:
      python 01b_retrieval_gold_signoff.py --approve-conflict RQ-0001 --note "why the hit does not apply"
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
REVIEW = ROOT / "review" / "retrieval_gold_manual_review.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--approve", action="store_true", help="Approve all currently PENDING required review rows after manual inspection")
    ap.add_argument("--approve-conflict", metavar="QUERY_ID",
                    help="Approve one exact-hit conflict row as-is; requires --note")
    ap.add_argument("--note", default="", help="Researcher justification for --approve-conflict")
    args = ap.parse_args()

    if not REVIEW.exists():
        raise FileNotFoundError("Run 01a_prepare_retrieval_gold_review.py first")
    df = pd.read_csv(REVIEW, dtype=str).fillna("")
    required = df["review_required"].astype(str).str.upper().eq("YES")
    hits = df.get("catalog_exact_hit_precheck", pd.Series("", index=df.index)).astype(str)
    is_conflict = pd.Series([
        bool(h) and g not in h.split("|")
        for h, g in zip(hits, df["derived_gold_target_id"].astype(str))
    ], index=df.index)

    if args.approve_conflict:
        note = args.note.strip()
        if not note:
            raise RuntimeError("--approve-conflict requires a non-empty --note justification")
        row = required & df["query_id"].eq(args.approve_conflict) & is_conflict
        if not row.any():
            raise RuntimeError(f"{args.approve_conflict} is not a review-required exact-hit conflict row")
        df.loc[row, "researcher_status"] = "APPROVED"
        df.loc[row, "researcher_note"] = "CONFLICT_RESOLVED: " + note
        df.to_csv(REVIEW, index=False, encoding="utf-8-sig")

    if args.approve:
        pending = required & df["researcher_status"].astype(str).str.upper().eq("PENDING")
        exact_conflict = pending & is_conflict
        if exact_conflict.any():
            rows = df.loc[
                exact_conflict,
                ["query_id", "cas_inputs", "derived_gold_target_id", "catalog_exact_hit_precheck"]
            ]
            print("[BLOCKED] Cannot bulk-approve rows whose CAS hits a different catalog target:")
            print(rows.to_string(index=False))
            raise RuntimeError(
                "Resolve each conflict via review/retrieval_gold_overrides.csv or "
                "--approve-conflict QUERY_ID --note ..., then rerun --approve."
            )

        df.loc[pending, "researcher_status"] = "APPROVED"
        df.loc[pending & df["researcher_note"].eq(""), "researcher_note"] = (
            "Researcher reviewed this derived retrieval-scope claim against the complete frozen catalog."
        )
        df.to_csv(REVIEW, index=False, encoding="utf-8-sig")

    req = df[required].copy()
    counts = req["researcher_status"].astype(str).str.upper().value_counts().to_dict()
    print(f"[RETRIEVAL GOLD SIGN-OFF] required={len(req)} status={counts}")
    if not args.approve:
        print("Use --approve only after manual review of every required row.")


if __name__ == "__main__":
    main()
