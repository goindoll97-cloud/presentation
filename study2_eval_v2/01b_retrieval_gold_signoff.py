# -*- coding: utf-8 -*-
"""Explicit researcher sign-off for derived retrieval GOLD review rows.

Run only after manually inspecting review/retrieval_gold_manual_review.csv and
regulatory_catalog.csv. This script never changes retrieval_GOLD.csv; it records
human approval of the stronger retrieval-scope NOT_FOUND claims.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
REVIEW = ROOT / "review" / "retrieval_gold_manual_review.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--approve", action="store_true", help="Approve all currently PENDING required review rows")
    args = ap.parse_args()

    if not REVIEW.exists():
        raise FileNotFoundError("Run 01a_prepare_retrieval_gold_review.py first")
    df = pd.read_csv(REVIEW, dtype=str).fillna("")
    required = df["review_required"].astype(str).str.upper().eq("YES")

    if args.approve:
        pending = required & df["researcher_status"].astype(str).str.upper().eq("PENDING")
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
