# -*- coding: utf-8 -*-
"""Researcher sign-off helper for Validation V2.

Run 04b first. Inspect sources/validation_manual_source_review.csv and the cited
evidence. Then run this script with --approve to record an explicit researcher
sign-off for the current fingerprints only.

This is deliberately separate from AI-assisted source verification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from validation_common import SOURCES, write_csv

REVIEW = SOURCES / "validation_manual_source_review.csv"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--approve",
        action="store_true",
        help="Approve all currently source-verified review items after researcher inspection.",
    )
    p.add_argument(
        "--note",
        default="Researcher reviewed the source-verification table and accepted the proposed GOLD labels.",
    )
    args = p.parse_args()

    if not REVIEW.exists():
        raise FileNotFoundError(
            f"{REVIEW} not found. Run 04b_audit_validation_v2.py first."
        )

    df = pd.read_csv(REVIEW, dtype=str).fillna("")
    needed = {
        "review_key", "review_fingerprint", "source_verification_status",
        "author_signoff_status", "author_signed_at_utc", "reviewer_note",
    }
    missing = needed.difference(df.columns)
    if missing:
        raise ValueError(f"Review file is missing columns: {sorted(missing)}")

    unverified = df["source_verification_status"].ne("VERIFIED")
    if unverified.any():
        bad = df.loc[unverified, "review_key"].tolist()
        raise RuntimeError(
            f"Cannot sign off: source verification incomplete for {bad}"
        )

    print(f"[REVIEW ITEMS] {len(df)}")
    print(df["category"].value_counts().to_string())
    if "proposed_gold_label" in df.columns:
        print("[PROPOSED GOLD]")
        print(df["proposed_gold_label"].value_counts().to_string())
    print(f"[SOURCE VERIFIED] {(~unverified).sum()}/{len(df)}")

    if not args.approve:
        print(
            "\nNo sign-off was written. Inspect "
            "sources/validation_manual_source_review.csv, then rerun with --approve."
        )
        return

    now = datetime.now(timezone.utc).isoformat()
    df["author_signoff_status"] = "APPROVED"
    df["author_signed_at_utc"] = now
    df["reviewer_note"] = args.note
    write_csv(df, REVIEW)

    print(f"[SIGNED OFF] {len(df)}/{len(df)}")
    print(f"[OUT] {REVIEW}")
    print(
        "Rerun 04b_audit_validation_v2.py. If all checks pass, "
        "05_freeze_validation_v2.py may be run."
    )


if __name__ == "__main__":
    main()
