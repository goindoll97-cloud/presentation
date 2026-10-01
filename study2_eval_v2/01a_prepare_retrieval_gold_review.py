# -*- coding: utf-8 -*-
"""Prepare researcher review table for derived retrieval GOLD.

Only retrieval-scope claims that are stronger than the frozen pairwise benchmark
require new approval. In practice, derived NOT_FOUND (and any MULTI_TARGET)
rows require explicit human review against the complete frozen catalog.

Existing approvals are preserved only when query_id, CAS signature, derived GOLD
and review requirement are unchanged.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REVIEW_DIR = ROOT / "review"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)

QINPUT = DATA / "retrieval_INPUT.csv"
QGOLD = DATA / "retrieval_GOLD.csv"
CATALOG = DATA / "regulatory_catalog.csv"
OUT = REVIEW_DIR / "retrieval_gold_manual_review.csv"

KEYS = ["query_id", "cas_inputs", "derived_gold_target_id", "review_required"]


def clean(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def main() -> None:
    for p in [QINPUT, QGOLD, CATALOG]:
        if not p.exists():
            raise FileNotFoundError(p)

    qin = pd.read_csv(QINPUT, dtype=str).fillna("")
    gold = pd.read_csv(QGOLD, dtype=str).fillna("")
    catalog = pd.read_csv(CATALOG, dtype=str).fillna("")
    if set(qin["query_id"]) != set(gold["query_id"]):
        raise RuntimeError("Retrieval INPUT/GOLD query sets differ")

    df = qin.merge(gold, on="query_id", how="inner", validate="one_to_one")
    out = pd.DataFrame({
        "query_id": df["query_id"],
        "cas_inputs": df["cas_inputs"],
        "derived_gold_target_id": df["gold_target_id"],
        "gold_status": df["gold_status"],
        "review_required": df["retrieval_scope_review_required"].apply(
            lambda x: "YES" if clean(x).upper() == "YES" else "NO"
        ),
        "source_categories": df["source_categories"],
        "source_case_ids": df["source_case_ids"],
        "gold_derivation": df["gold_derivation"],
        "catalog_target_count": str(len(catalog)),
        "review_instruction": df["retrieval_scope_review_required"].apply(
            lambda x: (
                "Confirm that this CAS/CAS-set belongs to NONE of the 29 frozen catalog targets; "
                "review regulatory_catalog.csv target-by-target."
                if clean(x).upper() == "YES"
                else "No new approval required: target is inherited from an already verified frozen MATCH row."
            )
        ),
        "researcher_status": df["retrieval_scope_review_required"].apply(
            lambda x: "PENDING" if clean(x).upper() == "YES" else "NOT_REQUIRED"
        ),
        "researcher_note": "",
    })

    if OUT.exists():
        prev = pd.read_csv(OUT, dtype=str).fillna("")
        needed = set(KEYS + ["researcher_status", "researcher_note"])
        if needed.issubset(prev.columns):
            old = {
                tuple(clean(r[k]) for k in KEYS): (
                    clean(r["researcher_status"]).upper(), clean(r["researcher_note"])
                )
                for _, r in prev.iterrows()
            }
            for idx, r in out.iterrows():
                if r["review_required"] != "YES":
                    continue
                key = tuple(clean(r[k]) for k in KEYS)
                status, note = old.get(key, ("PENDING", ""))
                if status in {"APPROVED", "REJECTED", "PENDING"}:
                    out.at[idx, "researcher_status"] = status
                    out.at[idx, "researcher_note"] = note

    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    required = out[out["review_required"].eq("YES")]
    counts = required["researcher_status"].value_counts().to_dict()
    print(f"[RETRIEVAL GOLD REVIEW] required={len(required)} status={counts}")
    print(f"[OUT] {OUT}")


if __name__ == "__main__":
    main()
