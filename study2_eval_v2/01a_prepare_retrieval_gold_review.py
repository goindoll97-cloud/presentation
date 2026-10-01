# -*- coding: utf-8 -*-
"""Prepare researcher review table for derived retrieval GOLD.

Only retrieval-scope claims that are stronger than the frozen pairwise benchmark
require new approval. Derived NOT_FOUND (and any MULTI_TARGET) rows therefore
require explicit human review against the complete frozen catalog.

The review file may contain source candidate names because it is a human-only QC
artifact and is never supplied to either evaluated system.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DATA = ROOT / "data"
VAL_INPUT = REPO_ROOT / "validation_v2" / "data" / "validation_master_INPUT.csv"
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


def split_set(x) -> set[str]:
    return {clean(v) for v in clean(x).split("|") if clean(v)}


def exact_catalog_hits(signature: str, catalog: pd.DataFrame) -> str:
    q = split_set(signature)
    hits = []
    for r in catalog.itertuples(index=False):
        tid = clean(r.target_id)
        refs = split_set(getattr(r, "reference_cas_set", ""))
        members = split_set(getattr(r, "official_member_cas_set", ""))
        comps = split_set(getattr(r, "mixture_component_cas_set", ""))
        if len(q) == 1 and (q & refs or q & members):
            hits.append(tid)
        if len(q) > 1 and comps and q == comps:
            hits.append(tid)
    return "|".join(sorted(set(hits)))


def main() -> None:
    for p in [QINPUT, QGOLD, CATALOG, VAL_INPUT]:
        if not p.exists():
            raise FileNotFoundError(p)

    qin = pd.read_csv(QINPUT, dtype=str).fillna("")
    gold = pd.read_csv(QGOLD, dtype=str).fillna("")
    catalog = pd.read_csv(CATALOG, dtype=str).fillna("")
    vin = pd.read_csv(VAL_INPUT, dtype=str).fillna("")
    if set(qin["query_id"]) != set(gold["query_id"]):
        raise RuntimeError("Retrieval INPUT/GOLD query sets differ")

    name_map = {}
    for r in vin.itertuples(index=False):
        cid = clean(getattr(r, "case_id", ""))
        name = clean(getattr(r, "candidate_name", ""))
        if cid:
            name_map[cid] = name

    df = qin.merge(gold, on="query_id", how="inner", validate="one_to_one")
    source_names = []
    exact_hits = []
    for _, r in df.iterrows():
        case_ids = [clean(v) for v in clean(r["source_case_ids"]).split("|") if clean(v)]
        source_names.append("|".join(sorted({name_map.get(c, "") for c in case_ids if name_map.get(c, "")})))
        exact_hits.append(exact_catalog_hits(clean(r["cas_inputs"]), catalog))

    out = pd.DataFrame({
        "query_id": df["query_id"],
        "cas_inputs": df["cas_inputs"],
        "source_candidate_names_human_only": source_names,
        "derived_gold_target_id": df["gold_target_id"],
        "gold_status": df["gold_status"],
        "review_required": df["retrieval_scope_review_required"].apply(
            lambda x: "YES" if clean(x).upper() == "YES" else "NO"
        ),
        "source_categories": df["source_categories"],
        "source_case_ids": df["source_case_ids"],
        "gold_derivation": df["gold_derivation"],
        "catalog_exact_hit_precheck": exact_hits,
        "catalog_target_count": str(len(catalog)),
        "review_instruction": df["retrieval_scope_review_required"].apply(
            lambda x: (
                "Confirm that this CAS/CAS-set belongs to NONE of the 29 frozen catalog targets. "
                "Check regulatory_catalog.csv target-by-target and investigate any exact-hit precheck."
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
    conflicts = required[[
        bool(h) and g not in h.split("|")
        for h, g in zip(required["catalog_exact_hit_precheck"], required["derived_gold_target_id"])
    ]]
    print(f"[RETRIEVAL GOLD REVIEW] required={len(required)} status={counts}")
    print(f"[PRECHECK] review-required rows whose CAS hits a different catalog target={len(conflicts)}")
    if len(conflicts):
        print(conflicts[["query_id", "cas_inputs", "catalog_exact_hit_precheck"]].to_string(index=False))
    print(f"[OUT] {OUT}")


if __name__ == "__main__":
    main()
