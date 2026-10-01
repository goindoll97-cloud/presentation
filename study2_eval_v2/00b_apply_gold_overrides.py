# -*- coding: utf-8 -*-
"""Apply documented researcher corrections to the derived retrieval GOLD only.

The derived NOT_FOUND label comes from pairwise NO_MATCH rows and can be wrong
when a candidate actually belongs to another catalog target. Because 01 rebuilds
retrieval_GOLD.csv from frozen Validation V2 on every run, a correction must live
in a separate, versioned file rather than as a manual edit of the GOLD CSV.

review/retrieval_gold_overrides.csv (researcher-authored; created empty if absent)
  cas_inputs           query CAS signature as shown in the review file
  from_gold_target_id  the derived GOLD value being replaced (must match exactly)
  to_gold_target_id    catalog target_id or NOT_FOUND
  evidence             regulatory/chemical basis for the correction (required)
  researcher_note      free text

Every overridden row is marked RESEARCHER_OVERRIDE and requires a fresh researcher
approval in 01a/01b. Frozen Validation V2 source files are never modified.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REVIEW_DIR = ROOT / "review"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)
QINPUT = DATA / "retrieval_INPUT.csv"
QGOLD = DATA / "retrieval_GOLD.csv"
CATALOG = DATA / "regulatory_catalog.csv"
MANIFEST = DATA / "retrieval_build_manifest.json"
OVERRIDES = REVIEW_DIR / "retrieval_gold_overrides.csv"
COLUMNS = ["cas_inputs", "from_gold_target_id", "to_gold_target_id", "evidence", "researcher_note"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    for p in [QINPUT, QGOLD, CATALOG, MANIFEST]:
        if not p.exists():
            raise FileNotFoundError(p)
    if not OVERRIDES.exists():
        pd.DataFrame(columns=COLUMNS).to_csv(OVERRIDES, index=False, encoding="utf-8-sig")

    qin = pd.read_csv(QINPUT, dtype=str).fillna("")
    gold = pd.read_csv(QGOLD, dtype=str).fillna("")
    catalog = pd.read_csv(CATALOG, dtype=str).fillna("")
    ov = pd.read_csv(OVERRIDES, dtype=str).fillna("")
    missing = [c for c in COLUMNS if c not in ov.columns]
    if missing:
        raise RuntimeError(f"{OVERRIDES.name} is missing columns: {missing}")

    sig_to_qid = dict(zip(qin["cas_inputs"].map(clean), qin["query_id"].map(clean)))
    allowed = set(catalog["target_id"].map(clean)) | {"NOT_FOUND"}
    if ov["cas_inputs"].map(clean).duplicated().any():
        raise RuntimeError("Duplicate cas_inputs in GOLD override file")

    applied = []
    for r in ov.itertuples(index=False):
        sig, src, dst = clean(r.cas_inputs), clean(r.from_gold_target_id), clean(r.to_gold_target_id)
        qid = sig_to_qid.get(sig)
        if not qid:
            raise RuntimeError(f"GOLD override signature not in retrieval INPUT: {sig}")
        idx = gold.index[gold["query_id"].eq(qid)][0]
        current = clean(gold.at[idx, "gold_target_id"])
        if current != src:
            raise RuntimeError(
                f"GOLD override for {qid} expects from={src} but derived GOLD is {current}; "
                "the source changed, re-check this override"
            )
        if dst not in allowed or dst == src:
            raise RuntimeError(f"Invalid GOLD override target for {qid}: {src} -> {dst}")
        if not clean(r.evidence):
            raise RuntimeError(f"GOLD override for {qid} requires evidence")
        gold.at[idx, "gold_target_id"] = dst
        gold.at[idx, "gold_status"] = "NOT_FOUND" if dst == "NOT_FOUND" else "FOUND"
        gold.at[idx, "gold_derivation"] = f"RESEARCHER_OVERRIDE_FROM_{src}"
        gold.at[idx, "retrieval_scope_review_required"] = "YES"
        applied.append({"query_id": qid, "cas_inputs": sig, "from": src, "to": dst})

    gold.to_csv(QGOLD, index=False, encoding="utf-8-sig")
    build = json.loads(MANIFEST.read_text(encoding="utf-8"))
    build["gold_overrides"] = applied
    build["gold_override_policy"] = (
        "Derived retrieval GOLD may be corrected only through a versioned override file with "
        "required evidence; every overridden row requires fresh researcher approval. All overrides "
        "must be reported, including whether they were prompted by a Hybrid-gate conflict."
    )
    build["sha256"].update({
        "retrieval_GOLD.csv": sha256(QGOLD),
        "review/retrieval_gold_overrides.csv": sha256(OVERRIDES),
    })
    MANIFEST.write_text(json.dumps(build, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[GOLD OVERRIDES] applied={len(applied)}")
    for a in applied:
        print(f"  {a['query_id']} {a['cas_inputs']}: {a['from']} -> {a['to']}")


if __name__ == "__main__":
    main()
