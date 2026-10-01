# -*- coding: utf-8 -*-
"""Build real-mixture validation data from curated external-source records.

The mixture subset is identity/composition oriented. It supports CAS-less
mixtures and explicit component lists. Concentration-threshold experiments can
be added later as a separate challenge class without changing this file format.
"""
from __future__ import annotations

import pandas as pd

from validation_common import DATA, SEEDS, SOURCES, require_approved, split_input_gold, write_csv

SEED = SEEDS / "mixture_candidates_seed.csv"
AUDIT = DATA / "mixture_candidates_curated.csv"


def _norm_parts(x: str) -> list[str]:
    return [p.strip() for p in str(x).split("|") if p.strip()]


def main() -> None:
    if not SEED.exists():
        raise FileNotFoundError(SEED)
    seed = pd.read_csv(SEED, dtype=str).fillna("")
    needed = [
        "seed_id", "target_scope_id", "regulatory_scope_text", "target_components",
        "candidate_name", "candidate_cas", "candidate_components", "external_source",
        "external_source_url", "gold_label", "gold_reason", "curation_status",
    ]
    miss = [c for c in needed if c not in seed.columns]
    if miss:
        raise ValueError(f"Missing seed columns: {miss}")

    rows = []
    for i, r in seed.iterrows():
        row = r.to_dict()
        t = _norm_parts(r["target_components"])
        c = _norm_parts(r["candidate_components"])
        row.update({
            "case_id": f"MX-{i+1:03d}",
            "category": "MIXTURE",
            "target_component_count": len(t),
            "candidate_component_count": len(c),
            "candidate_has_cas": "YES" if str(r["candidate_cas"]).strip() else "NO",
            "component_set_exact": "YES" if set(x.lower() for x in t) == set(x.lower() for x in c) and t else "NO",
        })
        rows.append(row)
    curated = pd.DataFrame(rows)
    write_csv(curated, AUDIT)

    approved = require_approved(curated)
    if approved.empty:
        raise RuntimeError("No APPROVED mixture cases. Curate the seed CSV first.")

    for _, r in approved.iterrows():
        if not str(r["external_source_url"]).strip():
            raise ValueError(f"Approved mixture case lacks external source URL: {r['seed_id']}")
        if not str(r["candidate_name"]).strip():
            raise ValueError(f"Approved mixture case lacks candidate_name: {r['seed_id']}")
        if r["gold_label"] == "MATCH" and r["component_set_exact"] != "YES":
            raise ValueError(
                f"V2 starter MATCH mixture must have exact curated component set: {r['seed_id']}"
            )

    input_cols = [
        "case_id", "category", "target_scope_id", "regulatory_scope_text",
        "candidate_name", "candidate_cas", "candidate_components",
    ]
    gold_cols = [
        "case_id", "category", "gold_label", "gold_reason", "curation_status",
        "target_components", "target_component_count", "candidate_component_count",
        "candidate_has_cas", "component_set_exact", "external_source",
        "external_source_url", "seed_id",
    ]
    ip, gp = split_input_gold(approved, input_cols, gold_cols, "validation_mixture")

    manifest = approved[[
        "case_id", "target_scope_id", "candidate_name", "candidate_cas",
        "external_source", "external_source_url", "seed_id",
    ]].copy()
    manifest.insert(1, "category", "MIXTURE")
    write_csv(manifest, SOURCES / "mixture_source_manifest.csv")

    print(f"[OK] Mixture approved cases: {len(approved)}")
    print(f"[QC] CAS-less candidates: {(approved['candidate_has_cas'] == 'NO').sum()}/{len(approved)}")
    print(f"[OUT] {AUDIT}")
    print(f"[OUT] {ip}")
    print(f"[OUT] {gp}")


if __name__ == "__main__":
    main()
