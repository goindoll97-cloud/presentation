# -*- coding: utf-8 -*-
"""Build real-mixture validation data from curated external-source records.

The mixture subset is identity/composition oriented. It supports CAS-less
mixtures and explicit component lists. Multiple curated seed files can be kept
separately for provenance; all files matching mixture_candidates_seed*.csv are
combined deterministically before INPUT/GOLD export.
"""
from __future__ import annotations

import pandas as pd

from validation_common import DATA, SEEDS, SOURCES, require_approved, split_input_gold, write_csv

SEED_GLOB = "mixture_candidates_seed*.csv"
AUDIT = DATA / "mixture_candidates_curated.csv"


def _norm_parts(x: str) -> list[str]:
    return [p.strip() for p in str(x).split("|") if p.strip()]


def main() -> None:
    seed_files = sorted(SEEDS.glob(SEED_GLOB))
    if not seed_files:
        raise FileNotFoundError(f"No mixture seed files matching {SEEDS / SEED_GLOB}")

    frames = []
    for path in seed_files:
        part = pd.read_csv(path, dtype=str).fillna("")
        part["seed_file"] = path.name
        frames.append(part)
    seed = pd.concat(frames, ignore_index=True)

    needed = [
        "seed_id", "target_scope_id", "regulatory_scope_text", "target_components",
        "candidate_name", "candidate_cas", "candidate_components", "external_source",
        "external_source_url", "gold_label", "gold_reason", "curation_status",
    ]
    miss = [c for c in needed if c not in seed.columns]
    if miss:
        raise ValueError(f"Missing seed columns: {miss}")
    if seed["seed_id"].duplicated().any():
        dup = seed.loc[seed["seed_id"].duplicated(keep=False), "seed_id"].tolist()
        raise ValueError(f"Duplicate mixture seed_id across seed files: {dup}")

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
        raise RuntimeError("No APPROVED mixture cases. Curate the seed CSVs first.")

    for _, r in approved.iterrows():
        if not str(r["external_source_url"]).strip():
            raise ValueError(f"Approved mixture case lacks external source URL: {r['seed_id']}")
        if not str(r["candidate_name"]).strip():
            raise ValueError(f"Approved mixture case lacks candidate_name: {r['seed_id']}")
        if r["gold_label"] == "MATCH" and r["component_set_exact"] != "YES":
            raise ValueError(
                f"Approved MATCH mixture must have exact curated component set: {r['seed_id']}"
            )

    input_cols = [
        "case_id", "category", "target_scope_id", "regulatory_scope_text",
        "candidate_name", "candidate_cas", "candidate_components",
    ]
    gold_cols = [
        "case_id", "category", "gold_label", "gold_reason", "curation_status",
        "target_components", "target_component_count", "candidate_component_count",
        "candidate_has_cas", "component_set_exact", "external_source",
        "external_source_url", "seed_id", "seed_file",
    ]
    ip, gp = split_input_gold(approved, input_cols, gold_cols, "validation_mixture")

    manifest = approved[[
        "case_id", "target_scope_id", "candidate_name", "candidate_cas",
        "external_source", "external_source_url", "seed_id", "seed_file",
    ]].copy()
    manifest.insert(1, "category", "MIXTURE")
    write_csv(manifest, SOURCES / "mixture_source_manifest.csv")

    print(f"[OK] Mixture approved cases: {len(approved)}")
    print(f"[QC] Seed files: {len(seed_files)} ({', '.join(p.name for p in seed_files)})")
    print(f"[QC] Unique target mixture identities: {approved['target_scope_id'].nunique()}")
    print(f"[QC] MATCH/NO_MATCH/REVIEW: {approved['gold_label'].value_counts().to_dict()}")
    print(f"[QC] CAS-less candidates: {(approved['candidate_has_cas'] == 'NO').sum()}/{len(approved)}")
    print(f"[OUT] {AUDIT}")
    print(f"[OUT] {ip}")
    print(f"[OUT] {gp}")


if __name__ == "__main__":
    main()
