# -*- coding: utf-8 -*-
"""Build open-set chemical-group validation data from curated external candidates.

Design principle
----------------
The curated external source is the benchmark evidence. Live PubChem retrieval is
best-effort enrichment/QC only and must not decide the GOLD label or make dataset
construction fail because of API/search-index behaviour.
"""
from __future__ import annotations

import pandas as pd

from validation_common import (
    DATA, SEEDS, SOURCES, pubchem_by_exact_cas,
    require_approved, split_input_gold, write_csv,
)
from regulatory_group_reference import OFFICIAL_CAS_BY_RULE

SEED = SEEDS / "chemical_group_candidates_seed.csv"
AUDIT = DATA / "chemical_group_candidates_enriched.csv"


def main() -> None:
    if not SEED.exists():
        raise FileNotFoundError(SEED)

    seed = pd.read_csv(SEED, dtype=str).fillna("")
    needed = [
        "seed_id", "target_rule_id", "regulatory_scope_text", "candidate_name",
        "candidate_cas", "external_source", "external_source_url", "gold_label",
        "gold_reason", "independent_evidence_note", "curation_status",
    ]
    miss = [c for c in needed if c not in seed.columns]
    if miss:
        raise ValueError(f"Missing seed columns: {miss}")

    official_pairs = {
        (rule_id, cas)
        for rule_id, members in OFFICIAL_CAS_BY_RULE.items()
        for cas in members
    }
    n_official = sum(len(v) for v in OFFICIAL_CAS_BY_RULE.values())

    enriched_rows = []
    for i, r in seed.iterrows():
        cas = str(r["candidate_cas"]).strip()
        name = str(r["candidate_name"]).strip()
        pc = pubchem_by_exact_cas(cas, candidate_name=name) if cas else {
            "external_db": "", "external_db_status": "NO_CAS",
            "pubchem_lookup_route": "",
            "pubchem_cid": "", "pubchem_title": "", "pubchem_isomeric_smiles": "",
            "pubchem_canonical_smiles": "", "pubchem_inchikey": "",
            "external_record_url": "", "retrieved_at_utc": "",
        }
        row = r.to_dict()
        row.update(pc)
        row["official_list_hit"] = (
            "YES" if (str(r["target_rule_id"]), cas) in official_pairs else "NO"
        )
        row["live_pubchem_qc"] = (
            "PASS" if (not cas or pc.get("external_db_status") == "EXACT_CAS_VERIFIED")
            else "UNRESOLVED_NONBLOCKING"
        )
        row["case_id"] = f"CG-{i+1:03d}"
        row["category"] = "CHEMICAL_GROUP"
        enriched_rows.append(row)

    enriched = pd.DataFrame(enriched_rows)
    write_csv(enriched, AUDIT)

    approved = require_approved(enriched)
    if approved.empty:
        raise RuntimeError("No APPROVED chemical-group cases. Curate the seed CSV first.")

    # Blocking QC is limited to benchmark-design errors that would invalidate
    # the curated validation set. Live DB lookup is intentionally non-blocking.
    for _, r in approved.iterrows():
        if r["gold_label"] == "MATCH" and r["official_list_hit"] != "NO":
            raise ValueError(f"Open-set MATCH unexpectedly already in official list: {r['seed_id']}")
        if not str(r["external_source_url"]).strip():
            raise ValueError(f"Approved case lacks external source URL: {r['seed_id']}")

    input_cols = [
        "case_id", "category", "target_rule_id", "regulatory_scope_text",
        "candidate_name", "candidate_cas",
    ]
    gold_cols = [
        "case_id", "category", "gold_label", "gold_reason", "curation_status",
        "official_list_hit", "external_source", "external_source_url",
        "external_db_status", "live_pubchem_qc", "pubchem_lookup_route",
        "pubchem_cid", "pubchem_title", "pubchem_isomeric_smiles",
        "pubchem_canonical_smiles", "pubchem_inchikey",
        "independent_evidence_note", "seed_id",
    ]
    ip, gp = split_input_gold(approved, input_cols, gold_cols, "validation_chemical_group")

    manifest = approved[[
        "case_id", "target_rule_id", "candidate_name", "candidate_cas",
        "external_source", "external_source_url", "official_list_hit",
        "live_pubchem_qc", "seed_id",
    ]].copy()
    manifest.insert(1, "category", "CHEMICAL_GROUP")
    write_csv(manifest, SOURCES / "chemical_group_source_manifest.csv")

    unresolved = int((approved["live_pubchem_qc"] == "UNRESOLVED_NONBLOCKING").sum())
    print(f"[OK] Chemical-group approved cases: {len(approved)}")
    print(f"[QC] Frozen closed-registry rows: {n_official}")
    print(f"[QC] Open-set official_list_hit=NO: {(approved['official_list_hit'] == 'NO').sum()}/{len(approved)}")
    print(f"[QC] PubChem live unresolved (non-blocking): {unresolved}/{len(approved)}")
    print("[QC] PubChem lookup routes:")
    print(approved["pubchem_lookup_route"].value_counts(dropna=False).to_string())
    print(f"[OUT] {AUDIT}")
    print(f"[OUT] {ip}")
    print(f"[OUT] {gp}")


if __name__ == "__main__":
    main()
