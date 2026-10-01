# -*- coding: utf-8 -*-
"""Build balanced chemical-group validation data from curated candidates.

Design principle
----------------
The curated source evidence establishes the benchmark GOLD label. Live PubChem
retrieval is best-effort enrichment/QC only and never creates a GOLD label.

The V2 paper-facing subset intentionally mixes:
- DIRECT_ENUMERATED_MATCH: candidate CAS is explicitly listed in Appendix 3;
- OPEN_SET_GROUP_MATCH: candidate belongs to the generic group but its CAS is
  not directly enumerated, exercising the catch-all/generic-scope problem;
- HARD_NEGATIVE: chemically related or confusable candidate outside the scope.
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
    if seed["seed_id"].duplicated().any():
        raise ValueError("Duplicate chemical-group seed_id")

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
        list_hit = "YES" if (str(r["target_rule_id"]), cas) in official_pairs else "NO"
        row["official_list_hit"] = list_hit
        label = str(r["gold_label"]).strip().upper()
        if label == "MATCH" and list_hit == "YES":
            subtype = "DIRECT_ENUMERATED_MATCH"
        elif label == "MATCH":
            subtype = "OPEN_SET_GROUP_MATCH"
        elif label == "NO_MATCH":
            subtype = "HARD_NEGATIVE"
        else:
            subtype = "REVIEW"
        row["challenge_subtype"] = subtype
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

    # Blocking QC: source must exist and a NO_MATCH must never be an exact
    # official-list hit for the same target group.
    for _, r in approved.iterrows():
        if not str(r["external_source_url"]).strip():
            raise ValueError(f"Approved case lacks external source URL: {r['seed_id']}")
        if r["gold_label"] == "NO_MATCH" and r["official_list_hit"] == "YES":
            raise ValueError(
                f"NO_MATCH contradicts exact official-list membership: {r['seed_id']}"
            )

    input_cols = [
        "case_id", "category", "target_rule_id", "regulatory_scope_text",
        "candidate_name", "candidate_cas",
    ]
    gold_cols = [
        "case_id", "category", "gold_label", "gold_reason", "curation_status",
        "challenge_subtype", "official_list_hit", "external_source",
        "external_source_url", "external_db_status", "live_pubchem_qc",
        "pubchem_lookup_route", "pubchem_cid", "pubchem_title",
        "pubchem_isomeric_smiles", "pubchem_canonical_smiles", "pubchem_inchikey",
        "independent_evidence_note", "seed_id",
    ]
    ip, gp = split_input_gold(
        approved, input_cols, gold_cols, "validation_chemical_group"
    )

    manifest = approved[[
        "case_id", "target_rule_id", "candidate_name", "candidate_cas",
        "gold_label", "challenge_subtype", "external_source",
        "external_source_url", "official_list_hit", "live_pubchem_qc", "seed_id",
    ]].copy()
    manifest.insert(1, "category", "CHEMICAL_GROUP")
    write_csv(manifest, SOURCES / "chemical_group_source_manifest.csv")

    unresolved = int(
        (approved["live_pubchem_qc"] == "UNRESOLVED_NONBLOCKING").sum()
    )
    print(f"[OK] Chemical-group approved cases: {len(approved)}")
    print(f"[QC] Frozen closed-registry rows: {n_official}")
    print(f"[QC] Rule distribution: {approved['target_rule_id'].value_counts().to_dict()}")
    print(f"[QC] Label distribution: {approved['gold_label'].value_counts().to_dict()}")
    print(f"[QC] Challenge subtypes: {approved['challenge_subtype'].value_counts().to_dict()}")
    print(f"[QC] PubChem live unresolved (non-blocking): {unresolved}/{len(approved)}")
    print(f"[OUT] {AUDIT}")
    print(f"[OUT] {ip}")
    print(f"[OUT] {gp}")


if __name__ == "__main__":
    main()
