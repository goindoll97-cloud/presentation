# -*- coding: utf-8 -*-
"""Build open-set chemical-group validation data from curated external candidates.

Workflow
--------
1. Read curated seed candidates.
2. Resolve CAS against PubChem using exact-CAS synonym verification.
3. Check whether the CAS is directly enumerated in the frozen closed registry.
4. Keep all rows in an audit file.
5. Export final INPUT/GOLD only for curation_status=APPROVED.

External DB identity evidence never creates a GOLD label automatically.
"""
from __future__ import annotations

import pandas as pd

from validation_common import (
    DATA, SEEDS, SOURCES, pubchem_by_exact_cas,
    require_approved, split_input_gold, write_csv,
)
from regulatory_group_reference import OFFICIAL_GROUP_MEMBERS

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
        for rule_id, members in OFFICIAL_GROUP_MEMBERS.items()
        for cas in members
    }
    n_official = sum(len(v) for v in OFFICIAL_GROUP_MEMBERS.values())

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
        row["official_list_hit"] = "YES" if (str(r["target_rule_id"]), cas) in official_pairs else "NO"
        row["case_id"] = f"CG-{i+1:03d}"
        row["category"] = "CHEMICAL_GROUP"
        enriched_rows.append(row)

    enriched = pd.DataFrame(enriched_rows)
    write_csv(enriched, AUDIT)

    approved = require_approved(enriched)
    if approved.empty:
        raise RuntimeError("No APPROVED chemical-group cases. Curate the seed CSV first.")

    # Strong QC: an approved open-set MATCH must not be an exact closed-registry hit,
    # must have an external source, and must resolve by exact CAS when a CAS is supplied.
    for _, r in approved.iterrows():
        if r["gold_label"] == "MATCH" and r["official_list_hit"] != "NO":
            raise ValueError(f"Open-set MATCH unexpectedly already in official list: {r['seed_id']}")
        if not str(r["external_source_url"]).strip():
            raise ValueError(f"Approved case lacks external source URL: {r['seed_id']}")
        if str(r["candidate_cas"]).strip() and r["external_db_status"] != "EXACT_CAS_VERIFIED":
            raise ValueError(
                f"Approved CAS case not exact-verified in PubChem: {r['seed_id']} "
                f"status={r['external_db_status']} route={r.get('pubchem_lookup_route', '')}"
            )

    input_cols = [
        "case_id", "category", "target_rule_id", "regulatory_scope_text",
        "candidate_name", "candidate_cas",
    ]
    gold_cols = [
        "case_id", "category", "gold_label", "gold_reason", "curation_status",
        "official_list_hit", "external_source", "external_source_url",
        "external_db_status", "pubchem_lookup_route", "pubchem_cid", "pubchem_title",
        "pubchem_isomeric_smiles", "pubchem_canonical_smiles", "pubchem_inchikey",
        "independent_evidence_note", "seed_id",
    ]
    ip, gp = split_input_gold(approved, input_cols, gold_cols, "validation_chemical_group")

    manifest = approved[[
        "case_id", "target_rule_id", "candidate_name", "candidate_cas",
        "external_source", "external_source_url", "official_list_hit", "seed_id",
    ]].copy()
    manifest.insert(1, "category", "CHEMICAL_GROUP")
    write_csv(manifest, SOURCES / "chemical_group_source_manifest.csv")

    print(f"[OK] Chemical-group approved cases: {len(approved)}")
    print(f"[QC] Frozen closed-registry rows: {n_official}")
    print(f"[QC] Open-set official_list_hit=NO: {(approved['official_list_hit'] == 'NO').sum()}/{len(approved)}")
    print("[QC] PubChem lookup routes:")
    print(approved["pubchem_lookup_route"].value_counts(dropna=False).to_string())
    print(f"[OUT] {AUDIT}")
    print(f"[OUT] {ip}")
    print(f"[OUT] {gp}")


if __name__ == "__main__":
    main()
