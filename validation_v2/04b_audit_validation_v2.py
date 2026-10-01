# -*- coding: utf-8 -*-
"""Pre-freeze audit for Validation V2.

This script performs structural/data-integrity QC and creates a manual source-review
checklist for the externally curated CHEMICAL_GROUP and MIXTURE subsets.

Important: automatic QC cannot establish legal/chemical truth by itself. The manual
review checklist must be completed before the paper-facing freeze.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from validation_common import DATA, ROOT, SOURCES, write_csv

AUDIT_JSON = ROOT / "VALIDATION_V2_AUDIT.json"
MANUAL_REVIEW = SOURCES / "validation_manual_source_review.csv"
ALLOWED_LABELS = {"MATCH", "NO_MATCH", "REVIEW"}
FORBIDDEN_INPUT_COLS = {
    "gold_label", "gold_reason", "curation_status", "official_list_hit",
    "external_source", "external_source_url", "component_set_exact",
}


def _add(checks, name, ok, details="", severity="FAIL"):
    status = "PASS" if ok else severity
    checks.append({"check": name, "status": status, "details": str(details)})


def _load(stem: str):
    ip = DATA / f"validation_{stem}_INPUT.csv"
    gp = DATA / f"validation_{stem}_GOLD.csv"
    if not ip.exists() or not gp.exists():
        raise FileNotFoundError(f"Missing {ip} or {gp}")
    return pd.read_csv(ip, dtype=str).fillna(""), pd.read_csv(gp, dtype=str).fillna("")


def _build_manual_review(group_in, group_gold, mix_in, mix_gold):
    rows = []

    g = group_in.merge(group_gold, on=["case_id", "category"], how="inner", suffixes=("", "_gold"))
    for _, r in g.iterrows():
        rows.append({
            "review_key": r["case_id"],
            "category": "CHEMICAL_GROUP",
            "target_scope": r.get("regulatory_scope_text", ""),
            "candidate": r.get("candidate_name", ""),
            "candidate_cas": r.get("candidate_cas", ""),
            "proposed_gold_label": r.get("gold_label", ""),
            "source": r.get("external_source", ""),
            "source_url": r.get("external_source_url", ""),
            "gold_reason": r.get("gold_reason", ""),
            "manual_review_status": "PENDING",
            "reviewer_note": "",
        })

    m = mix_in.merge(mix_gold, on=["case_id", "category"], how="inner", suffixes=("", "_gold"))
    # One source-level review per unique target mixture identity; pairwise negatives are
    # separately checked by automatic component-set QC below.
    for target_id, sub in m.groupby("target_scope_id", sort=True):
        match = sub[sub["gold_label"].eq("MATCH")]
        r = match.iloc[0] if len(match) else sub.iloc[0]
        rows.append({
            "review_key": f"MIXTURE_TARGET:{target_id}",
            "category": "MIXTURE",
            "target_scope": r.get("regulatory_scope_text", ""),
            "candidate": r.get("candidate_name", ""),
            "candidate_cas": r.get("candidate_cas", ""),
            "proposed_gold_label": "MATCH",
            "source": r.get("external_source", ""),
            "source_url": r.get("external_source_url", ""),
            "gold_reason": r.get("gold_reason", ""),
            "manual_review_status": "PENDING",
            "reviewer_note": "",
        })

    new = pd.DataFrame(rows)
    # Preserve prior decisions when rerun after manual review.
    if MANUAL_REVIEW.exists():
        old = pd.read_csv(MANUAL_REVIEW, dtype=str).fillna("")
        if {"review_key", "manual_review_status", "reviewer_note"}.issubset(old.columns):
            keep = old[["review_key", "manual_review_status", "reviewer_note"]].drop_duplicates("review_key")
            new = new.drop(columns=["manual_review_status", "reviewer_note"]).merge(keep, on="review_key", how="left")
            new["manual_review_status"] = new["manual_review_status"].replace("", pd.NA).fillna("PENDING")
            new["reviewer_note"] = new["reviewer_note"].fillna("")
    write_csv(new, MANUAL_REVIEW)
    return new


def main():
    checks = []
    ps_i, ps_g = _load("parent_salt")
    cg_i, cg_g = _load("chemical_group")
    mx_i, mx_g = _load("mixture")
    ma_i, ma_g = _load("master")

    category_sets = {
        "PARENT_SALT": (ps_i, ps_g),
        "CHEMICAL_GROUP": (cg_i, cg_g),
        "MIXTURE": (mx_i, mx_g),
    }

    # Global integrity.
    _add(checks, "master_input_gold_same_case_ids", set(ma_i.case_id) == set(ma_g.case_id), f"input={len(ma_i)}, gold={len(ma_g)}")
    _add(checks, "master_case_id_unique", not ma_i.case_id.duplicated().any() and not ma_g.case_id.duplicated().any())
    _add(checks, "master_labels_valid", set(ma_g.gold_label).issubset(ALLOWED_LABELS), sorted(set(ma_g.gold_label)))
    leakage = sorted(FORBIDDEN_INPUT_COLS.intersection(ma_i.columns))
    _add(checks, "master_input_no_gold_or_audit_leakage", not leakage, leakage)

    union_i = set().union(*(set(x[0].case_id) for x in category_sets.values()))
    union_g = set().union(*(set(x[1].case_id) for x in category_sets.values()))
    _add(checks, "master_equals_category_union", set(ma_i.case_id) == union_i and set(ma_g.case_id) == union_g)

    # Parent/salt provenance.
    _add(checks, "parent_salt_n72", len(ps_g) == 72, len(ps_g))
    if "independent_holdout" in ps_g.columns:
        _add(checks, "parent_salt_marked_nonindependent", ps_g.independent_holdout.eq("NO").all())
    if "source_case_id" in ps_g.columns:
        _add(checks, "parent_salt_source_case_unique", not ps_g.source_case_id.duplicated().any())

    # Chemical-group logic/provenance.
    cgm = cg_i.merge(cg_g, on=["case_id", "category"], how="inner")
    _add(checks, "chemical_group_join_complete", len(cgm) == len(cg_i) == len(cg_g), len(cgm))
    _add(checks, "chemical_group_source_urls_present", cgm.external_source_url.astype(bool).all())
    _add(checks, "chemical_group_gold_reasons_present", cgm.gold_reason.astype(bool).all())
    bad_open_match = cgm.gold_label.eq("MATCH") & ~cgm.official_list_hit.eq("NO")
    _add(checks, "chemical_group_open_set_match_not_exact_registry_hit", not bad_open_match.any(), cgm.loc[bad_open_match, "case_id"].tolist())
    _add(checks, "chemical_group_min_3_rules", cgm.target_rule_id.nunique() >= 3, cgm.target_rule_id.value_counts().to_dict())
    if "live_pubchem_qc" in cgm.columns:
        unresolved = int(cgm.live_pubchem_qc.ne("RESOLVED").sum())
        _add(checks, "chemical_group_pubchem_live_qc", unresolved == 0, f"unresolved={unresolved}", severity="WARN")

    # Mixture pair logic.
    mxm = mx_i.merge(mx_g, on=["case_id", "category"], how="inner")
    _add(checks, "mixture_join_complete", len(mxm) == len(mx_i) == len(mx_g), len(mxm))
    _add(checks, "mixture_source_urls_present", mxm.external_source_url.astype(bool).all())
    match_bad = mxm.gold_label.eq("MATCH") & ~mxm.component_set_exact.eq("YES")
    no_bad = mxm.gold_label.eq("NO_MATCH") & ~mxm.component_set_exact.eq("NO")
    _add(checks, "mixture_match_exact_components", not match_bad.any(), mxm.loc[match_bad, "case_id"].tolist())
    _add(checks, "mixture_no_match_nonexact_components", not no_bad.any(), mxm.loc[no_bad, "case_id"].tolist())

    pair_fail = []
    for tid, sub in mxm.groupby("target_scope_id"):
        vc = sub.gold_label.value_counts().to_dict()
        if len(sub) != 2 or vc.get("MATCH", 0) != 1 or vc.get("NO_MATCH", 0) != 1:
            pair_fail.append({"target_scope_id": tid, "n": len(sub), "labels": vc})
    _add(checks, "mixture_each_target_one_match_one_hard_negative", not pair_fail, pair_fail)
    _add(checks, "mixture_unique_targets_15", mxm.target_scope_id.nunique() >= 15, mxm.target_scope_id.nunique())

    review = _build_manual_review(cg_i, cg_g, mx_i, mx_g)
    statuses = review.manual_review_status.astype(str).str.upper()
    invalid_status = ~statuses.isin({"PENDING", "APPROVED", "REJECTED"})
    _add(checks, "manual_review_status_values_valid", not invalid_status.any(), review.loc[invalid_status, "review_key"].tolist())
    pending = int(statuses.eq("PENDING").sum())
    rejected = int(statuses.eq("REJECTED").sum())
    _add(checks, "manual_source_review_complete", pending == 0 and rejected == 0, f"approved={int(statuses.eq('APPROVED').sum())}, pending={pending}, rejected={rejected}")

    n_fail = sum(c["status"] == "FAIL" for c in checks)
    n_warn = sum(c["status"] == "WARN" for c in checks)
    overall = "PASS" if n_fail == 0 else "FAIL"
    report = {
        "protocol": "VALIDATION_V2_PRE_FREEZE_AUDIT",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "n_fail": n_fail,
        "n_warn": n_warn,
        "n_manual_review_items": int(len(review)),
        "manual_review_file": str(MANUAL_REVIEW.relative_to(ROOT)).replace("\\", "/"),
        "checks": checks,
    }
    AUDIT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(pd.DataFrame(checks).to_string(index=False))
    print(f"\n[AUDIT] overall={overall} fail={n_fail} warn={n_warn}")
    print(f"[MANUAL REVIEW] {MANUAL_REVIEW}")
    print(f"[REPORT] {AUDIT_JSON}")
    if overall != "PASS":
        raise RuntimeError("Validation V2 pre-freeze audit did not pass. Resolve FAIL items before freezing.")


if __name__ == "__main__":
    main()
