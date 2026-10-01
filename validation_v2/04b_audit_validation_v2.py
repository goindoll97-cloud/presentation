# -*- coding: utf-8 -*-
"""Pre-freeze audit for Validation V2.

This script separates three different things:
1) automatic structural/data-integrity QC;
2) source verification documented in seeds/source_verification_evidence_v2.csv;
3) final researcher sign-off recorded in sources/validation_manual_source_review.csv.

AI-assisted source verification is never represented as human review.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from validation_common import DATA, ROOT, SEEDS, SOURCES, write_csv

AUDIT_JSON = ROOT / "VALIDATION_V2_AUDIT.json"
MANUAL_REVIEW = SOURCES / "validation_manual_source_review.csv"
SOURCE_EVIDENCE = SEEDS / "source_verification_evidence_v2.csv"

ALLOWED_LABELS = {"MATCH", "NO_MATCH", "REVIEW"}
EXPECTED_GROUP_RULES = {
    "MG_SALTS",
    "TBT_TRIALKYLTIN",
    "NONYLPHENOL_NPE",
    "LEAD_COMPOUNDS",
    "CRVI_COMPOUNDS",
}
FORBIDDEN_INPUT_COLS = {
    "gold_label", "gold_reason", "curation_status", "official_list_hit",
    "challenge_subtype", "external_source", "external_source_url",
    "component_set_exact",
}


def _add(checks, name, ok, details="", severity="FAIL"):
    status = "PASS" if ok else severity
    checks.append({"check": name, "status": status, "details": str(details)})


def _load(stem: str):
    ip = DATA / f"validation_{stem}_INPUT.csv"
    gp = DATA / f"validation_{stem}_GOLD.csv"
    if not ip.exists() or not gp.exists():
        raise FileNotFoundError(f"Missing {ip} or {gp}")
    return (
        pd.read_csv(ip, dtype=str).fillna(""),
        pd.read_csv(gp, dtype=str).fillna(""),
    )


def _fingerprint(values) -> str:
    raw = "|".join(str(v) for v in values)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _build_review_table(group_in, group_gold, mix_in, mix_gold):
    if not SOURCE_EVIDENCE.exists():
        raise FileNotFoundError(SOURCE_EVIDENCE)
    evidence = pd.read_csv(SOURCE_EVIDENCE, dtype=str).fillna("")
    if evidence["review_key"].duplicated().any():
        raise ValueError("Duplicate review_key in source-verification evidence")

    rows = []
    g = group_in.merge(
        group_gold, on=["case_id", "category"], how="inner", suffixes=("", "_gold")
    )
    for _, r in g.iterrows():
        fp = _fingerprint([
            r["case_id"], r.get("target_rule_id", ""), r.get("candidate_name", ""),
            r.get("candidate_cas", ""), r.get("gold_label", ""),
            r.get("challenge_subtype", ""),
        ])
        rows.append({
            "review_key": r["case_id"],
            "review_fingerprint": fp,
            "category": "CHEMICAL_GROUP",
            "target_scope": r.get("regulatory_scope_text", ""),
            "candidate": r.get("candidate_name", ""),
            "candidate_cas": r.get("candidate_cas", ""),
            "proposed_gold_label": r.get("gold_label", ""),
            "challenge_subtype": r.get("challenge_subtype", ""),
            "source": r.get("external_source", ""),
            "source_url": r.get("external_source_url", ""),
            "gold_reason": r.get("gold_reason", ""),
        })

    m = mix_in.merge(
        mix_gold, on=["case_id", "category"], how="inner", suffixes=("", "_gold")
    )
    # One source-level sign-off per unique target mixture identity. Pairwise
    # negatives are validated independently by component-set QC below.
    for target_id, sub in m.groupby("target_scope_id", sort=True):
        match = sub[sub["gold_label"].eq("MATCH")]
        r = match.iloc[0] if len(match) else sub.iloc[0]
        key = f"MIXTURE_TARGET:{target_id}"
        fp = _fingerprint([
            key, r.get("regulatory_scope_text", ""), r.get("candidate_name", ""),
            r.get("candidate_components", ""), "MATCH",
        ])
        rows.append({
            "review_key": key,
            "review_fingerprint": fp,
            "category": "MIXTURE",
            "target_scope": r.get("regulatory_scope_text", ""),
            "candidate": r.get("candidate_name", ""),
            "candidate_cas": r.get("candidate_cas", ""),
            "proposed_gold_label": "MATCH",
            "challenge_subtype": "EXACT_MIXTURE_IDENTITY",
            "source": r.get("external_source", ""),
            "source_url": r.get("external_source_url", ""),
            "gold_reason": r.get("gold_reason", ""),
        })

    new = pd.DataFrame(rows)
    new = new.merge(evidence, on=["review_key", "category"], how="left")

    # Preserve researcher sign-off only when the reviewed content fingerprint
    # is unchanged. Any seed/GOLD change automatically invalidates old sign-off.
    old_keep = None
    if MANUAL_REVIEW.exists():
        old = pd.read_csv(MANUAL_REVIEW, dtype=str).fillna("")
        cols = {
            "review_key", "review_fingerprint",
            "author_signoff_status", "author_signed_at_utc", "reviewer_note",
        }
        if cols.issubset(old.columns):
            old_keep = old[list(cols)].drop_duplicates("review_key")
    if old_keep is not None:
        new = new.merge(
            old_keep, on=["review_key", "review_fingerprint"], how="left"
        )

    for c, default in [
        ("author_signoff_status", "PENDING"),
        ("author_signed_at_utc", ""),
        ("reviewer_note", ""),
    ]:
        if c not in new.columns:
            new[c] = default
        else:
            new[c] = new[c].replace("", pd.NA).fillna(default)

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
    _add(
        checks, "master_input_gold_same_case_ids",
        set(ma_i.case_id) == set(ma_g.case_id),
        f"input={len(ma_i)}, gold={len(ma_g)}",
    )
    _add(
        checks, "master_case_id_unique",
        not ma_i.case_id.duplicated().any() and not ma_g.case_id.duplicated().any(),
    )
    _add(
        checks, "master_labels_valid",
        set(ma_g.gold_label).issubset(ALLOWED_LABELS),
        sorted(set(ma_g.gold_label)),
    )
    leakage = sorted(FORBIDDEN_INPUT_COLS.intersection(ma_i.columns))
    _add(checks, "master_input_no_gold_or_audit_leakage", not leakage, leakage)

    union_i = set().union(*(set(x[0].case_id) for x in category_sets.values()))
    union_g = set().union(*(set(x[1].case_id) for x in category_sets.values()))
    _add(
        checks, "master_equals_category_union",
        set(ma_i.case_id) == union_i and set(ma_g.case_id) == union_g,
    )

    # Parent/salt provenance.
    _add(checks, "parent_salt_n72", len(ps_g) == 72, len(ps_g))
    if "independent_holdout" in ps_g.columns:
        _add(
            checks, "parent_salt_marked_nonindependent",
            ps_g.independent_holdout.eq("NO").all(),
        )
    if "source_case_id" in ps_g.columns:
        _add(
            checks, "parent_salt_source_case_unique",
            not ps_g.source_case_id.duplicated().any(),
        )

    # Chemical-group logic/provenance.
    cgm = cg_i.merge(cg_g, on=["case_id", "category"], how="inner")
    _add(
        checks, "chemical_group_join_complete",
        len(cgm) == len(cg_i) == len(cg_g), len(cgm),
    )
    _add(
        checks, "chemical_group_source_urls_present",
        cgm.external_source_url.astype(bool).all(),
    )
    _add(
        checks, "chemical_group_gold_reasons_present",
        cgm.gold_reason.astype(bool).all(),
    )

    present_rules = set(cgm.target_rule_id)
    _add(
        checks, "chemical_group_all_5_rules_present",
        present_rules == EXPECTED_GROUP_RULES,
        sorted(present_rules),
    )

    per_rule_n = cgm.target_rule_id.value_counts().to_dict()
    _add(
        checks, "chemical_group_balanced_6_per_rule",
        all(per_rule_n.get(r, 0) == 6 for r in EXPECTED_GROUP_RULES),
        per_rule_n,
    )

    balance_bad = {}
    for rule_id, sub in cgm.groupby("target_rule_id"):
        vc = sub.gold_label.value_counts().to_dict()
        if vc.get("MATCH", 0) != 3 or vc.get("NO_MATCH", 0) != 3:
            balance_bad[rule_id] = vc
    _add(
        checks, "chemical_group_each_rule_3_match_3_no_match",
        not balance_bad, balance_bad,
    )

    allowed_subtypes = {
        "DIRECT_ENUMERATED_MATCH", "OPEN_SET_GROUP_MATCH",
        "HARD_NEGATIVE", "REVIEW",
    }
    _add(
        checks, "chemical_group_challenge_subtypes_valid",
        set(cgm.challenge_subtype).issubset(allowed_subtypes),
        cgm.challenge_subtype.value_counts().to_dict(),
    )

    subtype_bad = (
        ((cgm.challenge_subtype == "DIRECT_ENUMERATED_MATCH") &
         ((cgm.gold_label != "MATCH") | (cgm.official_list_hit != "YES"))) |
        ((cgm.challenge_subtype == "OPEN_SET_GROUP_MATCH") &
         ((cgm.gold_label != "MATCH") | (cgm.official_list_hit != "NO"))) |
        ((cgm.challenge_subtype == "HARD_NEGATIVE") &
         ((cgm.gold_label != "NO_MATCH") | (cgm.official_list_hit == "YES")))
    )
    _add(
        checks, "chemical_group_subtype_registry_consistency",
        not subtype_bad.any(),
        cgm.loc[subtype_bad, "case_id"].tolist(),
    )

    if "live_pubchem_qc" in cgm.columns:
        unresolved = int(cgm.live_pubchem_qc.ne("PASS").sum())
        _add(
            checks, "chemical_group_pubchem_live_qc",
            unresolved == 0, f"unresolved={unresolved}", severity="WARN",
        )

    # Mixture pair logic.
    mxm = mx_i.merge(mx_g, on=["case_id", "category"], how="inner")
    _add(
        checks, "mixture_join_complete",
        len(mxm) == len(mx_i) == len(mx_g), len(mxm),
    )
    _add(
        checks, "mixture_source_urls_present",
        mxm.external_source_url.astype(bool).all(),
    )
    match_bad = mxm.gold_label.eq("MATCH") & ~mxm.component_set_exact.eq("YES")
    no_bad = mxm.gold_label.eq("NO_MATCH") & ~mxm.component_set_exact.eq("NO")
    _add(
        checks, "mixture_match_exact_components",
        not match_bad.any(), mxm.loc[match_bad, "case_id"].tolist(),
    )
    _add(
        checks, "mixture_no_match_nonexact_components",
        not no_bad.any(), mxm.loc[no_bad, "case_id"].tolist(),
    )

    pair_fail = []
    for tid, sub in mxm.groupby("target_scope_id"):
        vc = sub.gold_label.value_counts().to_dict()
        if len(sub) != 2 or vc.get("MATCH", 0) != 1 or vc.get("NO_MATCH", 0) != 1:
            pair_fail.append({"target_scope_id": tid, "n": len(sub), "labels": vc})
    _add(
        checks, "mixture_each_target_one_match_one_hard_negative",
        not pair_fail, pair_fail,
    )
    _add(
        checks, "mixture_unique_targets_15",
        mxm.target_scope_id.nunique() >= 15, mxm.target_scope_id.nunique(),
    )

    # Source verification + researcher sign-off.
    review = _build_review_table(cg_i, cg_g, mx_i, mx_g)
    source_missing = review["source_verification_status"].ne("VERIFIED")
    _add(
        checks, "source_verification_complete",
        not source_missing.any(),
        review.loc[source_missing, "review_key"].tolist(),
    )
    ai_method = review["verification_method"].str.startswith(
        "AI_ASSISTED_SOURCE_VERIFICATION"
    )
    _add(
        checks, "source_verification_method_recorded",
        ai_method.all(),
        review.loc[~ai_method, "review_key"].tolist(),
    )

    signoff = review["author_signoff_status"].astype(str).str.upper()
    invalid_signoff = ~signoff.isin({"PENDING", "APPROVED", "REJECTED"})
    _add(
        checks, "author_signoff_status_values_valid",
        not invalid_signoff.any(),
        review.loc[invalid_signoff, "review_key"].tolist(),
    )
    pending = int(signoff.eq("PENDING").sum())
    rejected = int(signoff.eq("REJECTED").sum())
    _add(
        checks, "researcher_author_signoff_complete",
        pending == 0 and rejected == 0,
        f"approved={int(signoff.eq('APPROVED').sum())}, "
        f"pending={pending}, rejected={rejected}",
    )

    n_fail = sum(c["status"] == "FAIL" for c in checks)
    n_warn = sum(c["status"] == "WARN" for c in checks)
    overall = "PASS" if n_fail == 0 else "FAIL"
    report = {
        "protocol": "VALIDATION_V2_PRE_FREEZE_AUDIT",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "n_fail": n_fail,
        "n_warn": n_warn,
        "n_source_verified_items": int(
            review.source_verification_status.eq("VERIFIED").sum()
        ),
        "n_author_signoff_items": int(len(review)),
        "manual_review_file": str(MANUAL_REVIEW.relative_to(ROOT)).replace("\\", "/"),
        "source_evidence_file": str(SOURCE_EVIDENCE.relative_to(ROOT)).replace("\\", "/"),
        "checks": checks,
    }
    AUDIT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(pd.DataFrame(checks).to_string(index=False))
    print(f"\n[AUDIT] overall={overall} fail={n_fail} warn={n_warn}")
    print(
        f"[SOURCE VERIFIED] "
        f"{int(review.source_verification_status.eq('VERIFIED').sum())}/{len(review)}"
    )
    print(f"[AUTHOR SIGN-OFF] approved={int(signoff.eq('APPROVED').sum())}/{len(review)}")
    print(f"[REVIEW FILE] {MANUAL_REVIEW}")
    print(f"[REPORT] {AUDIT_JSON}")
    if overall != "PASS":
        raise RuntimeError(
            "Validation V2 pre-freeze audit did not pass. Resolve FAIL items "
            "before freezing. Source verification and researcher sign-off are "
            "tracked separately."
        )


if __name__ == "__main__":
    main()
