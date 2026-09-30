# -*- coding: utf-8 -*-
"""Study 2 / Step 04C METHOD-SAFE dual benchmark freeze.

Purpose
-------
Repair the benchmark-design issue identified at preflight without discarding prior work.

Set A / PRIMARY_SOURCE_CURATED
    Original frozen 60-case benchmark (5 parent rules; 30 MATCH / 30 NO_MATCH).
    Labels were source-curated before the expanded structure-QC benchmark was built.
    This is the PRIMARY inferential benchmark for LLM vs deterministic comparison.

Set B / SECONDARY_STRUCTURE_ANCHORED
    Expanded 134-case benchmark (13 parent rules; 64 MATCH / 70 NO_MATCH).
    Inclusion used structure-consistency QC. Therefore this set is retained as a
    structure-anchored coverage/stress benchmark, NOT as an unbiased estimate of
    general RDKit performance.

The two sets overlap. Step 05 should evaluate each unique rule/CAS pair only once,
then map the same prediction back to both benchmark sets. This keeps cost down and
prevents duplicated model calls for the same chemical identity question.

No Claude API calls occur in this script.
"""

from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
INTER = ROOT / "intermediate"
INTER.mkdir(parents=True, exist_ok=True)

PRIMARY_CASE_FILES = [
    ROOT / "broad_salt_validation_cases(1).csv",
    ROOT / "data" / "broad_salt_validation_cases(1).csv",
    ROOT / "broad_salt_validation_cases.csv",
]
PRIMARY_RULE_FILES = [
    ROOT / "broad_salt_rules(1).csv",
    ROOT / "data" / "broad_salt_rules(1).csv",
    ROOT / "broad_salt_rules.csv",
]
SECONDARY_CASE_FILES = [
    ROOT / "broad_salt_validation_cases_CURATED_BALANCED_134.csv",
    ROOT / "data" / "broad_salt_validation_cases_CURATED_BALANCED_134.csv",
]
SECONDARY_RULE_FILES = [
    ROOT / "broad_salt_rules_CURATED_13.csv",
    ROOT / "data" / "broad_salt_rules_CURATED_13.csv",
]

def clean(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    return "" if s.lower() in {"nan","none","null","<na>"} else s

def first_existing(paths):
    return next((p for p in paths if p.exists()), None)

def load(paths, label):
    p = first_existing(paths)
    if p is None:
        raise FileNotFoundError(f"Missing {label}. Expected one of: {[str(x) for x in paths]}")
    return pd.read_csv(p, dtype=str).fillna(""), p

def parent_majority_baseline(df, parent_col="reference_parent_name"):
    tab = pd.crosstab(df[parent_col], df["reference_membership"])
    if len(df) == 0:
        return float("nan")
    return float(tab.max(axis=1).sum() / len(df))

def main():
    print("="*104)
    print("04C METHOD-SAFE / DUAL BENCHMARK FREEZE (PRIMARY SOURCE-CURATED + SECONDARY STRUCTURE-ANCHORED)")
    print("="*104)

    pcase, pcase_path = load(PRIMARY_CASE_FILES, "primary 60-case benchmark")
    prule, prule_path = load(PRIMARY_RULE_FILES, "primary rule file")
    scase, scase_path = load(SECONDARY_CASE_FILES, "secondary 134-case benchmark")
    srule, srule_path = load(SECONDARY_RULE_FILES, "secondary rule file")

    # Rule metadata: secondary 13-rule file is the superset.
    rule_lut = srule.set_index("rule_id", drop=False)
    primary_rule_lut = prule.set_index("rule_id", drop=False)

    # ---------------- Primary normalization ----------------
    p = pcase.copy()
    p["reference_parent_name"] = p["rule_id"].map(prule.set_index("rule_id")["reference_parent_name"])
    p["reference_parent_cas"] = p["rule_id"].map(prule.set_index("rule_id")["reference_parent_cas"])
    p["reference_parent_smiles"] = p["rule_id"].map(prule.set_index("rule_id")["reference_smiles"])
    p["challenge_class"] = "SOURCE_CURATED_CORE"
    p["difficulty"] = "UNSTRATIFIED"
    p["benchmark_set"] = "PRIMARY_SOURCE_CURATED"
    p["benchmark_role"] = "PRIMARY_INFERENTIAL"
    p["rdkit_structure_qc_used_for_inclusion"] = False
    p["reference_type"] = "SOURCE_SUPPORTED_OPERATIONAL_REFERENCE_NOT_EXPERT_LEGAL_GOLD"
    p["curation_note"] = p["notes"].map(clean)
    p["stereo_scope_sensitivity_flag"] = False

    # The source-curated R-warfarin sodium case has an intentionally conservative
    # stereo-scope sensitivity flag. Its MATCH label is retained, but results should
    # also be reported with this case excluded.
    p.loc[
        p["chemical_name"].str.contains(r"\(R\)-Warfarin sodium", case=False, regex=True, na=False),
        "stereo_scope_sensitivity_flag"
    ] = True
    p.loc[
        p["stereo_scope_sensitivity_flag"].astype(bool),
        "curation_note"
    ] = (
        p.loc[p["stereo_scope_sensitivity_flag"].astype(bool), "curation_note"].astype(str)
        + " CURATION NOTE: reference Warfarin is stereochemically unspecified; "
          "the stereospecific sodium salt is retained as a curated MATCH and must be included in a sensitivity analysis."
    )

    # ---------------- Secondary normalization ----------------
    s = scase.copy()
    s["benchmark_set"] = "SECONDARY_STRUCTURE_ANCHORED"
    s["benchmark_role"] = "SECONDARY_COVERAGE_STRESS"
    s["rdkit_structure_qc_used_for_inclusion"] = True
    s["reference_type"] = "SOURCE_SUPPORTED_PLUS_STRUCTURE_CONSISTENCY_REFERENCE_NOT_EXPERT_LEGAL_GOLD"
    s["stereo_scope_sensitivity_flag"] = s["case_id"].eq("BAL048")

    s.loc[s["case_id"].eq("BAL048"), "curation_note"] = (
        "CURATED STEREO-SCOPE MATCH: reference Warfarin is stereochemically unspecified, "
        "whereas the candidate is (R)-Warfarin sodium. Retained as MATCH by curation; "
        "report a sensitivity analysis excluding this case."
    )
    s.loc[s["case_id"].eq("BAL078"), "curation_note"] = (
        "SOURCE-SUPPORTED NEUTRAL-DRAWN ACID-BASE SALT: Dehydroabietylammonium "
        "pentachlorophenoxide is represented with neutral disconnected PCP and amine fragments. "
        "This case is a known representation stressor for FragmentParent-only deterministic logic."
    )

    # Save transparent set-specific snapshots.
    p.to_csv(INTER/"04_primary_source_curated_60.csv", index=False, encoding="utf-8-sig")
    s.to_csv(INTER/"04_secondary_structure_anchored_134.csv", index=False, encoding="utf-8-sig")

    # ---------------- Membership map ----------------
    membership_rows = []
    for _, r in p.iterrows():
        membership_rows.append({
            "benchmark_set":"PRIMARY_SOURCE_CURATED",
            "original_case_id":clean(r["case_id"]),
            "rule_id":clean(r["rule_id"]),
            "candidate_cas":clean(r["cas"]),
            "reference_membership":clean(r["reference_membership"]),
            "reference_parent_name":clean(r["reference_parent_name"]),
            "difficulty":"UNSTRATIFIED",
            "challenge_class":"SOURCE_CURATED_CORE",
            "stereo_scope_sensitivity_flag":bool(r["stereo_scope_sensitivity_flag"]),
        })
    for _, r in s.iterrows():
        membership_rows.append({
            "benchmark_set":"SECONDARY_STRUCTURE_ANCHORED",
            "original_case_id":clean(r["case_id"]),
            "rule_id":clean(r["rule_id"]),
            "candidate_cas":clean(r["cas"]),
            "reference_membership":clean(r["reference_membership"]),
            "reference_parent_name":clean(r["reference_parent_name"]),
            "difficulty":clean(r["difficulty"]).upper(),
            "challenge_class":clean(r["challenge_class"]),
            "stereo_scope_sensitivity_flag":bool(r["stereo_scope_sensitivity_flag"]),
        })
    membership = pd.DataFrame(membership_rows)

    # ---------------- Unique evaluation cases ----------------
    # CAS-less source-curated case gets a case-specific key and remains non-operational.
    def eval_key(row):
        cas = clean(row.get("cas"))
        if cas:
            return f"{clean(row.get('rule_id'))}|CAS|{cas}"
        return f"{clean(row.get('rule_id'))}|NO_CAS|{clean(row.get('case_id'))}"

    source_rows = []
    for source_set, df in [
        ("PRIMARY_SOURCE_CURATED", p),
        ("SECONDARY_STRUCTURE_ANCHORED", s),
    ]:
        for _, r in df.iterrows():
            d = r.to_dict()
            d["_source_set"] = source_set
            d["_eval_key"] = eval_key(d)
            source_rows.append(d)
    all_rows = pd.DataFrame(source_rows)

    # Overlap labels must agree.
    conflicts = []
    for key, g in all_rows.groupby("_eval_key"):
        labs = sorted(set(g["reference_membership"].map(clean)))
        if len(labs) > 1:
            conflicts.append((key, labs))
    if conflicts:
        raise RuntimeError(f"Reference-label conflict across benchmark sets: {conflicts[:10]}")

    unique_rows = []
    for i, (key, g) in enumerate(all_rows.groupby("_eval_key", sort=True), start=1):
        # Prefer PRIMARY representation for overlaps because it is the primary source-curated set.
        g = g.copy()
        primary = g[g["_source_set"].eq("PRIMARY_SOURCE_CURATED")]
        r = primary.iloc[0] if len(primary) else g.iloc[0]

        rid = clean(r["rule_id"])
        if rid in rule_lut.index:
            rr = rule_lut.loc[rid]
        elif rid in primary_rule_lut.index:
            rr = primary_rule_lut.loc[rid]
        else:
            raise RuntimeError(f"Missing rule metadata: {rid}")
        if isinstance(rr, pd.DataFrame):
            rr = rr.iloc[0]

        pids = g.loc[g["_source_set"].eq("PRIMARY_SOURCE_CURATED"), "case_id"].map(clean).tolist()
        sids = g.loc[g["_source_set"].eq("SECONDARY_STRUCTURE_ANCHORED"), "case_id"].map(clean).tolist()
        in_p = bool(pids)
        in_s = bool(sids)

        # For overlap cases, secondary difficulty/challenge metadata is retained separately.
        sec_rows = g[g["_source_set"].eq("SECONDARY_STRUCTURE_ANCHORED")]
        sec_diff = clean(sec_rows.iloc[0].get("difficulty")) if len(sec_rows) else ""
        sec_chal = clean(sec_rows.iloc[0].get("challenge_class")) if len(sec_rows) else ""

        cas = clean(r.get("cas"))
        candidate_smiles = clean(r.get("smiles"))
        ref_smiles = clean(r.get("reference_parent_smiles")) or clean(rr.get("reference_smiles"))
        label = clean(r.get("reference_membership"))

        unique_rows.append({
            "identity_benchmark_id":f"ID2U_{i:03d}",
            "case_id":f"DUAL{i:03d}",
            "eval_key":key,
            "rule_id":rid,
            "designation_id":clean(r.get("designation_id")) or clean(rr.get("designation_id")),
            "regulatory_scope_text":clean(rr.get("regulatory_source"))
                or f"{clean(rr.get('reference_parent_name'))} ({clean(rr.get('reference_parent_cas'))}) and its salts",
            "scope_text_origin":"FROZEN_RULE_SOURCE",
            "reference_parent_name":clean(r.get("reference_parent_name")) or clean(rr.get("reference_parent_name")),
            "reference_parent_cas":clean(r.get("reference_parent_cas")) or clean(rr.get("reference_parent_cas")),
            "reference_parent_smiles":ref_smiles,
            "salt_scope":clean(rr.get("salt_scope")) or "all_counterion_salts",
            "candidate_name":clean(r.get("chemical_name")),
            "candidate_cas":cas,
            "candidate_source_smiles":candidate_smiles,
            "reference_membership":label,
            "reference_membership_bool":label == "MATCH",
            "reference_source":clean(r.get("reference_source")),
            "reference_evidence":clean(r.get("reference_evidence")) or clean(r.get("notes")),
            "reference_notes":clean(r.get("curation_note")) or clean(r.get("notes")),
            "reference_type":clean(r.get("reference_type")) or "SOURCE_SUPPORTED_OPERATIONAL_REFERENCE_NOT_EXPERT_LEGAL_GOLD",
            "case_class":"POSITIVE_MEMBER" if label=="MATCH" else "NEGATIVE_NONMEMBER",
            "challenge_class":sec_chal if in_s else "SOURCE_CURATED_CORE",
            "difficulty":sec_diff if in_s else "UNSTRATIFIED",
            "benchmark_sets":"PRIMARY_SOURCE_CURATED;SECONDARY_STRUCTURE_ANCHORED" if (in_p and in_s)
                             else ("PRIMARY_SOURCE_CURATED" if in_p else "SECONDARY_STRUCTURE_ANCHORED"),
            "in_primary_source_curated":in_p,
            "in_secondary_structure_anchored":in_s,
            "primary_case_id":";".join(pids),
            "secondary_case_id":";".join(sids),
            "secondary_difficulty":sec_diff,
            "secondary_challenge_class":sec_chal,
            "stereo_scope_sensitivity_flag":bool(g.get("stereo_scope_sensitivity_flag", pd.Series(False)).astype(bool).any()),
            "rdkit_structure_qc_used_for_secondary_inclusion":bool(in_s),
            "operational_cas_eligible":bool(cas),
            "controlled_structure_eligible":bool(candidate_smiles and ref_smiles),
            "rule_source_verified":True,
            "rule_frozen_before_validation":True,
            "regulatory_source":clean(rr.get("regulatory_source")),
            "chemical_identity_source":clean(rr.get("chemical_identity_source")),
            "rule_version":clean(rr.get("rule_version")),
            "rule_frozen_date":clean(rr.get("frozen_date")),
        })

    unique_df = pd.DataFrame(unique_rows)

    # Map unique case id back to membership rows.
    key_to_case = dict(zip(unique_df["eval_key"], unique_df["case_id"]))
    membership["eval_key"] = membership.apply(
        lambda r: f"{clean(r['rule_id'])}|CAS|{clean(r['candidate_cas'])}"
        if clean(r["candidate_cas"])
        else f"{clean(r['rule_id'])}|NO_CAS|{clean(r['original_case_id'])}",
        axis=1
    )
    membership["case_id"] = membership["eval_key"].map(key_to_case)

    # ---------------- Readiness / design diagnostics ----------------
    primary_all = p.copy()
    primary_op = p[p["cas"].map(clean).ne("")].copy()
    secondary_all = s.copy()

    readiness = {
        "created_at_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "benchmark_version":"METHODSAFE_DUAL_2026-09-30",
        "ready_for_claude_api":True,
        "failed_checks":[],
        "n_unique_eval_cases":int(len(unique_df)),
        "n_unique_operational_cases":int(unique_df["operational_cas_eligible"].astype(bool).sum()),
        "n_primary_total":int(len(primary_all)),
        "n_primary_operational":int(len(primary_op)),
        "n_secondary_total":int(len(secondary_all)),
        "n_overlap_rule_cas":int(
            len(set(zip(p["rule_id"],p["cas"])) & set(zip(s["rule_id"],s["cas"])))
        ),
        "primary_match":int((primary_all["reference_membership"]=="MATCH").sum()),
        "primary_no_match":int((primary_all["reference_membership"]=="NO_MATCH").sum()),
        "secondary_match":int((secondary_all["reference_membership"]=="MATCH").sum()),
        "secondary_no_match":int((secondary_all["reference_membership"]=="NO_MATCH").sum()),
        "primary_parent_majority_label_baseline_all":parent_majority_baseline(primary_all),
        "primary_parent_majority_label_baseline_operational":parent_majority_baseline(primary_op),
        "secondary_parent_majority_label_baseline":parent_majority_baseline(secondary_all),
        "label_conflicts_across_sets":0,
        "primary_role":"PRIMARY_INFERENTIAL_SOURCE_CURATED",
        "secondary_role":"SECONDARY_STRUCTURE_ANCHORED_COVERAGE_STRESS",
        "methodological_note":(
            "The 134-case secondary set used structure-consistency QC for inclusion and must not be presented "
            "as an unbiased estimate of generic RDKit performance. The original 60-case source-curated set is "
            "the primary inferential comparator benchmark. Predictions are made once per unique identity pair."
        ),
    }

    # Core files consumed by Step 05.
    unique_df.to_csv(INTER/"04_identity_challenge_benchmark.csv", index=False, encoding="utf-8-sig")
    srule.to_csv(INTER/"04_identity_challenge_rules.csv", index=False, encoding="utf-8-sig")
    membership.to_csv(INTER/"04_dual_benchmark_membership.csv", index=False, encoding="utf-8-sig")
    (INTER/"04_identity_benchmark_readiness.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame([{"metric":k,"value":v} for k,v in readiness.items()
                  if not isinstance(v,(dict,list))]).to_csv(
        INTER/"04_methodsafe_design_qc.csv", index=False, encoding="utf-8-sig"
    )

    print(json.dumps(readiness, ensure_ascii=False, indent=2))
    print("\nPrimary parent x label:")
    print(pd.crosstab(primary_all["reference_parent_name"], primary_all["reference_membership"]).to_string())
    print("\nSecondary parent x label:")
    print(pd.crosstab(secondary_all["reference_parent_name"], secondary_all["reference_membership"]).to_string())
    print("\n[READY] Next run Step 05 METHOD-SAFE in dry-run mode.")

if __name__ == "__main__":
    main()