# -*- coding: utf-8 -*-
"""Study 2 / Step 04B CURATED BALANCED FINAL

Freeze the manually curated, difficulty-balanced shared-DB chemical-identity benchmark.
This is a benchmark-construction/provenance utility, not the active V4 comparator.

Inputs
------
- broad_salt_rules_CURATED_13.csv
- broad_salt_validation_cases_CURATED_BALANCED_134.csv

No Claude API calls occur in this step.

Outputs
-------
intermediate/04_identity_challenge_benchmark.csv
intermediate/04_identity_challenge_rules.csv
intermediate/04_identity_challenge_by_rule.csv
intermediate/04_identity_challenge_by_difficulty.csv
intermediate/04_identity_challenge_qc.csv
intermediate/04_identity_benchmark_readiness.json

Active FAIR V5 next steps:
    04C_prepare_dual_benchmark_METHODSAFE.py
    05_compare_identity_SHARED_DB_FAIR_V5.py
    06_make_identity_results_FAIR_V5.py
"""

from __future__ import annotations
import json, os
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
INTER = ROOT / "intermediate"
INTER.mkdir(parents=True, exist_ok=True)

RULE_FILE_CANDIDATES = [
    ROOT / "data" / "broad_salt_rules_CURATED_13.csv",
    ROOT / "broad_salt_rules_CURATED_13.csv",
]
CASE_FILE_CANDIDATES = [
    ROOT / "data" / "broad_salt_validation_cases_CURATED_BALANCED_134.csv",
    ROOT / "broad_salt_validation_cases_CURATED_BALANCED_134.csv",
]

MIN_RULES = 13
MIN_CASES = 120
MIN_CASES_PER_RULE = 10
MAX_CASES_PER_RULE = 20
MIN_POS_FRAC = 0.40
MAX_POS_FRAC = 0.60
MIN_HARD = 20
MIN_MODERATE = 60
MIN_EASY = 30
MIN_STRATUM_POS_FRAC = 0.40
MAX_STRATUM_POS_FRAC = 0.60

def clean(x):
    if x is None: return ""
    try:
        if pd.isna(x): return ""
    except Exception:
        pass
    s=str(x).strip()
    return "" if s.lower() in {"nan","none","null","<na>"} else s

def first_existing(paths):
    return next((p for p in paths if p.exists()), None)

def read_csv(path):
    return pd.read_csv(path, dtype=str).fillna("")

def main():
    print("="*100)
    print("04B CURATED BALANCED FINAL / FREEZE SHARED-DB IDENTITY BENCHMARK (NO CLAUDE)")
    print("="*100)

    rp=first_existing(RULE_FILE_CANDIDATES)
    cp=first_existing(CASE_FILE_CANDIDATES)
    if rp is None or cp is None:
        raise FileNotFoundError(
            "Place broad_salt_rules_CURATED_13.csv and "
            "broad_salt_validation_cases_CURATED_BALANCED_134.csv in repo root or data/."
        )

    rules=read_csv(rp).drop_duplicates(subset=["rule_id"],keep="first")
    cases=read_csv(cp).drop_duplicates(subset=["case_id"],keep="first")

    req_rule=["rule_id","designation_id","reference_parent_name","reference_parent_cas",
              "reference_smiles","salt_scope","regulatory_source","chemical_identity_source"]
    req_case=["case_id","rule_id","designation_id","reference_parent_name","reference_parent_cas",
              "reference_parent_smiles","chemical_name","cas","smiles","reference_membership",
              "challenge_class","difficulty","reference_source","reference_evidence",
              "benchmark_source","curation_note"]
    mr=[c for c in req_rule if c not in rules.columns]
    mc=[c for c in req_case if c not in cases.columns]
    if mr: raise ValueError(f"Missing rule columns: {mr}")
    if mc: raise ValueError(f"Missing case columns: {mc}")

    if not set(cases["reference_membership"]).issubset({"MATCH","NO_MATCH"}):
        raise ValueError("reference_membership must be MATCH or NO_MATCH only.")
    if not set(cases["difficulty"].str.upper()).issubset({"EASY","MODERATE","HARD"}):
        raise ValueError("difficulty must be EASY/MODERATE/HARD only.")
    if not set(cases["rule_id"]).issubset(set(rules["rule_id"])):
        raise ValueError("Unknown rule_id detected.")

    lut=rules.set_index("rule_id",drop=False)
    rows=[]
    for _,c in cases.iterrows():
        rr=lut.loc[c["rule_id"]]
        if isinstance(rr,pd.DataFrame): rr=rr.iloc[0]
        lab=clean(c["reference_membership"])
        rows.append({
            "identity_benchmark_id":f"ID2_{clean(c['case_id'])}",
            "case_id":clean(c["case_id"]),
            "rule_id":clean(c["rule_id"]),
            "designation_id":clean(c["designation_id"]),
            "regulatory_scope_text":clean(rr.get("regulatory_source"))
                or f"{clean(rr.get('reference_parent_name'))} ({clean(rr.get('reference_parent_cas'))}) and its salts",
            "scope_text_origin":"CURATED_BALANCED_FROZEN_RULE_SOURCE",
            "reference_parent_name":clean(c["reference_parent_name"]),
            "reference_parent_cas":clean(c["reference_parent_cas"]),
            "reference_parent_smiles":clean(c["reference_parent_smiles"]),
            "salt_scope":clean(rr.get("salt_scope")) or "all_counterion_salts",
            "candidate_name":clean(c["chemical_name"]),
            "candidate_cas":clean(c["cas"]),
            "candidate_source_smiles":clean(c["smiles"]),
            "reference_membership":lab,
            "reference_membership_bool":lab=="MATCH",
            "reference_source":clean(c["reference_source"]),
            "reference_evidence":clean(c.get("reference_evidence")),
            "reference_notes":clean(c.get("curation_note")),
            "reference_type":"SOURCE_SUPPORTED_CURATED_OPERATIONAL_REFERENCE_NOT_EXPERT_LEGAL_GOLD",
            "case_class":"POSITIVE_MEMBER" if lab=="MATCH" else "NEGATIVE_NONMEMBER",
            "challenge_class":clean(c["challenge_class"]),
            "difficulty":clean(c["difficulty"]).upper(),
            "benchmark_source":clean(c.get("benchmark_source")),
            "generation_route":"CURATED_BALANCED_MANUAL_FREEZE",
            "operational_cas_eligible":bool(clean(c["cas"])),
            "controlled_structure_eligible":bool(clean(c["smiles"]) and clean(c["reference_parent_smiles"])),
            "rule_source_verified":True,
            "rule_frozen_before_validation":True,
            "regulatory_source":clean(rr.get("regulatory_source")),
            "chemical_identity_source":clean(rr.get("chemical_identity_source")),
            "rule_version":clean(rr.get("rule_version")) or "CURATED_BALANCED_2026-09-29",
            "rule_frozen_date":clean(rr.get("frozen_date")) or "2026-09-29",
        })

    bench=pd.DataFrame(rows)

    n=len(bench); n_rules=bench["rule_id"].nunique()
    n_match=(bench["reference_membership"]=="MATCH").sum()
    n_no=(bench["reference_membership"]=="NO_MATCH").sum()
    pos_frac=n_match/max(n,1)
    per_rule=bench.groupby("rule_id").size()
    per_rule_lab=bench.groupby(["rule_id","reference_membership"]).size().unstack(fill_value=0)
    diff_tab=bench.groupby(["difficulty","reference_membership"]).size().unstack(fill_value=0)

    diff_pos={}
    for d in ["EASY","MODERATE","HARD"]:
        tot=int(diff_tab.loc[d].sum()) if d in diff_tab.index else 0
        m=int(diff_tab.loc[d].get("MATCH",0)) if d in diff_tab.index else 0
        diff_pos[d]=m/max(tot,1)

    checks={
        "min_rules":int(n_rules)>=MIN_RULES,
        "min_cases":int(n)>=MIN_CASES,
        "each_rule_10_to_20_cases":bool(((per_rule>=MIN_CASES_PER_RULE)&(per_rule<=MAX_CASES_PER_RULE)).all()),
        "each_rule_has_MATCH_and_NO_MATCH":bool((per_rule_lab.get("MATCH",0)>0).all() and (per_rule_lab.get("NO_MATCH",0)>0).all()),
        "overall_positive_fraction_0.40_to_0.60":MIN_POS_FRAC<=pos_frac<=MAX_POS_FRAC,
        "easy_n":int((bench["difficulty"]=="EASY").sum())>=MIN_EASY,
        "moderate_n":int((bench["difficulty"]=="MODERATE").sum())>=MIN_MODERATE,
        "hard_n":int((bench["difficulty"]=="HARD").sum())>=MIN_HARD,
        "easy_label_balance_0.40_to_0.60":MIN_STRATUM_POS_FRAC<=diff_pos["EASY"]<=MAX_STRATUM_POS_FRAC,
        "moderate_label_balance_0.40_to_0.60":MIN_STRATUM_POS_FRAC<=diff_pos["MODERATE"]<=MAX_STRATUM_POS_FRAC,
        "hard_label_balance_0.40_to_0.60":MIN_STRATUM_POS_FRAC<=diff_pos["HARD"]<=MAX_STRATUM_POS_FRAC,
        "all_have_reference_source":bool(bench["reference_source"].map(clean).ne("").all()),
        "all_have_cas":bool(bench["candidate_cas"].map(clean).ne("").all()),
        "all_have_candidate_structure":bool(bench["candidate_source_smiles"].map(clean).ne("").all()),
        "all_have_parent_structure":bool(bench["reference_parent_smiles"].map(clean).ne("").all()),
        "no_duplicate_case_ids":bool(~bench["identity_benchmark_id"].duplicated().any()),
        "no_duplicate_rule_cas":bool(~bench[["rule_id","candidate_cas"]].duplicated().any()),
    }
    checks = {k: bool(v) for k, v in checks.items()}
    failed=[k for k,v in checks.items() if not v]
    ready=bool(not failed)

    readiness={
        "created_at_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "benchmark_version":"CURATED_BALANCED_134_2026-09-29",
        "ready_for_claude_api":bool(ready),
        "failed_checks":failed,
        "checks":checks,
        "n_rules":int(n_rules),
        "n_cases":int(n),
        "n_match":int(n_match),
        "n_no_match":int(n_no),
        "positive_fraction":float(pos_frac),
        "n_easy":int((bench["difficulty"]=="EASY").sum()),
        "n_moderate":int((bench["difficulty"]=="MODERATE").sum()),
        "n_hard":int((bench["difficulty"]=="HARD").sum()),
        "easy_positive_fraction":float(diff_pos["EASY"]),
        "moderate_positive_fraction":float(diff_pos["MODERATE"]),
        "hard_positive_fraction":float(diff_pos["HARD"]),
        "note":"Difficulty strata are label-balanced before Claude evaluation; reference is source-supported chemical identity, not expert legal gold.",
    }

    bench.to_csv(INTER/"04_identity_challenge_benchmark.csv",index=False,encoding="utf-8-sig")
    rules.to_csv(INTER/"04_identity_challenge_rules.csv",index=False,encoding="utf-8-sig")
    bench.groupby(["reference_parent_name","rule_id","reference_membership"]).size().rename("n").reset_index().to_csv(
        INTER/"04_identity_challenge_by_rule.csv",index=False,encoding="utf-8-sig")
    bench.groupby(["difficulty","reference_membership"]).size().rename("n").reset_index().to_csv(
        INTER/"04_identity_challenge_by_difficulty.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{"metric":k,"value":v} for k,v in readiness.items() if not isinstance(v,(dict,list))]).to_csv(
        INTER/"04_identity_challenge_qc.csv",index=False,encoding="utf-8-sig")
    (INTER/"04_identity_benchmark_readiness.json").write_text(
        json.dumps(readiness,ensure_ascii=False,indent=2),encoding="utf-8")

    print(json.dumps(readiness,ensure_ascii=False,indent=2))
    print("\n[Per-parent composition]")
    print(bench.groupby(["reference_parent_name","reference_membership"]).size().unstack(fill_value=0).to_string())
    print("\n[Difficulty x label]")
    print(diff_tab.to_string())

    if ready:
        print("\n[READY] Run 04C_prepare_dual_benchmark_METHODSAFE.py next, then dry-run 05_compare_identity_SHARED_DB_FAIR_V5.py with Claude disabled.")
    else:
        print("\n[BLOCKED] Do not enable Claude. Inspect failed_checks above.")

if __name__=="__main__":
    main()
