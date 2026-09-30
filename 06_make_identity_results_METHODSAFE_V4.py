# -*- coding: utf-8 -*-
"""Study 2 / Step 06 METHOD-SAFE results.

Reads Step 05 predictions made once per unique identity pair and reports:
- PRIMARY_SOURCE_CURATED: original source-curated 60-case set (59 CAS-operational)
- SECONDARY_STRUCTURE_ANCHORED: expanded 134-case structure-QC set
- micro metrics + parent-macro metrics
- within-benchmark parent-majority label baseline
- difficulty results for the secondary set
- sensitivity excluding stereochemically ambiguous R-warfarin sodium
- frozen RDKit primary vs post-preflight V4.3 sensitivity kept separate

Selective accuracy/recall/specificity use only MATCH/NO_MATCH decisions.
Overall correct resolution counts REVIEW as unresolved and uses all cases.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INTER = ROOT / "intermediate"
PRED_FILE = INTER / "05_identity_system_predictions.csv"
MEMBERSHIP_FILE = INTER / "04_dual_benchmark_membership.csv"
READINESS_FILE = INTER / "04_identity_benchmark_readiness.json"

PRIMARY_SYSTEMS = ["CLAUDE_DB", "RDKIT_CAS_LOOKUP", "HYBRID_OPERATIONAL"]
SENSITIVITY_SYSTEMS = ["RDKIT_V43_SENSITIVITY", "HYBRID_V43_SENSITIVITY"]

def clean(x):
    if x is None: return ""
    try:
        if pd.isna(x): return ""
    except Exception:
        pass
    s=str(x).strip()
    return "" if s.lower() in {"nan","none","null","<na>"} else s

def as_bool(x):
    if isinstance(x,(bool,np.bool_)): return bool(x)
    return clean(x).lower() in {"1","true","yes","y"}

def selective_metrics(g: pd.DataFrame) -> dict:
    n=len(g)
    d=g[g["decided"].map(as_bool)].copy()
    coverage=len(d)/n if n else np.nan
    overall_correct_resolution=float(
        (g["decided"].map(as_bool) & g["correct"].map(as_bool)).mean()
    ) if n else np.nan
    if len(d)==0:
        return {
            "n_total":n,"n_decided":0,"coverage":coverage,
            "accuracy":np.nan,"precision":np.nan,"recall":np.nan,
            "specificity":np.nan,"balanced_accuracy":np.nan,
            "false_safe_rate":np.nan,"false_positive_rate":np.nan,
            "overall_correct_resolution_rate":overall_correct_resolution,
            "TP":0,"TN":0,"FP":0,"FN":0,
        }
    t=d["truth_bool"].map(as_bool)
    p=d["pred_bool"].map(as_bool)
    tp=int((t&p).sum()); tn=int((~t&~p).sum())
    fp=int((~t&p).sum()); fn=int((t&~p).sum())
    rec=tp/(tp+fn) if tp+fn else np.nan
    spec=tn/(tn+fp) if tn+fp else np.nan
    ba=np.nanmean([rec,spec]) if (np.isfinite(rec) or np.isfinite(spec)) else np.nan
    return {
        "n_total":n,"n_decided":len(d),"coverage":coverage,
        "accuracy":(tp+tn)/len(d),
        "precision":tp/(tp+fp) if tp+fp else np.nan,
        "recall":rec,
        "specificity":spec,
        "balanced_accuracy":ba,
        "false_safe_rate":fn/(tp+fn) if tp+fn else np.nan,
        "false_positive_rate":fp/(tn+fp) if tn+fp else np.nan,
        "overall_correct_resolution_rate":overall_correct_resolution,
        "TP":tp,"TN":tn,"FP":fp,"FN":fn,
    }

def parent_majority_label_baseline(membership: pd.DataFrame, set_name: str) -> dict:
    g=membership[membership["benchmark_set"].eq(set_name)].copy()
    # Operational comparison only: remove the one CAS-missing primary case.
    g=g[g["candidate_cas"].map(clean).ne("")].copy()
    tab=pd.crosstab(g["reference_parent_name"],g["reference_membership"])
    correct=int(tab.max(axis=1).sum()) if len(tab) else 0
    return {
        "benchmark_set":set_name,
        "n_operational":len(g),
        "parent_majority_correct":correct,
        "parent_majority_label_accuracy":correct/len(g) if len(g) else np.nan,
        "interpretation":"Within-benchmark label-imbalance sanity baseline; not a deployable trained model.",
    }

def add_set_label(pred: pd.DataFrame, set_name: str) -> pd.DataFrame:
    if set_name=="PRIMARY_SOURCE_CURATED":
        mask=pred["in_primary_source_curated"].map(as_bool)
    elif set_name=="SECONDARY_STRUCTURE_ANCHORED":
        mask=pred["in_secondary_structure_anchored"].map(as_bool)
    else:
        raise ValueError(set_name)
    out=pred[mask].copy()
    out["benchmark_set"]=set_name
    return out

def main():
    if not PRED_FILE.exists():
        raise FileNotFoundError("Run Step 05 paid evaluation first; 05_identity_system_predictions.csv is missing.")
    if not MEMBERSHIP_FILE.exists():
        raise FileNotFoundError("Run 04C_prepare_dual_benchmark_METHODSAFE.py first.")

    pred=pd.read_csv(PRED_FILE).fillna("")
    membership=pd.read_csv(MEMBERSHIP_FILE).fillna("")
    readiness=json.loads(READINESS_FILE.read_text(encoding="utf-8")) if READINESS_FILE.exists() else {}

    sets=["PRIMARY_SOURCE_CURATED","SECONDARY_STRUCTURE_ANCHORED"]

    # 1) Overall set-specific metrics by repeat/system.
    perf=[]
    for set_name in sets:
        sg=add_set_label(pred,set_name)
        for (system,rep),g in sg.groupby(["system","repeat"],sort=False):
            perf.append({"benchmark_set":set_name,"system":system,"repeat":int(rep),**selective_metrics(g)})
    perf=pd.DataFrame(perf)
    perf.to_csv(INTER/"06_methodsafe_performance_by_repeat.csv",index=False,encoding="utf-8-sig")

    agg_rows=[]
    metric_cols=[
        "coverage","accuracy","precision","recall","specificity","balanced_accuracy",
        "false_safe_rate","false_positive_rate","overall_correct_resolution_rate",
    ]
    for (set_name,system),g in perf.groupby(["benchmark_set","system"],sort=False):
        row={"benchmark_set":set_name,"system":system,"n_repeats":g["repeat"].nunique(),
             "n_total":int(g["n_total"].max())}
        for c in metric_cols:
            x=pd.to_numeric(g[c],errors="coerce")
            row[c+"_mean"]=float(x.mean()) if x.notna().any() else np.nan
            row[c+"_sd"]=float(x.std(ddof=1)) if x.notna().sum()>1 else 0.0 if x.notna().sum()==1 else np.nan
        agg_rows.append(row)
    agg=pd.DataFrame(agg_rows)
    agg.to_csv(INTER/"06_methodsafe_performance_mean.csv",index=False,encoding="utf-8-sig")

    # 2) Parent-level metrics, then macro averages across parents.
    parent_rows=[]
    macro_rows=[]
    for set_name in sets:
        sg=add_set_label(pred,set_name)
        for (system,rep,parent),g in sg.groupby(["system","repeat","reference_parent_name"],sort=False):
            m=selective_metrics(g)
            parent_rows.append({
                "benchmark_set":set_name,"system":system,"repeat":int(rep),
                "reference_parent_name":parent,**m
            })
    parent_df=pd.DataFrame(parent_rows)
    parent_df.to_csv(INTER/"06_methodsafe_performance_by_parent.csv",index=False,encoding="utf-8-sig")

    for (set_name,system,rep),g in parent_df.groupby(["benchmark_set","system","repeat"],sort=False):
        macro_rows.append({
            "benchmark_set":set_name,"system":system,"repeat":int(rep),
            "n_parents":g["reference_parent_name"].nunique(),
            "macro_parent_accuracy":pd.to_numeric(g["accuracy"],errors="coerce").mean(),
            "macro_parent_balanced_accuracy":pd.to_numeric(g["balanced_accuracy"],errors="coerce").mean(),
            "macro_parent_coverage":pd.to_numeric(g["coverage"],errors="coerce").mean(),
            "macro_parent_correct_resolution":pd.to_numeric(g["overall_correct_resolution_rate"],errors="coerce").mean(),
        })
    macro=pd.DataFrame(macro_rows)
    macro.to_csv(INTER/"06_methodsafe_parent_macro_by_repeat.csv",index=False,encoding="utf-8-sig")

    macro_mean=(
        macro.groupby(["benchmark_set","system"],as_index=False)
        .agg(
            n_repeats=("repeat","nunique"),
            macro_parent_accuracy_mean=("macro_parent_accuracy","mean"),
            macro_parent_accuracy_sd=("macro_parent_accuracy","std"),
            macro_parent_balanced_accuracy_mean=("macro_parent_balanced_accuracy","mean"),
            macro_parent_balanced_accuracy_sd=("macro_parent_balanced_accuracy","std"),
            macro_parent_coverage_mean=("macro_parent_coverage","mean"),
            macro_parent_correct_resolution_mean=("macro_parent_correct_resolution","mean"),
        )
    )
    macro_mean.to_csv(INTER/"06_methodsafe_parent_macro_mean.csv",index=False,encoding="utf-8-sig")

    # 3) Secondary difficulty analysis only; do not mix it with the primary source-curated set.
    sec=add_set_label(pred,"SECONDARY_STRUCTURE_ANCHORED")
    diff_rows=[]
    for (system,rep,difficulty),g in sec.groupby(["system","repeat","secondary_difficulty"],sort=False,dropna=False):
        if not clean(difficulty):
            continue
        diff_rows.append({
            "benchmark_set":"SECONDARY_STRUCTURE_ANCHORED",
            "system":system,"repeat":int(rep),"difficulty":difficulty,
            **selective_metrics(g)
        })
    pd.DataFrame(diff_rows).to_csv(
        INTER/"06_methodsafe_secondary_performance_by_difficulty.csv",index=False,encoding="utf-8-sig"
    )

    # 4) Parent-majority label baseline.
    baselines=pd.DataFrame([
        parent_majority_label_baseline(membership,"PRIMARY_SOURCE_CURATED"),
        parent_majority_label_baseline(membership,"SECONDARY_STRUCTURE_ANCHORED"),
    ])
    baselines.to_csv(INTER/"06_parent_majority_label_baseline.csv",index=False,encoding="utf-8-sig")

    # 5) Sensitivity excluding R-warfarin sodium (stereo scope unspecified).
    sens_rows=[]
    for set_name in sets:
        sg=add_set_label(pred,set_name)
        sg=sg[~sg["stereo_scope_sensitivity_flag"].map(as_bool)].copy()
        for (system,rep),g in sg.groupby(["system","repeat"],sort=False):
            sens_rows.append({
                "benchmark_set":set_name,
                "sensitivity_analysis":"EXCLUDE_STEREO_SCOPE_AMBIGUOUS_R_WARFARIN_SODIUM",
                "system":system,"repeat":int(rep),**selective_metrics(g)
            })
    pd.DataFrame(sens_rows).to_csv(
        INTER/"06_methodsafe_sensitivity_excluding_stereo_case.csv",index=False,encoding="utf-8-sig"
    )

    # 6) Transparent interpretation table.
    interpretation=pd.DataFrame([
        {
            "benchmark_set":"PRIMARY_SOURCE_CURATED",
            "role":"PRIMARY_INFERENTIAL",
            "n_total_source_cases":readiness.get("n_primary_total",60),
            "n_operational_cases":readiness.get("n_primary_operational",59),
            "rdkit_qc_used_for_inclusion":False,
            "recommended_claim":"Primary comparison of LLM+DB, frozen DB+RDKit, and frozen Hybrid.",
        },
        {
            "benchmark_set":"SECONDARY_STRUCTURE_ANCHORED",
            "role":"SECONDARY_COVERAGE_STRESS",
            "n_total_source_cases":readiness.get("n_secondary_total",134),
            "n_operational_cases":readiness.get("n_secondary_total",134),
            "rdkit_qc_used_for_inclusion":True,
            "recommended_claim":"Structure-anchored stress/coverage analysis; do not call generic RDKit performance estimate.",
        },
        {
            "benchmark_set":"RDKIT_V43_SENSITIVITY",
            "role":"POST_PREFLIGHT_METHOD_REFINEMENT",
            "n_total_source_cases":"",
            "n_operational_cases":"",
            "rdkit_qc_used_for_inclusion":"",
            "recommended_claim":"Sensitivity analysis only; V4.3 was developed after preflight and must not replace the frozen primary comparator in main inference.",
        },
    ])
    interpretation.to_csv(INTER/"06_methodsafe_interpretation_guardrails.csv",index=False,encoding="utf-8-sig")

    print("="*96)
    print("METHOD-SAFE Study 2 results prepared")
    print("="*96)
    print("\nParent-majority label baselines:")
    print(baselines.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\nPrimary-system mean performance:")
    show=agg[
        agg["system"].isin(PRIMARY_SYSTEMS)
    ][[
        "benchmark_set","system","n_total","coverage_mean","accuracy_mean",
        "balanced_accuracy_mean","overall_correct_resolution_rate_mean",
        "recall_mean","specificity_mean","false_safe_rate_mean",
    ]]
    print(show.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\nV4.3 outputs are sensitivity-only and saved separately in the same result tables.")

if __name__=="__main__":
    main()