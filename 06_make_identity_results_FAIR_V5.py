# -*- coding: utf-8 -*-
"""Study 2 / Step 06 FAIR V5 result tables.

Primary scoring is case-level. Repeated LLM calls were already collapsed to a
consensus in Step 05, so deterministic systems are not pseudo-replicated.
The paired primary endpoint is overall correct resolution: REVIEW remains in the
denominator as unresolved and is therefore not silently dropped from McNemar.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INTER = ROOT / "intermediate"
PRED = INTER / "05_v5_identity_system_predictions_caselevel.csv"
META = INTER / "05_v5_run_metadata.json"
OUT_PERF = INTER / "06_fair_v5_performance_mean.csv"
OUT_PARENT = INTER / "06_fair_v5_performance_by_parent.csv"
OUT_DIFF = INTER / "06_fair_v5_performance_by_difficulty.csv"
OUT_CHALLENGE = INTER / "06_fair_v5_performance_by_challenge_class.csv"
OUT_TESTS = INTER / "06_fair_v5_paired_mcnemar.csv"
OUT_NOTE = INTER / "06_fair_v5_interpretation_note.md"

SYSTEMS = ["CLAUDE_DB", "RDKIT_SALT_AWARE_V5", "HYBRID_SALT_AWARE_V5"]


def clean(x) -> str:
    if x is None: return ""
    try:
        if pd.isna(x): return ""
    except Exception:
        pass
    s = str(x).strip()
    return "" if s.lower() in {"nan", "none", "null", "<na>"} else s


def as_bool(x) -> bool:
    if isinstance(x, (bool, np.bool_)): return bool(x)
    return clean(x).lower() in {"1", "true", "yes", "y"}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0: return np.nan, np.nan
    p = k / n; den = 1 + z*z/n; center = (p + z*z/(2*n)) / den
    half = z * math.sqrt((p*(1-p)/n) + z*z/(4*n*n)) / den
    return max(0.0, center-half), min(1.0, center+half)


def metrics(g: pd.DataFrame) -> dict:
    n = len(g); decided = g[g["decided"].map(as_bool)].copy(); nd = len(decided)
    overall_correct = int(g["overall_correct_resolution"].map(as_bool).sum())
    overall = overall_correct / n if n else np.nan; o_lo, o_hi = wilson(overall_correct, n)
    if nd == 0:
        return {"n_total": n, "n_decided": 0, "coverage": 0.0 if n else np.nan, "accuracy": np.nan, "accuracy_ci95_low": np.nan, "accuracy_ci95_high": np.nan, "precision": np.nan, "recall": np.nan, "specificity": np.nan, "balanced_accuracy": np.nan, "false_safe_rate": np.nan, "false_positive_rate": np.nan, "overall_correct_resolution_rate": overall, "overall_correct_resolution_ci95_low": o_lo, "overall_correct_resolution_ci95_high": o_hi, "TP":0,"TN":0,"FP":0,"FN":0}
    t = decided["truth_bool"].map(as_bool); p = decided["pred_bool"].map(as_bool)
    tp = int((t&p).sum()); tn = int((~t&~p).sum()); fp = int((~t&p).sum()); fn = int((t&~p).sum()); acc_n = tp+tn
    a_lo, a_hi = wilson(acc_n, nd); rec = tp/(tp+fn) if tp+fn else np.nan; spec = tn/(tn+fp) if tn+fp else np.nan
    ba = np.nanmean([rec,spec]) if np.isfinite(rec) or np.isfinite(spec) else np.nan
    return {"n_total": n, "n_decided": nd, "coverage": nd/n if n else np.nan, "accuracy": acc_n/nd, "accuracy_ci95_low": a_lo, "accuracy_ci95_high": a_hi, "precision": tp/(tp+fp) if tp+fp else np.nan, "recall": rec, "specificity": spec, "balanced_accuracy": ba, "false_safe_rate": fn/(tp+fn) if tp+fn else np.nan, "false_positive_rate": fp/(tn+fp) if tn+fp else np.nan, "overall_correct_resolution_rate": overall, "overall_correct_resolution_ci95_low": o_lo, "overall_correct_resolution_ci95_high": o_hi, "TP":tp,"TN":tn,"FP":fp,"FN":fn}


def exact_mcnemar(df: pd.DataFrame, a: str, b: str) -> dict:
    """Exact paired McNemar on pre-specified overall-correct-resolution endpoint."""
    x = df[df["system"].eq(a)][["case_id","overall_correct_resolution"]].copy()
    y = df[df["system"].eq(b)][["case_id","overall_correct_resolution"]].copy()
    m = x.merge(y,on="case_id",suffixes=("_a","_b"))
    ca = m["overall_correct_resolution_a"].map(as_bool)
    cb = m["overall_correct_resolution_b"].map(as_bool)
    bo = int((ca & ~cb).sum()); co = int((~ca & cb).sum()); n = bo+co
    if n == 0: p=1.0
    else:
        k=min(bo,co); p=min(1.0,2*sum(math.comb(n,i) for i in range(k+1))/(2**n))
    return {"endpoint":"overall_correct_resolution","system_a":a,"system_b":b,"n_common":len(m),"a_only_correct":bo,"b_only_correct":co,"p_exact":p}


def benchmark_views(pred: pd.DataFrame, status: str):
    if status.startswith("FINAL_"):
        yield "FINAL_INDEPENDENT_HOLDOUT", pred; return
    if "in_primary_source_curated" in pred.columns:
        q = pred[pred["in_primary_source_curated"].map(as_bool)]
        if len(q): yield "DEVELOPMENT_SOURCE_CURATED", q
    if "in_secondary_structure_anchored" in pred.columns:
        q = pred[pred["in_secondary_structure_anchored"].map(as_bool)]
        if len(q): yield "DEVELOPMENT_STRUCTURE_ANCHORED", q
    if not any(c in pred.columns for c in ["in_primary_source_curated","in_secondary_structure_anchored"]): yield "DEVELOPMENT_ALL", pred


def main() -> None:
    if not PRED.exists(): raise FileNotFoundError("Run 05_compare_identity_SHARED_DB_FAIR_V5.py with paid execution enabled first")
    pred = pd.read_csv(PRED).fillna(""); meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}; status = clean(meta.get("analysis_status")) or "UNKNOWN"
    required = ["case_id","system","decided","truth_bool","pred_bool","overall_correct_resolution"]
    missing = [c for c in required if c not in pred.columns]
    if missing: raise ValueError(f"Step-05 prediction file missing required columns: {missing}")
    rows=[]; parent=[]; diff=[]; chall=[]; tests=[]
    for bench_name, q in benchmark_views(pred, status):
        for s in SYSTEMS:
            g=q[q["system"].eq(s)]
            if len(g): rows.append({"benchmark_set":bench_name,"system":s,**metrics(g)})
        if "reference_parent_name" in q.columns:
            for (parent_name,s),g in q.groupby(["reference_parent_name","system"],dropna=False): parent.append({"benchmark_set":bench_name,"reference_parent_name":parent_name,"system":s,**metrics(g)})
        if "difficulty" in q.columns:
            for (level,s),g in q.groupby(["difficulty","system"],dropna=False):
                if clean(level): diff.append({"benchmark_set":bench_name,"difficulty":level,"system":s,**metrics(g)})
        if "challenge_class" in q.columns:
            for (level,s),g in q.groupby(["challenge_class","system"],dropna=False):
                if clean(level): chall.append({"benchmark_set":bench_name,"challenge_class":level,"system":s,**metrics(g)})
        for a,b in [(SYSTEMS[0],SYSTEMS[1]),(SYSTEMS[0],SYSTEMS[2]),(SYSTEMS[1],SYSTEMS[2])]:
            t=exact_mcnemar(q,a,b); t["benchmark_set"]=bench_name; t["confirmatory_interpretation_allowed"]=status.startswith("FINAL_"); tests.append(t)
    pd.DataFrame(rows).to_csv(OUT_PERF,index=False,encoding="utf-8-sig"); pd.DataFrame(parent).to_csv(OUT_PARENT,index=False,encoding="utf-8-sig"); pd.DataFrame(diff).to_csv(OUT_DIFF,index=False,encoding="utf-8-sig"); pd.DataFrame(chall).to_csv(OUT_CHALLENGE,index=False,encoding="utf-8-sig"); pd.DataFrame(tests).to_csv(OUT_TESTS,index=False,encoding="utf-8-sig")
    lines=["# FAIR V5 interpretation guardrail\n",f"- Analysis status: **{status}**\n"]
    if status.startswith("FINAL_"):
        lines += ["- The independent-holdout overlap audit and pre-evaluation protocol/holdout-freeze checks passed in Step 05.\n","- Case-level McNemar tests use the pre-specified overall-correct-resolution endpoint and may be interpreted as confirmatory pairwise comparisons.\n"]
    else:
        lines += ["- These results use data already available during method refinement and are therefore **development/exploratory**, not confirmatory.\n","- McNemar p-values are retained for diagnostics only and should not be used as final inferential evidence.\n","- Freeze the V5 protocol with Step 04D, freeze a new candidate-CAS-disjoint holdout with Step 04E, then evaluate it for final claims.\n"]
    lines += ["- All three systems receive the same PubChem-resolved candidate structure, reference-parent structure, and stereochemistry policy.\n","- The hybrid uses the frozen salt-aware RDKit decision first and calls no extra LLM arm; it reuses the same case-level LLM consensus only when RDKit returns REVIEW.\n","- Selective accuracy excludes REVIEW; overall correct resolution keeps REVIEW in the denominator as unresolved.\n"]
    OUT_NOTE.write_text("".join(lines),encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False)); print(f"\nAnalysis status: {status}"); print(f"Saved: {OUT_PERF.name}, {OUT_PARENT.name}, {OUT_DIFF.name}, {OUT_TESTS.name}")


if __name__=="__main__":
    main()
