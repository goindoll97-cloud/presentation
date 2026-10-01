# -*- coding: utf-8 -*-
"""Aggregate performance and efficiency results for Study 2 V6."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INTER = ROOT / "intermediate_v6"
PRED = INTER / "09_v6_system_predictions_caselevel.csv"
EFF = INTER / "09_v6_efficiency.csv"


def as_bool(x) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z*z/n
    ctr = (p + z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return max(0.0, ctr-half), min(1.0, ctr+half)


def performance(g: pd.DataFrame) -> dict:
    n = len(g)
    dec = g[g["decided"].map(as_bool)].copy()
    n_dec = len(dec)
    cov = n_dec / n if n else np.nan
    overall_n = int(g["overall_correct_resolution"].map(as_bool).sum())
    overall = overall_n / n if n else np.nan
    if n_dec:
        truth = dec["truth_bool"].map(as_bool)
        pred = dec["pred_bool"].map(as_bool)
        tp = int((truth & pred).sum()); tn = int((~truth & ~pred).sum())
        fp = int((~truth & pred).sum()); fn = int((truth & ~pred).sum())
        acc = (tp + tn) / n_dec
        rec = tp/(tp+fn) if tp+fn else np.nan
        spec = tn/(tn+fp) if tn+fp else np.nan
        bal = np.nanmean([rec, spec]) if np.isfinite(rec) or np.isfinite(spec) else np.nan
    else:
        tp = tn = fp = fn = 0
        acc = rec = spec = bal = np.nan
    acl, ach = wilson(tp+tn, n_dec) if n_dec else (np.nan, np.nan)
    ocl, och = wilson(overall_n, n)
    ccl, cch = wilson(n_dec, n)
    return {
        "n_total": n, "n_decided": n_dec, "coverage": cov, "coverage_ci95_low": ccl, "coverage_ci95_high": cch,
        "accuracy": acc, "accuracy_ci95_low": acl, "accuracy_ci95_high": ach,
        "recall": rec, "specificity": spec, "balanced_accuracy": bal,
        "false_safe_rate": fn/(tp+fn) if tp+fn else np.nan,
        "false_positive_rate": fp/(tn+fp) if tn+fp else np.nan,
        "overall_correct_resolution_rate": overall,
        "overall_correct_resolution_ci95_low": ocl, "overall_correct_resolution_ci95_high": och,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


def exact_mcnemar(df: pd.DataFrame, a: str, b: str) -> dict:
    x = df[df.system.eq(a)][["case_id", "overall_correct_resolution"]]
    y = df[df.system.eq(b)][["case_id", "overall_correct_resolution"]]
    m = x.merge(y, on="case_id", suffixes=("_a", "_b"))
    ca = m["overall_correct_resolution_a"].map(as_bool)
    cb = m["overall_correct_resolution_b"].map(as_bool)
    ao = int((ca & ~cb).sum()); bo = int((~ca & cb).sum()); n = ao + bo
    p = 1.0 if n == 0 else min(1.0, 2.0 * sum(math.comb(n, i) for i in range(min(ao, bo)+1)) / (2**n))
    return {"system_a": a, "system_b": b, "n_common": len(m), "a_only_correct": ao, "b_only_correct": bo, "p_exact": p}


def grouped(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for (grp, sys), g in df.groupby([col, "system"], dropna=False):
        rows.append({col: grp, "system": sys, **performance(g)})
    return pd.DataFrame(rows)


def main() -> None:
    if not PRED.exists() or not EFF.exists():
        raise FileNotFoundError("Run 09_run_identity_e2e_LLM_HYBRID_V6.py with IDENTITY_V6_EXECUTE_CLAUDE=1 first")
    pred = pd.read_csv(PRED)
    eff = pd.read_csv(EFF)

    perf_rows = []
    for sys, g in pred.groupby("system"):
        perf_rows.append({"system": sys, **performance(g)})
    perf = pd.DataFrame(perf_rows)
    perf.to_csv(INTER / "10_v6_performance.csv", index=False, encoding="utf-8-sig")
    grouped(pred, "difficulty").to_csv(INTER / "10_v6_performance_by_difficulty.csv", index=False, encoding="utf-8-sig")
    grouped(pred, "reference_parent_name").to_csv(INTER / "10_v6_performance_by_parent.csv", index=False, encoding="utf-8-sig")
    grouped(pred, "challenge_class").to_csv(INTER / "10_v6_performance_by_challenge_class.csv", index=False, encoding="utf-8-sig")

    mc = pd.DataFrame([exact_mcnemar(pred, "LLM_PUBCHEM", "HYBRID_PUBCHEM")])
    mc.to_csv(INTER / "10_v6_mcnemar.csv", index=False, encoding="utf-8-sig")

    e = eff.set_index("system")
    base = e.loc["LLM_PUBCHEM"]
    hy = e.loc["HYBRID_PUBCHEM"]
    summary = pd.DataFrame([
        {"metric": "LLM case rate", "LLM_PUBCHEM": base.llm_case_rate, "HYBRID_PUBCHEM": hy.llm_case_rate, "relative_reduction": 1 - hy.llm_case_rate/base.llm_case_rate if base.llm_case_rate else np.nan},
        {"metric": "LLM calls", "LLM_PUBCHEM": base.llm_calls, "HYBRID_PUBCHEM": hy.llm_calls, "relative_reduction": 1 - hy.llm_calls/base.llm_calls if base.llm_calls else np.nan},
        {"metric": "Input tokens", "LLM_PUBCHEM": base.input_tokens, "HYBRID_PUBCHEM": hy.input_tokens, "relative_reduction": 1 - hy.input_tokens/base.input_tokens if base.input_tokens else np.nan},
        {"metric": "Output tokens", "LLM_PUBCHEM": base.output_tokens, "HYBRID_PUBCHEM": hy.output_tokens, "relative_reduction": 1 - hy.output_tokens/base.output_tokens if base.output_tokens else np.nan},
        {"metric": "API cost (USD)", "LLM_PUBCHEM": base.api_cost_usd, "HYBRID_PUBCHEM": hy.api_cost_usd, "relative_reduction": 1 - hy.api_cost_usd/base.api_cost_usd if base.api_cost_usd else np.nan},
        {"metric": "LLM API time (s)", "LLM_PUBCHEM": base.llm_api_elapsed_sec, "HYBRID_PUBCHEM": hy.llm_api_elapsed_sec, "relative_reduction": 1 - hy.llm_api_elapsed_sec/base.llm_api_elapsed_sec if base.llm_api_elapsed_sec else np.nan},
        {"metric": "Operational elapsed proxy (s)", "LLM_PUBCHEM": base.operational_elapsed_proxy_sec, "HYBRID_PUBCHEM": hy.operational_elapsed_proxy_sec, "relative_reduction": 1 - hy.operational_elapsed_proxy_sec/base.operational_elapsed_proxy_sec if base.operational_elapsed_proxy_sec else np.nan},
    ])
    summary.to_csv(INTER / "10_v6_efficiency_summary.csv", index=False, encoding="utf-8-sig")

    failures = pred[~pred["overall_correct_resolution"].map(as_bool)].copy()
    failures.to_csv(INTER / "10_v6_failure_or_review_cases.csv", index=False, encoding="utf-8-sig")

    print("\n=== PERFORMANCE ===")
    print(perf.to_string(index=False))
    print("\n=== EFFICIENCY ===")
    print(summary.to_string(index=False))
    print("\n=== MCNEMAR ===")
    print(mc.to_string(index=False))
    print("\n[SAVED] V6 result tables in intermediate_v6/")


if __name__ == "__main__":
    main()
