# -*- coding: utf-8 -*-
"""Score frozen retrieval predictions against retrieval GOLD.

This is the first evaluation step allowed to read retrieval_GOLD.csv. Scoring is
blocked unless every LLM-evaluated query completed the full frozen repeat count.
Category breakdown is based on the GOLD target type, not an arbitrary first
source category from a multi-source query.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
INTER = ROOT / "intermediate"
GOLD = DATA / "retrieval_GOLD.csv"
PRED = INTER / "retrieval_predictions_caselevel.csv"
LLM_CALLS = INTER / "llm_only_calls.csv"
HY_CALLS = INTER / "hybrid_review_calls.csv"
RUN_META = INTER / "retrieval_run_metadata.json"

OUT_CASES = INTER / "retrieval_predictions_scored.csv"
OUT_SUMMARY = INTER / "retrieval_metrics_summary.csv"
OUT_BREAKDOWN = INTER / "retrieval_metrics_by_gold_category.csv"
OUT_CONFLICTS = INTER / "retrieval_not_found_gate_conflicts.csv"
OUT_JSON = INTER / "retrieval_metrics.json"


def clean(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def predicted_outcome(status: str, target_id: str) -> str:
    s = clean(status).upper()
    t = clean(target_id)
    if s == "FOUND" and t:
        return t
    if s == "NOT_FOUND":
        return "NOT_FOUND"
    return "REVIEW"


def gold_category(target_id: str) -> str:
    t = clean(target_id)
    if t.startswith("PS::"):
        return "PARENT_SALT"
    if t.startswith("CG::"):
        return "CHEMICAL_GROUP"
    if t.startswith("MX::"):
        return "MIXTURE"
    if t == "NOT_FOUND":
        return "NOT_FOUND"
    return "UNKNOWN"


def call_stats(path: Path) -> dict:
    if not path.exists():
        return {"ok_calls": 0, "input_tokens": 0, "output_tokens": 0, "elapsed_sec": 0.0}
    df = pd.read_csv(path, dtype=str).fillna("")
    ok = df[df["call_status"].astype(str).eq("OK")].copy()
    return {
        "ok_calls": int(len(ok)),
        "input_tokens": int(pd.to_numeric(ok.get("input_tokens", 0), errors="coerce").fillna(0).sum()),
        "output_tokens": int(pd.to_numeric(ok.get("output_tokens", 0), errors="coerce").fillna(0).sum()),
        "elapsed_sec": float(pd.to_numeric(ok.get("elapsed_sec", 0), errors="coerce").fillna(0).sum()),
    }


def metrics(df: pd.DataFrame) -> dict:
    n = len(df)
    resolved = df["predicted_outcome"].ne("REVIEW")
    correct = df["correct"]
    found_gold = df["gold_status"].eq("FOUND")
    nf_gold = df["gold_status"].eq("NOT_FOUND")
    tp_nf = int((nf_gold & df["predicted_outcome"].eq("NOT_FOUND")).sum())
    fp_nf = int((~nf_gold & df["predicted_outcome"].eq("NOT_FOUND")).sum())
    fn_nf = int((nf_gold & ~df["predicted_outcome"].eq("NOT_FOUND")).sum())
    nf_precision = tp_nf / (tp_nf + fp_nf) if (tp_nf + fp_nf) else float("nan")
    nf_recall = tp_nf / (tp_nf + fn_nf) if (tp_nf + fn_nf) else float("nan")
    return {
        "n": int(n),
        "exact_accuracy": float(correct.mean()) if n else float("nan"),
        "resolved_rate": float(resolved.mean()) if n else float("nan"),
        "review_rate": float((~resolved).mean()) if n else float("nan"),
        "accuracy_on_resolved": float(correct[resolved].mean()) if resolved.any() else float("nan"),
        "found_target_accuracy": float(correct[found_gold].mean()) if found_gold.any() else float("nan"),
        "not_found_accuracy": float(correct[nf_gold].mean()) if nf_gold.any() else float("nan"),
        "not_found_precision": float(nf_precision),
        "not_found_recall": float(nf_recall),
        "n_wrong_target": int((found_gold & df["predicted_outcome"].ne("REVIEW") & df["predicted_outcome"].ne("NOT_FOUND") & ~correct).sum()),
        "n_false_not_found": int((found_gold & df["predicted_outcome"].eq("NOT_FOUND")).sum()),
        "n_false_found": int((nf_gold & df["predicted_outcome"].str.startswith(("PS::", "CG::", "MX::"))).sum()),
    }


def verify_complete_run(run: dict) -> None:
    repeat = run.get("repeat_completeness", {})
    for condition in ["LLM_ONLY", "HYBRID_REVIEW"]:
        r = repeat.get(condition, {})
        if not r:
            raise RuntimeError(f"Missing repeat completeness report for {condition}")
        if int(r.get("complete_queries", -1)) != int(r.get("expected_queries", -2)):
            raise RuntimeError(f"Cannot score incomplete LLM repeats: {condition} -> {r}")
        if r.get("incomplete"):
            raise RuntimeError(f"Cannot score incomplete LLM repeats: {condition} -> {r['incomplete']}")


def main() -> None:
    for p in [GOLD, PRED, RUN_META, LLM_CALLS, HY_CALLS]:
        if not p.exists():
            raise FileNotFoundError(p)

    run = json.loads(RUN_META.read_text(encoding="utf-8"))
    verify_complete_run(run)

    gold = pd.read_csv(GOLD, dtype=str).fillna("")
    pred = pd.read_csv(PRED, dtype=str).fillna("")
    if set(pred["query_id"]) != set(gold["query_id"]):
        raise RuntimeError("Prediction/GOLD query sets differ")

    merged = pred.merge(gold, on="query_id", how="left", validate="many_to_one")
    merged["predicted_outcome"] = merged.apply(lambda r: predicted_outcome(r["status"], r["target_id"]), axis=1)
    merged["correct"] = merged["predicted_outcome"].eq(merged["gold_target_id"])
    merged["gold_category"] = merged["gold_target_id"].apply(gold_category)
    merged.to_csv(OUT_CASES, index=False, encoding="utf-8-sig")

    summaries = []
    for system, g in merged.groupby("system", sort=False):
        m = metrics(g); m["system"] = system; summaries.append(m)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    breakdown = []
    for (system, cat), g in merged.groupby(["system", "gold_category"], sort=True):
        m = metrics(g); m["system"] = system; m["gold_category"] = cat; breakdown.append(m)
    breakdown_df = pd.DataFrame(breakdown)
    breakdown_df.to_csv(OUT_BREAKDOWN, index=False, encoding="utf-8-sig")

    conflicts = merged[
        merged["system"].eq("HYBRID")
        & merged["gold_status"].eq("NOT_FOUND")
        & merged["decision_source"].eq("DETERMINISTIC_GATE")
        & merged["status"].eq("FOUND")
    ].copy()
    conflicts.to_csv(OUT_CONFLICTS, index=False, encoding="utf-8-sig")

    llm_stats = call_stats(LLM_CALLS)
    hy_stats = call_stats(HY_CALLS)
    result = {
        "protocol": "STUDY2_LLM_VS_HYBRID_CAS_RETRIEVAL_RESULTS_V2",
        "n_queries": int(gold["query_id"].nunique()),
        "systems": {row["system"]: {k: v for k, v in row.items() if k != "system"} for row in summaries},
        "efficiency": {
            "LLM_ONLY": llm_stats,
            "HYBRID_LLM_ESCALATIONS": hy_stats,
            "hybrid_direct_queries": int(run.get("hybrid_direct_queries", 0)),
            "hybrid_review_queries": int(run.get("hybrid_review_queries", 0)),
            "llm_call_reduction_fraction": 1.0 - (hy_stats["ok_calls"] / llm_stats["ok_calls"]) if llm_stats["ok_calls"] else float("nan"),
            "input_token_reduction_fraction": 1.0 - (hy_stats["input_tokens"] / llm_stats["input_tokens"]) if llm_stats["input_tokens"] else float("nan"),
        },
        "qc": {
            "repeat_completeness": run.get("repeat_completeness", {}),
            "not_found_hybrid_deterministic_conflicts": int(len(conflicts)),
        },
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")

    print(summary_df.to_string(index=False))
    print("\nBy GOLD category:")
    print(breakdown_df.to_string(index=False))
    print("\nEfficiency:")
    print(json.dumps(result["efficiency"], ensure_ascii=False, indent=2, allow_nan=True))
    print(f"\n[QC] NOT_FOUND vs Hybrid deterministic conflicts: {len(conflicts)}")
    for p in [OUT_CASES, OUT_SUMMARY, OUT_BREAKDOWN, OUT_CONFLICTS, OUT_JSON]:
        print(f"[OUT] {p}")


if __name__ == "__main__":
    main()
