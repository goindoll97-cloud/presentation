# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
INTER = ROOT / "intermediate_gm_v1"
PRED = INTER / "14_gm_system_predictions_caselevel.csv"
EFF = INTER / "14_gm_efficiency.csv"


def metrics(g: pd.DataFrame) -> dict:
    truth = g.reference_membership.astype(str)
    pred = g.decision.astype(str)
    resolved = pred.isin(["MATCH", "NO_MATCH"])
    tp = int(((pred == "MATCH") & (truth == "MATCH")).sum())
    fp = int(((pred == "MATCH") & (truth == "NO_MATCH")).sum())
    fn = int(((pred != "MATCH") & (truth == "MATCH")).sum())
    tn = int(((pred == "NO_MATCH") & (truth == "NO_MATCH")).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(g),
        "accuracy_review_in_denominator": float((pred == truth).mean()),
        "coverage": float(resolved.mean()),
        "review_rate": float((pred == "REVIEW").mean()),
        "match_precision": precision,
        "match_recall": recall,
        "match_f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn_or_review_on_match": fn,
    }


def main():
    if not PRED.exists():
        raise FileNotFoundError("Paid run output missing. Run Step 14 with GROUP_MIX_V1_EXECUTE_CLAUDE=1 first.")
    df = pd.read_csv(PRED, dtype=str).fillna("")

    rows = []
    for system, g in df.groupby("system", sort=False):
        rows.append({"system": system, "task_type": "ALL", **metrics(g)})
        for task, gg in g.groupby("task_type", sort=False):
            rows.append({"system": system, "task_type": task, **metrics(gg)})
    perf = pd.DataFrame(rows)
    perf.to_csv(INTER / "15_gm_performance.csv", index=False, encoding="utf-8-sig")

    subgroup = []
    for (system, rule), g in df.groupby(["system", "rule_id"], sort=False):
        subgroup.append({"system": system, "rule_id": rule, **metrics(g)})
    pd.DataFrame(subgroup).to_csv(INTER / "15_gm_performance_by_rule.csv", index=False, encoding="utf-8-sig")

    boundary = df[df.task_type.eq("MIXTURE_THRESHOLD")].copy()
    b = []
    for (system, bc), g in boundary.groupby(["system", "boundary_class"], sort=False):
        b.append({"system": system, "boundary_class": bc, **metrics(g)})
    pd.DataFrame(b).to_csv(INTER / "15_gm_mixture_boundary_performance.csv", index=False, encoding="utf-8-sig")

    fail = df[(df.decision != df.reference_membership) | (df.decision == "REVIEW")].copy()
    fail.to_csv(INTER / "15_gm_failure_or_review_cases.csv", index=False, encoding="utf-8-sig")

    summary = {
        "performance_rows": len(perf),
        "n_prediction_rows": len(df),
        "n_failure_or_review_rows": len(fail),
        "efficiency_file_present": EFF.exists(),
    }
    (INTER / "15_gm_results_metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(perf.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
