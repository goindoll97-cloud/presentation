# -*- coding: utf-8 -*-
"""Run LLM-only and Hybrid CAS-only regulatory retrieval.

No GOLD file is read by this script. Runtime settings are checked against the
frozen protocol before any API call. LLM-only and Hybrid-review calls are
interleaved by query/repeat to reduce temporal confounding. Every LLM-evaluated
query must have exactly N_REPEATS successful calls before predictions are made.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import eval_common as ec
import retrieval_runtime as rt

ROOT = Path(__file__).resolve().parent
INTER = ROOT / "intermediate"
INTER.mkdir(parents=True, exist_ok=True)
PROTOCOL_FREEZE = ROOT / "EVAL_PROTOCOL_FREEZE.json"
GATE_FILE = INTER / "hybrid_gate_preflight.csv"
LLM_CALLS = INTER / "llm_only_calls.csv"
HYBRID_CALLS = INTER / "hybrid_review_calls.csv"
PREDICTIONS = INTER / "retrieval_predictions_caselevel.csv"
RUN_METADATA = INTER / "retrieval_run_metadata.json"
EXECUTE = os.getenv("RETRIEVAL_EXECUTE_LLM", "0").strip().lower() in {"1", "true", "yes", "on"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_runtime_contract() -> dict:
    return {
        "model": rt.MODEL,
        "prompt_version": rt.PROMPT_VERSION,
        "system_prompt_sha256": hashlib.sha256(rt.SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "n_repeats": int(rt.N_REPEATS),
        "sampling_control": rt.SAMPLING_MODE,
        "max_tokens": int(rt.MAX_TOKENS),
        "effort": rt.EFFORT,
    }


def verify_protocol() -> tuple[dict, dict]:
    if not PROTOCOL_FREEZE.exists():
        raise RuntimeError("Run 03_freeze_eval_protocol.py first")
    f = json.loads(PROTOCOL_FREEZE.read_text(encoding="utf-8"))
    files = f.get("sha256", {})
    engine_now = ec.parent_engine_runtime_info()
    engine_frozen = f.get("hybrid_parent_engine", {})
    runtime_now = current_runtime_contract()
    runtime_frozen = f.get("runtime_contract", {})

    checks = {
        "eval_common.py": files.get("eval_common.py") == sha256(ROOT / "eval_common.py"),
        "retrieval_runtime.py": files.get("retrieval_runtime.py") == sha256(ROOT / "retrieval_runtime.py"),
        "04_run_retrieval_llm_hybrid.py": files.get("04_run_retrieval_llm_hybrid.py") == sha256(Path(__file__).resolve()),
        "hybrid_gate_preflight.csv": files.get("intermediate/hybrid_gate_preflight.csv") == sha256(GATE_FILE),
        "parent_engine_sha256": files.get("../cheminformatics_identity_V5_FAIR.py") == sha256(ec.PARENT_ENGINE),
        "runtime_contract": runtime_now == runtime_frozen,
        "rdkit_version": ec.clean(engine_now.get("rdkit_version")) == ec.clean(engine_frozen.get("rdkit_version")),
        "engine_version": ec.clean(engine_now.get("engine_version")) == ec.clean(engine_frozen.get("engine_version")),
        "engine_self_tests": int(engine_now.get("self_test_passed", 0)) == int(engine_now.get("self_test_n", -1)),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "Evaluation protocol/runtime changed after freeze: "
            + json.dumps({"checks": checks, "runtime_frozen": runtime_frozen, "runtime_now": runtime_now}, ensure_ascii=False)
        )
    return checks, f


def run_interleaved(qin: pd.DataFrame, review_ids: set[str], catalog: pd.DataFrame, smap: dict):
    allowed = set(catalog["target_id"].astype(str))
    llm_out, hy_out = [], []
    for repeat in range(1, rt.N_REPEATS + 1):
        for _, row in qin.iterrows():
            qid = ec.clean(row.get("query_id"))
            prompt = ec.build_prompt(row, catalog, smap)
            ans = rt.run_one(qid, prompt, "LLM_ONLY", repeat, allowed)
            llm_out.append(ans)
            print(
                f"[LLM_ONLY] repeat={repeat} query={qid} status={ans.get('status')} "
                f"target={ans.get('target_id')} call={ans.get('call_status')}", flush=True
            )
            if qid in review_ids:
                hans = rt.run_one(qid, prompt, "HYBRID_REVIEW", repeat, allowed)
                hy_out.append(hans)
                print(
                    f"[HYBRID_REVIEW] repeat={repeat} query={qid} status={hans.get('status')} "
                    f"target={hans.get('target_id')} call={hans.get('call_status')}", flush=True
                )
    return pd.DataFrame(llm_out), pd.DataFrame(hy_out)


def completeness_report(calls: pd.DataFrame, expected_ids: set[str], condition: str) -> tuple[bool, dict]:
    if not expected_ids:
        return True, {"condition": condition, "expected_queries": 0, "complete_queries": 0, "incomplete": []}
    if calls.empty:
        return False, {"condition": condition, "expected_queries": len(expected_ids), "complete_queries": 0,
                       "incomplete": sorted(expected_ids)}
    ok = calls[calls["call_status"].astype(str).eq("OK")].copy()
    counts = ok.groupby("query_id").size().to_dict()
    incomplete = [qid for qid in sorted(expected_ids) if int(counts.get(qid, 0)) != int(rt.N_REPEATS)]
    report = {
        "condition": condition,
        "expected_queries": len(expected_ids),
        "complete_queries": len(expected_ids) - len(incomplete),
        "required_ok_per_query": int(rt.N_REPEATS),
        "min_ok_per_query": min([int(counts.get(q, 0)) for q in expected_ids]) if expected_ids else 0,
        "max_ok_per_query": max([int(counts.get(q, 0)) for q in expected_ids]) if expected_ids else 0,
        "incomplete": incomplete,
    }
    return not incomplete, report


def consensus_map(calls: pd.DataFrame) -> dict[str, tuple[str, str]]:
    out = {}
    ok = calls[calls["call_status"].astype(str).eq("OK")].copy()
    for qid, g in ok.groupby("query_id", sort=False):
        if len(g) != rt.N_REPEATS:
            raise RuntimeError(f"Consensus requested with incomplete repeats: {qid} has {len(g)}/{rt.N_REPEATS}")
        out[str(qid)] = ec.consensus_outcome(g)
    return out


def main() -> None:
    protocol_checks, frozen_protocol = verify_protocol()
    qin, catalog, smap, frozen_checks = ec.load_inputs()
    gate = pd.read_csv(GATE_FILE, dtype=str).fillna("")
    if set(gate["query_id"]) != set(qin["query_id"]):
        raise RuntimeError("Hybrid gate query set differs from frozen retrieval INPUT")
    gate_map = gate.set_index("query_id").to_dict("index")
    review_ids = set(gate.loc[gate["gate_status"].eq("REVIEW"), "query_id"])
    all_ids = set(qin["query_id"])

    meta = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "execute_llm": EXECUTE,
        "runtime_contract": current_runtime_contract(),
        "n_queries": int(len(qin)),
        "hybrid_direct_queries": int((gate["gate_status"] == "FOUND").sum()),
        "hybrid_review_queries": int((gate["gate_status"] == "REVIEW").sum()),
        "expected_llm_only_calls": int(len(qin) * rt.N_REPEATS),
        "expected_hybrid_llm_calls": int(len(review_ids) * rt.N_REPEATS),
        "execution_order": "INTERLEAVED_BY_REPEAT_AND_QUERY",
        "protocol_checks": protocol_checks,
        "frozen_input_checks": frozen_checks,
        "frozen_protocol_sha256": sha256(PROTOCOL_FREEZE),
    }
    RUN_METADATA.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))

    if not EXECUTE:
        print("[DRY RUN] No LLM calls were made. Set RETRIEVAL_EXECUTE_LLM=1 after inspecting EVAL_PROTOCOL_FREEZE.json.")
        return

    llm_calls, hy_calls = run_interleaved(qin, review_ids, catalog, smap)
    llm_calls.to_csv(LLM_CALLS, index=False, encoding="utf-8-sig")
    hy_calls.to_csv(HYBRID_CALLS, index=False, encoding="utf-8-sig")

    llm_complete, llm_report = completeness_report(llm_calls, all_ids, "LLM_ONLY")
    hy_complete, hy_report = completeness_report(hy_calls, review_ids, "HYBRID_REVIEW")
    meta["repeat_completeness"] = {"LLM_ONLY": llm_report, "HYBRID_REVIEW": hy_report}
    meta["actual_llm_only_ok_calls"] = int(llm_calls["call_status"].astype(str).eq("OK").sum())
    meta["actual_hybrid_ok_calls"] = int(hy_calls["call_status"].astype(str).eq("OK").sum()) if len(hy_calls) else 0
    RUN_METADATA.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if not llm_complete or not hy_complete:
        print("[INCOMPLETE] Successful repeats are not N/N for every LLM-evaluated query.")
        print(json.dumps(meta["repeat_completeness"], ensure_ascii=False, indent=2))
        print("Re-run the same command; successful calls are cached and only failed calls need fresh API success.")
        raise RuntimeError("INCOMPLETE_LLM_REPEATS: prediction generation blocked")

    llm_cons = consensus_map(llm_calls)
    hy_cons = consensus_map(hy_calls)
    rows = []
    for _, r in qin.iterrows():
        qid = ec.clean(r.get("query_id"))
        llm_status, llm_target = llm_cons[qid]
        rows.append({
            "query_id": qid, "system": "LLM_ONLY", "status": llm_status, "target_id": llm_target,
            "decision_source": "LLM_FULL_CATALOG", "gate_reason": "",
        })
        g = gate_map[qid]
        if ec.clean(g.get("gate_status")).upper() == "FOUND":
            h_status, h_target, source = "FOUND", ec.clean(g.get("gate_target_id")), "DETERMINISTIC_GATE"
        else:
            h_status, h_target = hy_cons[qid]
            source = "LLM_AFTER_GATE_REVIEW"
        rows.append({
            "query_id": qid, "system": "HYBRID", "status": h_status, "target_id": h_target,
            "decision_source": source, "gate_reason": ec.clean(g.get("gate_reason")),
        })

    pred = pd.DataFrame(rows)
    if len(pred) != 2 * len(qin):
        raise RuntimeError("Unexpected case-level prediction row count")
    pred.to_csv(PREDICTIONS, index=False, encoding="utf-8-sig")
    meta["prediction_file_sha256"] = sha256(PREDICTIONS)
    RUN_METADATA.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OUT] {LLM_CALLS}")
    print(f"[OUT] {HYBRID_CALLS}")
    print(f"[OUT] {PREDICTIONS}")
    print(f"[OUT] {RUN_METADATA}")


if __name__ == "__main__":
    main()
