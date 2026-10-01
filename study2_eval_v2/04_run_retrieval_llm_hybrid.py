# -*- coding: utf-8 -*-
"""Run LLM-only and Hybrid CAS-only regulatory retrieval.

No GOLD file is read by this script.
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

EXECUTE = os.getenv("RETRIEVAL_EXECUTE_LLM", "0").strip().lower() in {
    "1", "true", "yes", "on"
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_protocol() -> dict:
    if not PROTOCOL_FREEZE.exists():
        raise RuntimeError("Run 03_freeze_eval_protocol.py first")
    f = json.loads(PROTOCOL_FREEZE.read_text(encoding="utf-8"))
    files = f.get("sha256", {})
    checks = {
        "eval_common.py": files.get("eval_common.py") == sha256(ROOT / "eval_common.py"),
        "retrieval_runtime.py": files.get("retrieval_runtime.py") == sha256(ROOT / "retrieval_runtime.py"),
        "04_run_retrieval_llm_hybrid.py": (
            files.get("04_run_retrieval_llm_hybrid.py") == sha256(Path(__file__).resolve())
        ),
        "hybrid_gate_preflight.csv": (
            files.get("intermediate/hybrid_gate_preflight.csv") == sha256(GATE_FILE)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Evaluation protocol changed after freeze: {checks}")
    return checks


def call_rows(rows: pd.DataFrame, catalog: pd.DataFrame, smap: dict, condition: str) -> pd.DataFrame:
    allowed = set(catalog["target_id"].astype(str))
    out = []
    for repeat in range(1, rt.N_REPEATS + 1):
        for _, row in rows.iterrows():
            qid = ec.clean(row.get("query_id"))
            prompt = ec.build_prompt(row, catalog, smap)
            ans = rt.run_one(qid, prompt, condition, repeat, allowed)
            out.append(ans)
            print(
                f"[{condition}] repeat={repeat} query={qid} "
                f"status={ans.get('status')} target={ans.get('target_id')} "
                f"call={ans.get('call_status')}",
                flush=True,
            )
    return pd.DataFrame(out)


def consensus_map(calls: pd.DataFrame) -> dict[str, tuple[str, str]]:
    out = {}
    if calls.empty:
        return out
    ok = calls[calls["call_status"].astype(str).eq("OK")].copy()
    for qid, g in ok.groupby("query_id", sort=False):
        out[str(qid)] = ec.consensus_outcome(g)
    return out


def main() -> None:
    protocol_checks = verify_protocol()
    qin, catalog, smap, frozen_checks = ec.load_inputs()
    gate = pd.read_csv(GATE_FILE, dtype=str).fillna("")
    if set(gate["query_id"]) != set(qin["query_id"]):
        raise RuntimeError("Hybrid gate query set differs from frozen retrieval INPUT")
    gate_map = gate.set_index("query_id").to_dict("index")

    review_ids = set(gate.loc[gate["gate_status"].eq("REVIEW"), "query_id"])
    review_rows = qin[qin["query_id"].isin(review_ids)].copy()

    meta = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "execute_llm": EXECUTE,
        "model": rt.MODEL,
        "prompt_version": rt.PROMPT_VERSION,
        "n_repeats": int(rt.N_REPEATS),
        "temperature": float(rt.TEMPERATURE),
        "n_queries": int(len(qin)),
        "hybrid_direct_queries": int((gate["gate_status"] == "FOUND").sum()),
        "hybrid_review_queries": int((gate["gate_status"] == "REVIEW").sum()),
        "expected_llm_only_calls": int(len(qin) * rt.N_REPEATS),
        "expected_hybrid_llm_calls": int(len(review_rows) * rt.N_REPEATS),
        "protocol_checks": protocol_checks,
        "frozen_input_checks": frozen_checks,
    }
    RUN_METADATA.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))

    if not EXECUTE:
        print(
            "[DRY RUN] No LLM calls were made. "
            "Set RETRIEVAL_EXECUTE_LLM=1 after inspecting EVAL_PROTOCOL_FREEZE.json."
        )
        return

    llm_calls = call_rows(qin, catalog, smap, "LLM_ONLY")
    hy_calls = call_rows(review_rows, catalog, smap, "HYBRID_REVIEW") if len(review_rows) else pd.DataFrame(
        columns=[
            "query_id", "status", "target_id", "reason", "confidence",
            "model_returned", "input_tokens", "output_tokens", "condition",
            "repeat", "call_status", "elapsed_sec", "runtime_source",
        ]
    )

    llm_calls.to_csv(LLM_CALLS, index=False, encoding="utf-8-sig")
    hy_calls.to_csv(HYBRID_CALLS, index=False, encoding="utf-8-sig")

    llm_cons = consensus_map(llm_calls)
    hy_cons = consensus_map(hy_calls)

    rows = []
    for _, r in qin.iterrows():
        qid = ec.clean(r.get("query_id"))
        llm_status, llm_target = llm_cons.get(qid, ("REVIEW", ""))
        rows.append({
            "query_id": qid,
            "system": "LLM_ONLY",
            "status": llm_status,
            "target_id": llm_target,
            "decision_source": "LLM_FULL_CATALOG",
            "gate_reason": "",
        })

        g = gate_map[qid]
        if ec.clean(g.get("gate_status")).upper() == "FOUND":
            h_status = "FOUND"
            h_target = ec.clean(g.get("gate_target_id"))
            source = "DETERMINISTIC_GATE"
        else:
            h_status, h_target = hy_cons.get(qid, ("REVIEW", ""))
            source = "LLM_AFTER_GATE_REVIEW"
        rows.append({
            "query_id": qid,
            "system": "HYBRID",
            "status": h_status,
            "target_id": h_target,
            "decision_source": source,
            "gate_reason": ec.clean(g.get("gate_reason")),
        })

    pred = pd.DataFrame(rows)
    if len(pred) != 2 * len(qin):
        raise RuntimeError("Unexpected case-level prediction row count")
    pred.to_csv(PREDICTIONS, index=False, encoding="utf-8-sig")

    meta["actual_llm_only_ok_calls"] = int(
        llm_calls["call_status"].astype(str).eq("OK").sum()
    )
    meta["actual_hybrid_ok_calls"] = int(
        hy_calls["call_status"].astype(str).eq("OK").sum()
    )
    meta["prediction_file_sha256"] = sha256(PREDICTIONS)
    RUN_METADATA.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OUT] {LLM_CALLS}")
    print(f"[OUT] {HYBRID_CALLS}")
    print(f"[OUT] {PREDICTIONS}")
    print(f"[OUT] {RUN_METADATA}")


if __name__ == "__main__":
    main()
