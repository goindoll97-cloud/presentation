# -*- coding: utf-8 -*-
"""Freeze Study 2 V6 end-to-end LLM-vs-hybrid protocol before execution."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INTER = ROOT / "intermediate_v6"
INTER.mkdir(parents=True, exist_ok=True)
BENCH = ROOT / "data" / "identity_e2e_v6_72.csv"
RUNTIME = ROOT / "identity_shared_runtime_V5.py"
ENGINE = ROOT / "cheminformatics_identity_V5_FAIR.py"
RUNNER = ROOT / "09_run_identity_e2e_LLM_HYBRID_V6.py"
OUT = INTER / "08_v6_protocol_freeze.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for p in (BENCH, RUNTIME, ENGINE, RUNNER):
        if not p.exists():
            raise FileNotFoundError(p)
    freeze = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": "STUDY2_E2E_V6_LLM_VS_HYBRID",
        "analysis_status": "REFRAMED_END_TO_END_EVALUATION_REUSING_72_CASES",
        "benchmark_file": BENCH.name,
        "benchmark_sha256": sha256(BENCH),
        "runtime_file": RUNTIME.name,
        "runtime_sha256": sha256(RUNTIME),
        "hybrid_gate_file": ENGINE.name,
        "hybrid_gate_sha256": sha256(ENGINE),
        "runner_file": RUNNER.name,
        "runner_sha256": sha256(RUNNER),
        "design": {
            "common_external_db": "PubChem PUG REST exact-CAS verified lookup",
            "system_1": "LLM + PubChem DB for every resolvable case",
            "system_2": "Hybrid + PubChem DB: deterministic low-cost structure gate first; LLM only on REVIEW",
            "outputs": ["MATCH", "NO_MATCH", "REVIEW"],
            "primary_endpoint": "overall_correct_resolution",
            "efficiency_endpoints": ["llm_case_rate", "llm_calls", "input_tokens", "output_tokens", "api_cost_usd", "llm_api_elapsed_sec"],
            "note": "The 72 cases are reused from the previous controlled-structure evaluation; V6 is a newly framed end-to-end evaluation and is not labelled an independent confirmatory holdout."
        },
    }
    OUT.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(freeze, ensure_ascii=False, indent=2))
    print(f"[SAVED] {OUT}")


if __name__ == "__main__":
    main()
