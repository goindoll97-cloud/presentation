# -*- coding: utf-8 -*-
"""Freeze FAIR V5 code + LLM protocol before constructing/opening final holdout."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INTER = ROOT / "intermediate"
INTER.mkdir(parents=True, exist_ok=True)
ENGINE = ROOT / "cheminformatics_identity_V5_FAIR.py"
RUNTIME = ROOT / "identity_shared_runtime_V5.py"
STEP05 = ROOT / "05_compare_identity_SHARED_DB_FAIR_V5.py"
OUT = INTER / "04D_fair_v5_protocol_freeze.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path.name}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    for p in [ENGINE, RUNTIME, STEP05]:
        if not p.exists():
            raise FileNotFoundError(f"Missing protocol file: {p.name}")
    eng = load_module(ENGINE, "identity_v5_engine_freeze")
    step05 = load_module(STEP05, "identity_v5_step05_freeze")
    tests = eng.generic_self_tests()
    if not tests or not all(bool(r.get("pass")) for r in tests):
        raise RuntimeError("Generic benchmark-independent V5 self-tests did not all pass")

    llm_protocol = step05.llm_protocol_config()
    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": "STUDY2_FAIR_V5",
        "engine_version": eng.ENGINE_VERSION,
        "policy_version": eng.POLICY_VERSION,
        "engine_file": ENGINE.name,
        "engine_sha256": sha256(ENGINE),
        "runtime_file": RUNTIME.name,
        "runtime_sha256": sha256(RUNTIME),
        "step05_file": STEP05.name,
        "step05_sha256": sha256(STEP05),
        "llm_protocol": llm_protocol,
        "generic_self_tests": tests,
        "confirmatory_requirement": (
            "Construct/curate the final holdout only after this protocol freeze, then freeze the exact holdout bytes with Step 04E before any Step-05 evaluation. "
            "The final holdout must not have been used to design/tune the engine or prompt, must be candidate-CAS-disjoint and reference-parent-disjoint from development data, and must carry an explicit isomer_scope for every row."
        ),
        "current_60_and_134_case_sets_status": "DEVELOPMENT_OR_EXPLORATORY_ONLY_AFTER_V5_METHOD_REFINEMENT",
    }
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\n[FROZEN] {OUT}")


if __name__ == "__main__":
    main()
