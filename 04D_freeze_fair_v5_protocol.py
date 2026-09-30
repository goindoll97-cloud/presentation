# -*- coding: utf-8 -*-
"""Freeze the FAIR V5 protocol before any independent final-holdout evaluation.

Run this after code review and generic self-tests, but BEFORE constructing or
opening the final holdout. The resulting manifest is checked by Step 05 V5.
"""
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
STEP05 = ROOT / "05_compare_identity_SHARED_DB_FAIR_V5.py"
OUT = INTER / "04D_fair_v5_protocol_freeze.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_engine():
    spec = importlib.util.spec_from_file_location("identity_v5_engine", ENGINE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load V5 engine")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    for p in [ENGINE, STEP05]:
        if not p.exists():
            raise FileNotFoundError(f"Missing protocol file: {p.name}")
    eng = load_engine()
    tests = eng.generic_self_tests()
    if not tests or not all(bool(r.get("pass")) for r in tests):
        raise RuntimeError("Generic benchmark-independent V5 self-tests did not all pass")

    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": "STUDY2_FAIR_V5",
        "engine_version": eng.ENGINE_VERSION,
        "policy_version": eng.POLICY_VERSION,
        "engine_file": ENGINE.name,
        "engine_sha256": sha256(ENGINE),
        "step05_file": STEP05.name,
        "step05_sha256": sha256(STEP05),
        "generic_self_tests": tests,
        "confirmatory_requirement": (
            "A final holdout is confirmatory only if it was frozen after this protocol freeze, "
            "was not used to design/tune the engine or prompt, and contains no candidate-CAS overlap "
            "with the development benchmark."
        ),
        "current_60_and_134_case_sets_status": "DEVELOPMENT_OR_EXPLORATORY_ONLY_AFTER_V5_METHOD_REFINEMENT",
    }
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\n[FROZEN] {OUT}")


if __name__ == "__main__":
    main()
