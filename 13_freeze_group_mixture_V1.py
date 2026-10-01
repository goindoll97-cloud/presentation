# -*- coding: utf-8 -*-
from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
INTER = ROOT / "intermediate_gm_v1"
INTER.mkdir(parents=True, exist_ok=True)
FILES = [
    DATA / "regulatory_group_rules_v1.csv",
    DATA / "group_membership_v1_50.csv",
    DATA / "mixture_threshold_v1_50.csv",
    ROOT / "regulatory_group_mixture_engine_V1.py",
    ROOT / "14_run_group_mixture_LLM_HYBRID_V1.py",
]

def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    missing = [str(p) for p in FILES if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing files before freeze. Run 12_build_group_mixture_benchmark_V1.py first: " + str(missing)
        )
    payload = {
        "protocol": "STUDY2_GROUP_MIXTURE_V1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_design": "SOURCE_ANCHORED_CLOSED_REGISTRY_V1",
        "files": {p.name: sha256(p) for p in FILES},
    }
    out = INTER / "13_group_mixture_protocol_freeze.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
