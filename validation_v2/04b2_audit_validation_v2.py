# -*- coding: utf-8 -*-
"""Run the Validation V2 audit with additive source-evidence patches.

This wrapper preserves the frozen/base source-verification evidence file while
allowing documented corrections to be layered on transparently. It is used for
the 2026-10-01 correction that removes reaction-mixture target 2021-1-1059 from
the NAMED_MIXTURE benchmark and replaces it with official named mixture
2023-1-1162.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SEEDS = ROOT / "seeds"
SOURCES = ROOT / "sources"
BASE_EVIDENCE = SEEDS / "source_verification_evidence_v2.csv"
PATCH_GLOB = "source_verification_evidence_patch_*.csv"
MERGED_EVIDENCE = SOURCES / "source_verification_evidence_merged_for_audit.csv"
BASE_AUDIT_SCRIPT = ROOT / "04b_audit_validation_v2.py"


def build_merged_evidence() -> Path:
    files = [BASE_EVIDENCE] + sorted(SEEDS.glob(PATCH_GLOB))
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing source-evidence files: " + ", ".join(missing))

    frames = []
    for path in files:
        df = pd.read_csv(path, dtype=str).fillna("")
        df["_evidence_file"] = path.name
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)

    # Patches are additive. Duplicate keys are forbidden because silently
    # overriding source evidence would make provenance ambiguous.
    dup = merged.duplicated(["review_key", "category"], keep=False)
    if dup.any():
        bad = merged.loc[dup, ["review_key", "category", "_evidence_file"]]
        raise ValueError(
            "Duplicate source-evidence review keys across base/patch files:\n"
            + bad.to_string(index=False)
        )

    merged = merged.drop(columns=["_evidence_file"])
    SOURCES.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_EVIDENCE, index=False, encoding="utf-8", lineterminator="\n")
    print(f"[EVIDENCE] base+patch rows={len(merged)} files={len(files)}")
    print(f"[EVIDENCE] merged={MERGED_EVIDENCE}")
    return MERGED_EVIDENCE


def main() -> None:
    merged = build_merged_evidence()

    spec = importlib.util.spec_from_file_location("validation_v2_base_audit", BASE_AUDIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {BASE_AUDIT_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Point the existing audited logic at the reproducibly merged evidence file.
    mod.SOURCE_EVIDENCE = merged
    mod.main()


if __name__ == "__main__":
    main()
