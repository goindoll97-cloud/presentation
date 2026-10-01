# -*- coding: utf-8 -*-
"""Preflight and freeze the LLM-only vs Hybrid retrieval protocol.

No GOLD file is read by this script.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import eval_common as ec
import retrieval_runtime as rt

ROOT = Path(__file__).resolve().parent
INTER = ROOT / "intermediate"
INTER.mkdir(parents=True, exist_ok=True)

GATE_FILE = INTER / "hybrid_gate_preflight.csv"
PROMPT_MANIFEST = INTER / "prompt_manifest.csv"
OUT = ROOT / "EVAL_PROTOCOL_FREEZE.json"
RUNNER = ROOT / "04_run_retrieval_llm_hybrid.py"
COMMON = ROOT / "eval_common.py"
RUNTIME = ROOT / "retrieval_runtime.py"
RETRIEVAL_FREEZE = ROOT / "RETRIEVAL_DATASET_FREEZE.json"
PUBCHEM_FREEZE = ROOT / "SHARED_PUBCHEM_FREEZE.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for p in [RUNNER, COMMON, RUNTIME, RETRIEVAL_FREEZE, PUBCHEM_FREEZE]:
        if not p.exists():
            raise FileNotFoundError(p)

    qin, catalog, smap, frozen_checks = ec.load_inputs()

    gates = pd.DataFrame([
        ec.gate_query(row, catalog, smap)
        for _, row in qin.iterrows()
    ])
    if set(gates["query_id"]) != set(qin["query_id"]):
        raise RuntimeError("Gate preflight query set differs from retrieval INPUT")
    if not gates["gate_status"].isin({"FOUND", "REVIEW"}).all():
        raise RuntimeError("Hybrid gate may only emit FOUND or REVIEW during preflight")
    gates.to_csv(GATE_FILE, index=False, encoding="utf-8-sig")

    allowed_targets = set(catalog["target_id"])
    bad_gate_target = gates[
        gates["gate_status"].eq("FOUND")
        & ~gates["gate_target_id"].isin(allowed_targets)
    ]
    if len(bad_gate_target):
        raise RuntimeError(
            "Hybrid gate emitted target outside frozen catalog: "
            + str(bad_gate_target.to_dict("records"))
        )

    review_ids = set(gates.loc[gates["gate_status"].eq("REVIEW"), "query_id"])
    manifest_rows = []
    for _, row in qin.iterrows():
        prompt = ec.build_prompt(row, catalog, smap)
        qid = ec.clean(row.get("query_id"))
        h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        manifest_rows.append({
            "query_id": qid,
            "condition": "LLM_ONLY",
            "prompt_sha256": h,
            "prompt_chars": len(prompt),
        })
        if qid in review_ids:
            manifest_rows.append({
                "query_id": qid,
                "condition": "HYBRID_REVIEW",
                "prompt_sha256": h,
                "prompt_chars": len(prompt),
            })

    pm = pd.DataFrame(manifest_rows)
    pm.to_csv(PROMPT_MANIFEST, index=False, encoding="utf-8-sig")

    pvt = pm.pivot_table(
        index="query_id", columns="condition", values="prompt_sha256",
        aggfunc="first",
    )
    both = pvt.dropna(subset=["LLM_ONLY", "HYBRID_REVIEW"])
    prompt_equal = bool((both["LLM_ONLY"] == both["HYBRID_REVIEW"]).all())
    if not prompt_equal:
        raise RuntimeError("LLM-only and Hybrid-review prompts differ for the same query")

    gate_counts = gates["gate_status"].value_counts().to_dict()
    reason_counts = gates["gate_reason"].value_counts().to_dict()

    freeze = {
        "protocol": "STUDY2_LLM_VS_HYBRID_CAS_RETRIEVAL_V1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_question": (
            "Given candidate CAS identifier(s) and shared PubChem SMILES, retrieve the "
            "correct target from the full frozen regulatory catalog, or return NOT_FOUND."
        ),
        "systems": {
            "LLM_ONLY": (
                "Every query is evaluated by the LLM against the full frozen regulatory catalog."
            ),
            "HYBRID": (
                "High-confidence exact-CAS, exact mixture-component, or unique RDKit parent/salt "
                "matches are decided deterministically; all remaining queries are sent to the "
                "same LLM with the same full-catalog prompt used by LLM-only."
            ),
        },
        "candidate_visible_information": [
            "CAS identifier(s)",
            "PubChem-derived SMILES for exactly those CAS identifiers",
        ],
        "forbidden_candidate_information": [
            "candidate chemical name",
            "PubChem title",
            "PubChem synonyms",
            "PubChem classification",
            "candidate category",
            "row-specific target scope",
            "GOLD label or GOLD target",
        ],
        "model": rt.MODEL,
        "prompt_version": rt.PROMPT_VERSION,
        "system_prompt_sha256": hashlib.sha256(
            rt.SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "n_repeats": int(rt.N_REPEATS),
        "sampling_control": rt.SAMPLING_MODE,
        "max_tokens": int(rt.MAX_TOKENS),
        "effort": rt.EFFORT,
        "transport_amendment": (
            "Deprecated temperature parameter removed before any successful LLM evaluation call. "
            "Model, prompt, retrieval data, catalog, gate logic, repeat count, max_tokens, and effort are unchanged."
        ),
        "n_queries": int(len(qin)),
        "n_catalog_targets": int(len(catalog)),
        "hybrid_gate_counts": {str(k): int(v) for k, v in gate_counts.items()},
        "hybrid_gate_reason_counts": {str(k): int(v) for k, v in reason_counts.items()},
        "expected_llm_only_calls": int(len(qin) * rt.N_REPEATS),
        "expected_hybrid_llm_calls": int(
            gate_counts.get("REVIEW", 0) * rt.N_REPEATS
        ),
        "hybrid_review_prompt_identical_to_llm_only": prompt_equal,
        "frozen_input_checks": frozen_checks,
        "sha256": {
            "eval_common.py": sha256(COMMON),
            "retrieval_runtime.py": sha256(RUNTIME),
            "04_run_retrieval_llm_hybrid.py": sha256(RUNNER),
            "RETRIEVAL_DATASET_FREEZE.json": sha256(RETRIEVAL_FREEZE),
            "SHARED_PUBCHEM_FREEZE.json": sha256(PUBCHEM_FREEZE),
            "intermediate/hybrid_gate_preflight.csv": sha256(GATE_FILE),
            "intermediate/prompt_manifest.csv": sha256(PROMPT_MANIFEST),
        },
    }
    OUT.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(freeze, ensure_ascii=False, indent=2))
    print(f"[OUT] {GATE_FILE}")
    print(f"[OUT] {PROMPT_MANIFEST}")
    print(f"[FROZEN] {OUT}")


if __name__ == "__main__":
    main()
