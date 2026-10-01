# -*- coding: utf-8 -*-
"""Study 2 V6: end-to-end regulatory identity evaluation.

Comparison
----------
1) LLM + PubChem DB: every DB-resolved case is judged by the LLM.
2) Hybrid + PubChem DB: a deterministic low-cost structure gate decides clear
   cases; only gate=REVIEW cases are sent to the LLM.

The benchmark itself contains no SMILES. Candidate and reference structures are
retrieved at run time from the same exact-CAS-verified PubChem pipeline.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INTER = ROOT / "intermediate_v6"
INTER.mkdir(parents=True, exist_ok=True)
BENCH = ROOT / "data" / "identity_e2e_v6_72.csv"
FREEZE = INTER / "08_v6_protocol_freeze.json"
RUNTIME_FILE = ROOT / "identity_shared_runtime_V5.py"
ENGINE_FILE = ROOT / "cheminformatics_identity_V5_FAIR.py"
EXECUTE = os.getenv("IDENTITY_V6_EXECUTE_CLAUDE", "0").strip().lower() in {"1", "true", "yes", "on"}

PROMPT_VERSION = "study2-e2e-v6-regulatory-scope-pubchem-20261001"
ANALYSIS_STATUS = "REFRAMED_END_TO_END_EVALUATION_REUSING_72_CASES"
SYSTEMS = ["LLM_PUBCHEM", "HYBRID_PUBCHEM"]

# Current Claude Sonnet 5 list price as of 2026-10-01; override in .env if needed.
INPUT_USD_PER_MTOK = float(os.getenv("IDENTITY_V6_INPUT_USD_PER_MTOK", "2.0"))
OUTPUT_USD_PER_MTOK = float(os.getenv("IDENTITY_V6_OUTPUT_USD_PER_MTOK", "10.0"))
PRICING_SOURCE = "Anthropic Claude Sonnet 5: USD 2/MTok input, USD 10/MTok output; checked 2026-10-01"

SYSTEM_PROMPT_V6 = """You are evaluating chemical-identity membership under a written regulatory scope.
The candidate and reference records were retrieved from PubChem using exact CAS-number verification.
Use the regulatory scope text and the supplied PubChem records to decide whether the candidate belongs to the stated identity scope.
Distinguish salts of the stated parent from covalent derivatives, positional/structural analogs, and other different substances.
Return REVIEW when the supplied DB information is insufficient or the identity relationship cannot be resolved reliably.
Do not use any benchmark truth label or hidden information. Return only the requested JSON object."""


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runtime = _load_module(RUNTIME_FILE, "study2_v6_runtime")
engine = _load_module(ENGINE_FILE, "study2_v6_gate")
runtime.CACHE_DIR = INTER / "identity_llm_cache_v6"
runtime.CACHE_DIR.mkdir(parents=True, exist_ok=True)
runtime.SYSTEM_PROMPT = SYSTEM_PROMPT_V6


def clean(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    return "" if s.lower() in {"nan", "none", "null", "<na>"} else s


def as_bool(x) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return clean(x).lower() in {"1", "true", "yes", "y"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def required(df: pd.DataFrame, cols: list[str]) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing benchmark columns: {miss}")


def verify_freeze() -> dict:
    if not FREEZE.exists():
        raise RuntimeError("Run 08_freeze_identity_e2e_V6.py before Step 09")
    f = json.loads(FREEZE.read_text(encoding="utf-8"))
    checks = {
        "benchmark_sha256_match": clean(f.get("benchmark_sha256")) == sha256(BENCH),
        "runtime_sha256_match": clean(f.get("runtime_sha256")) == sha256(RUNTIME_FILE),
        "hybrid_gate_sha256_match": clean(f.get("hybrid_gate_sha256")) == sha256(ENGINE_FILE),
        "runner_sha256_match": clean(f.get("runner_sha256")) == sha256(Path(__file__).resolve()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"V6 protocol changed after freeze: {checks}")
    f["verification_checks"] = checks
    return f


def prompt_for_v6(row: pd.Series, condition: str) -> str:
    if condition not in {"E2E_LLM_FULL", "E2E_HYBRID_REVIEW"}:
        raise ValueError(condition)
    payload = {
        "case_id": clean(row.get("case_id")),
        "regulatory_scope_text": clean(row.get("regulatory_scope_text")),
        "reference_parent": {
            "name": clean(row.get("reference_parent_name")),
            "cas": clean(row.get("reference_parent_cas")),
            "pubchem_title": clean(row.get("reference_pubchem_title")),
            "pubchem_smiles": clean(row.get("reference_smiles")),
            "structure_status": clean(row.get("reference_structure_status")),
        },
        "candidate": {
            "name": clean(row.get("candidate_name")),
            "cas": clean(row.get("candidate_cas")),
            "pubchem_title": clean(row.get("candidate_pubchem_title")),
            "pubchem_smiles": clean(row.get("candidate_smiles")),
            "structure_status": clean(row.get("candidate_structure_status")),
        },
        "isomer_scope": clean(row.get("isomer_scope")),
        "task": "Return MATCH if the candidate belongs to the written regulatory identity scope, NO_MATCH if it does not, or REVIEW if unresolved.",
    }
    if condition == "E2E_HYBRID_REVIEW":
        payload["low_cost_prefilter"] = {
            "decision": "REVIEW",
            "reason": clean(row.get("gate_reason")),
            "instruction": "The low-cost structure gate could not make a reliable final decision; resolve the case using the regulatory text and PubChem records.",
        }
    return "INPUT=" + json.dumps(payload, ensure_ascii=False) + "\nReturn decision=MATCH, NO_MATCH, or REVIEW."


runtime.set_prompt_builder(prompt_for_v6, PROMPT_VERSION)


def consensus(values: list[str]) -> str:
    vals = [clean(v).upper() if clean(v).upper() in {"MATCH", "NO_MATCH", "REVIEW"} else "REVIEW" for v in values]
    if not vals:
        return "REVIEW"
    counts = {x: vals.count(x) for x in ("MATCH", "NO_MATCH", "REVIEW")}
    winner, n = max(counts.items(), key=lambda z: z[1])
    return winner if n > len(vals) / 2 else "REVIEW"


def pred_bool(decision: str) -> Optional[bool]:
    d = clean(decision).upper()
    if d == "MATCH":
        return True
    if d == "NO_MATCH":
        return False
    return None


def enrich_from_pubchem(bench: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    refs = bench[["reference_parent_cas", "reference_parent_name"]].rename(columns={"reference_parent_cas": "cas", "reference_parent_name": "chemical_name"})
    cands = bench[["candidate_cas", "candidate_name"]].rename(columns={"candidate_cas": "cas", "candidate_name": "chemical_name"})
    inv = pd.concat([refs, cands], ignore_index=True).drop_duplicates("cas").reset_index(drop=True)
    inv["smiles"] = ""
    t0 = time.perf_counter()
    resolved, audit = runtime.enrich_inventory_with_pubchem(inv, INTER / "09_v6_pubchem_cache.csv")
    lookup_wall = float(time.perf_counter() - t0)
    if (audit["structure_status"].astype(str) == "PUBCHEM_REQUEST_ERROR").any():
        raise RuntimeError("Transient PubChem error: aborting completed V6 run")
    m = resolved.set_index("cas")
    out = bench.copy()
    for prefix, cas_col in (("candidate", "candidate_cas"), ("reference", "reference_parent_cas")):
        out[f"{prefix}_smiles"] = out[cas_col].map(m["smiles"])
        out[f"{prefix}_structure_status"] = out[cas_col].map(m["structure_status"])
        out[f"{prefix}_pubchem_title"] = out[cas_col].map(m["chemical_name"] if "chemical_name" in m.columns else pd.Series(dtype=str))
        if "pubchem_cid" in m.columns:
            out[f"{prefix}_pubchem_cid"] = out[cas_col].map(m["pubchem_cid"])
        if "pubchem_record_url" in m.columns:
            out[f"{prefix}_pubchem_record_url"] = out[cas_col].map(m["pubchem_record_url"])
    # Use PubChem-returned title from audit when available.
    aa = audit.drop_duplicates("cas").set_index("cas")
    if "pubchem_title" in aa.columns:
        out["candidate_pubchem_title"] = out["candidate_cas"].map(aa["pubchem_title"])
        out["reference_pubchem_title"] = out["reference_parent_cas"].map(aa["pubchem_title"])
    return out, audit, lookup_wall


def build_gate(bench: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    rows = []
    total_sec = 0.0
    for _, r in bench.iterrows():
        exact = clean(r.get("candidate_structure_status")) == "PUBCHEM_EXACT_CAS_VERIFIED" and clean(r.get("reference_structure_status")) == "PUBCHEM_EXACT_CAS_VERIFIED"
        if not exact:
            d, why, sec = "REVIEW", "DB_STRUCTURE_UNRESOLVED", 0.0
        else:
            t0 = time.perf_counter()
            d, why = engine.compare_salt_parent_v5(clean(r.get("candidate_smiles")), clean(r.get("reference_smiles")), clean(r.get("isomer_scope")))
            sec = float(time.perf_counter() - t0)
        total_sec += sec
        rows.append({"case_id": clean(r.get("case_id")), "gate_decision": d, "gate_reason": why, "gate_elapsed_sec": sec})
    return pd.DataFrame(rows), total_sec


def call_rows(rows: pd.DataFrame, condition: str) -> pd.DataFrame:
    out = []
    for rep in range(1, int(runtime.N_REPEATS) + 1):
        for _, r in rows.iterrows():
            ans = runtime.run_one_llm(r.to_dict(), condition, rep)
            ans["arm"] = condition
            out.append(ans)
            print(f"[{condition}] repeat={rep} case={clean(r.get('case_id'))} decision={clean(ans.get('decision'))} status={clean(ans.get('call_status'))}", flush=True)
    return pd.DataFrame(out)


def calls_cost(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "api_cost_usd": 0.0, "llm_api_elapsed_sec": 0.0}
    ok = df[df["call_status"].astype(str).eq("OK")].copy()
    it = pd.to_numeric(ok.get("input_tokens", 0), errors="coerce").fillna(0).sum()
    ot = pd.to_numeric(ok.get("output_tokens", 0), errors="coerce").fillna(0).sum()
    sec = pd.to_numeric(ok.get("elapsed_sec", 0), errors="coerce").fillna(0).sum()
    cost = float(it) / 1_000_000 * INPUT_USD_PER_MTOK + float(ot) / 1_000_000 * OUTPUT_USD_PER_MTOK
    return {"llm_calls": int(len(ok)), "input_tokens": int(it), "output_tokens": int(ot), "api_cost_usd": cost, "llm_api_elapsed_sec": float(sec)}


def build_caselevel(bench: pd.DataFrame, full_calls: pd.DataFrame, hybrid_calls: pd.DataFrame) -> pd.DataFrame:
    full_map = {}
    if not full_calls.empty:
        for cid, g in full_calls.groupby("case_id"):
            full_map[cid] = consensus(g["decision"].tolist())
    hy_map = {}
    if not hybrid_calls.empty:
        for cid, g in hybrid_calls.groupby("case_id"):
            hy_map[cid] = consensus(g["decision"].tolist())
    rows = []
    for _, r in bench.iterrows():
        cid = clean(r.get("case_id"))
        truth = clean(r.get("reference_membership")).upper() == "MATCH"
        db_ok = clean(r.get("candidate_structure_status")) == "PUBCHEM_EXACT_CAS_VERIFIED" and clean(r.get("reference_structure_status")) == "PUBCHEM_EXACT_CAS_VERIFIED"
        d_full = full_map.get(cid, "REVIEW") if db_ok else "REVIEW"
        gd = clean(r.get("gate_decision")).upper()
        if gd in {"MATCH", "NO_MATCH"}:
            d_hybrid, source = gd, "LOW_COST_STRUCTURE_GATE"
        elif db_ok:
            d_hybrid, source = hy_map.get(cid, "REVIEW"), "LLM_AFTER_GATE_REVIEW"
        else:
            d_hybrid, source = "REVIEW", "DB_UNRESOLVED"
        for system, d, source in [
            ("LLM_PUBCHEM", d_full, "LLM_ALL_DB_RESOLVED_CASES" if db_ok else "DB_UNRESOLVED"),
            ("HYBRID_PUBCHEM", d_hybrid, source),
        ]:
            pb = pred_bool(d)
            decided = pb is not None
            correct = bool(decided and pb == truth)
            rows.append({
                "case_id": cid, "system": system, "decision": d, "decision_source": source,
                "truth_bool": truth, "pred_bool": pb, "decided": decided, "correct": correct,
                "false_safe": bool(decided and truth and not pb),
                "false_positive": bool(decided and (not truth) and pb),
                "overall_correct_resolution": correct,
                "candidate_cas": clean(r.get("candidate_cas")), "candidate_name": clean(r.get("candidate_name")),
                "reference_parent_name": clean(r.get("reference_parent_name")),
                "challenge_class": clean(r.get("challenge_class")), "difficulty": clean(r.get("difficulty")),
                "gate_decision": clean(r.get("gate_decision")), "gate_reason": clean(r.get("gate_reason")),
            })
    return pd.DataFrame(rows)


def main() -> None:
    freeze = verify_freeze()
    bench = pd.read_csv(BENCH, dtype=str).fillna("")
    required(bench, ["case_id", "regulatory_scope_text", "reference_parent_name", "reference_parent_cas", "candidate_name", "candidate_cas", "reference_membership", "isomer_scope"])
    if any(c.lower().endswith("smiles") or c.lower() == "smiles" for c in bench.columns):
        raise RuntimeError("V6 benchmark must not contain pre-supplied SMILES")
    if bench["case_id"].duplicated().any():
        raise RuntimeError("case_id must be unique")

    bench, audit, lookup_wall = enrich_from_pubchem(bench)
    gate, gate_total_sec = build_gate(bench)
    bench = bench.merge(gate, on="case_id", how="left")
    bench.to_csv(INTER / "09_v6_benchmark_enriched.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(INTER / "09_v6_pubchem_audit.csv", index=False, encoding="utf-8-sig")
    gate.to_csv(INTER / "09_v6_hybrid_gate_preflight.csv", index=False, encoding="utf-8-sig")

    db_ok = bench["candidate_structure_status"].eq("PUBCHEM_EXACT_CAS_VERIFIED") & bench["reference_structure_status"].eq("PUBCHEM_EXACT_CAS_VERIFIED")
    full_rows = bench[db_ok].copy()
    hybrid_rows = bench[db_ok & bench["gate_decision"].eq("REVIEW")].copy()

    manifests = []
    for _, r in full_rows.iterrows():
        p = prompt_for_v6(r, "E2E_LLM_FULL")
        manifests.append({"case_id": clean(r.get("case_id")), "arm": "E2E_LLM_FULL", "prompt_sha256": hashlib.sha256(p.encode()).hexdigest(), "prompt_text": p})
    for _, r in hybrid_rows.iterrows():
        p = prompt_for_v6(r, "E2E_HYBRID_REVIEW")
        manifests.append({"case_id": clean(r.get("case_id")), "arm": "E2E_HYBRID_REVIEW", "prompt_sha256": hashlib.sha256(p.encode()).hexdigest(), "prompt_text": p})
    pd.DataFrame(manifests).to_csv(INTER / "09_v6_prompt_manifest.csv", index=False, encoding="utf-8-sig")

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "analysis_status": ANALYSIS_STATUS,
        "benchmark_sha256": sha256(BENCH),
        "protocol_freeze": freeze,
        "common_external_db": "PubChem exact-CAS verified",
        "llm_model": runtime.ANTHROPIC_MODEL,
        "llm_repeats": int(runtime.N_REPEATS),
        "pricing": {"input_usd_per_mtok": INPUT_USD_PER_MTOK, "output_usd_per_mtok": OUTPUT_USD_PER_MTOK, "source_note": PRICING_SOURCE},
        "n_cases": int(len(bench)), "n_db_resolved": int(db_ok.sum()),
        "n_hybrid_gate_review": int((db_ok & bench["gate_decision"].eq("REVIEW")).sum()),
        "planned_llm_calls_full": int(len(full_rows) * runtime.N_REPEATS),
        "planned_llm_calls_hybrid": int(len(hybrid_rows) * runtime.N_REPEATS),
        "pubchem_lookup_wall_sec": lookup_wall,
        "hybrid_gate_compute_sec": gate_total_sec,
    }
    (INTER / "09_v6_run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 92)
    print("STUDY 2 E2E V6 — LLM+PubChem vs Hybrid+PubChem")
    print("=" * 92)
    print(f"Cases={len(bench)} | DB-resolved={int(db_ok.sum())}")
    print(f"Full LLM planned calls={len(full_rows) * runtime.N_REPEATS}")
    print(f"Hybrid gate REVIEW cases={len(hybrid_rows)} | planned Hybrid LLM calls={len(hybrid_rows) * runtime.N_REPEATS}")
    print(f"Paid Claude execution enabled: {EXECUTE}")
    if not EXECUTE:
        print("[STOP BEFORE PAID API] Inspect 09_v6_hybrid_gate_preflight.csv and 09_v6_prompt_manifest.csv first.")
        return

    runtime.validate_api_key_for_http_header()
    full_calls = call_rows(full_rows, "E2E_LLM_FULL")
    hybrid_calls = call_rows(hybrid_rows, "E2E_HYBRID_REVIEW")
    calls = pd.concat([full_calls, hybrid_calls], ignore_index=True) if len(hybrid_calls) else full_calls.copy()
    calls.to_csv(INTER / "09_v6_llm_call_outputs.csv", index=False, encoding="utf-8-sig")

    cases = build_caselevel(bench, full_calls, hybrid_calls)
    cases.to_csv(INTER / "09_v6_system_predictions_caselevel.csv", index=False, encoding="utf-8-sig")

    full_e = calls_cost(full_calls)
    hy_e = calls_cost(hybrid_calls)
    eff = []
    for system, e, n_llm_cases, gate_sec in [
        ("LLM_PUBCHEM", full_e, len(full_rows), 0.0),
        ("HYBRID_PUBCHEM", hy_e, len(hybrid_rows), gate_total_sec),
    ]:
        eff.append({
            "system": system, "n_total": len(bench), "llm_cases": n_llm_cases,
            "llm_case_rate": n_llm_cases / len(bench), **e,
            "shared_pubchem_lookup_wall_sec": lookup_wall, "structure_gate_sec": gate_sec,
            "operational_elapsed_proxy_sec": lookup_wall + gate_sec + e["llm_api_elapsed_sec"],
            "input_usd_per_mtok": INPUT_USD_PER_MTOK, "output_usd_per_mtok": OUTPUT_USD_PER_MTOK,
        })
    pd.DataFrame(eff).to_csv(INTER / "09_v6_efficiency.csv", index=False, encoding="utf-8-sig")
    print("[DONE] V6 paid evaluation completed. Run 10_make_identity_e2e_results_V6.py next.")


if __name__ == "__main__":
    main()
