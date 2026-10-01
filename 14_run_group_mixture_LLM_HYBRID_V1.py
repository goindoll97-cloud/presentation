# -*- coding: utf-8 -*-
"""Study 2 extension V1: LLM+PubChem vs Hybrid+PubChem for groups and mixtures.

Default is dry-run. Set GROUP_MIX_V1_EXECUTE_CLAUDE=1 only after inspecting
preflight and prompt-manifest outputs. Benchmark truth labels are never included
in the LLM prompt.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
INTER = ROOT / "intermediate_gm_v1"
INTER.mkdir(parents=True, exist_ok=True)
GROUP_FILE = DATA / "group_membership_v1_50.csv"
MIX_FILE = DATA / "mixture_threshold_v1_50.csv"
RULE_FILE = DATA / "regulatory_group_rules_v1.csv"
FREEZE = INTER / "13_group_mixture_protocol_freeze.json"
RUNTIME_FILE = ROOT / "identity_shared_runtime_V5.py"
ENGINE_FILE = ROOT / "regulatory_group_mixture_engine_V1.py"
EXECUTE = os.getenv("GROUP_MIX_V1_EXECUTE_CLAUDE", "0").strip().lower() in {"1", "true", "yes", "on"}
PROMPT_VERSION = "study2-group-mixture-v1-20261001"
INPUT_USD_PER_MTOK = float(os.getenv("GROUP_MIX_V1_INPUT_USD_PER_MTOK", "2.0"))
OUTPUT_USD_PER_MTOK = float(os.getenv("GROUP_MIX_V1_OUTPUT_USD_PER_MTOK", "10.0"))


def _ensure_benchmark_files() -> None:
    required = [RULE_FILE, GROUP_FILE, MIX_FILE]
    if all(p.exists() for p in required):
        return
    builder = ROOT / "12_build_group_mixture_benchmark_V1.py"
    if not builder.exists():
        raise FileNotFoundError("Benchmark files are missing and Step 12 builder is unavailable")
    subprocess.run([sys.executable, str(builder)], cwd=ROOT, check=True)


_ensure_benchmark_files()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runtime = load_module(RUNTIME_FILE, "gm_v1_runtime")
engine = load_module(ENGINE_FILE, "gm_v1_engine")
runtime.CACHE_DIR = INTER / "llm_cache_gm_v1"
runtime.CACHE_DIR.mkdir(parents=True, exist_ok=True)
runtime.SYSTEM_PROMPT = """You are evaluating membership under a Korean chemical regulatory scope.
Use only the supplied written scope, authoritative Annex-3 member registry,
candidate/component identity, PubChem record, and mixture concentration when present.
Return MATCH when the candidate or mixture satisfies the stated scope; NO_MATCH when it does not;
REVIEW only when the supplied information is insufficient. Pay close attention to the difference
between 'at least' (>=) and 'greater than' (>). Do not use hidden benchmark labels.
Return only the requested JSON object."""


def clean(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_freeze() -> dict:
    if not FREEZE.exists():
        raise RuntimeError("Run 13_freeze_group_mixture_V1.py first")
    f = json.loads(FREEZE.read_text(encoding="utf-8"))
    checks = {
        "regulatory_group_rules_v1.csv": RULE_FILE,
        "group_membership_v1_50.csv": GROUP_FILE,
        "mixture_threshold_v1_50.csv": MIX_FILE,
        "regulatory_group_mixture_engine_V1.py": ENGINE_FILE,
        "14_run_group_mixture_LLM_HYBRID_V1.py": Path(__file__).resolve(),
    }
    status = {k: f.get("files", {}).get(k) == sha256(p) for k, p in checks.items()}
    if not all(status.values()):
        raise RuntimeError(f"Protocol changed after freeze: {status}")
    return status


def member_registry() -> dict[str, list[dict]]:
    rules = pd.read_csv(RULE_FILE, dtype=str).fillna("")
    out = {}
    for rid, g in rules.groupby("rule_id", sort=False):
        out[rid] = [{"name": clean(r.member_name), "cas": clean(r.member_cas)} for r in g.itertuples(index=False)]
    return out


MEMBERS = member_registry()


def prompt_for(row: pd.Series, condition: str) -> str:
    task = clean(row.get("task_type"))
    rid = clean(row.get("rule_id"))
    payload = {
        "case_id": clean(row.get("case_id")),
        "task_type": task,
        "regulatory_id": clean(row.get("regulatory_id")),
        "regulatory_scope_text": clean(row.get("regulatory_scope_text")),
        "authoritative_annex3_members": MEMBERS.get(rid, []),
    }
    if task == "CHEMICAL_GROUP":
        payload["candidate"] = {
            "name": clean(row.get("candidate_name")),
            "cas": clean(row.get("candidate_cas")),
            "pubchem_title": clean(row.get("pubchem_title")),
            "pubchem_smiles": clean(row.get("smiles")),
            "structure_status": clean(row.get("structure_status")),
        }
        payload["task"] = "Decide whether the candidate belongs to the stated regulatory chemical group."
    elif task == "MIXTURE_THRESHOLD":
        payload["mixture_component"] = {
            "name": clean(row.get("component_name")),
            "cas": clean(row.get("component_cas")),
            "concentration_pct": row.get("component_concentration_pct"),
            "pubchem_title": clean(row.get("pubchem_title")),
            "pubchem_smiles": clean(row.get("smiles")),
            "structure_status": clean(row.get("structure_status")),
        }
        payload["task"] = "Decide whether this mixture case satisfies both group identity and the concentration condition in the regulatory scope."
    else:
        raise ValueError(task)
    if condition == "GM_HYBRID_REVIEW":
        payload["low_cost_gate"] = {"decision": "REVIEW", "reason": clean(row.get("gate_reason"))}
    return "INPUT=" + json.dumps(payload, ensure_ascii=False) + "\nReturn decision=MATCH, NO_MATCH, or REVIEW."


runtime.set_prompt_builder(prompt_for, PROMPT_VERSION)


def consensus(values: list[str]) -> str:
    vals = [clean(v).upper() if clean(v).upper() in {"MATCH", "NO_MATCH", "REVIEW"} else "REVIEW" for v in values]
    if not vals:
        return "REVIEW"
    counts = {x: vals.count(x) for x in ("MATCH", "NO_MATCH", "REVIEW")}
    winner, n = max(counts.items(), key=lambda z: z[1])
    return winner if n > len(vals) / 2 else "REVIEW"


def enrich(cases: pd.DataFrame):
    inventory = pd.DataFrame({
        "cas": cases.apply(lambda r: clean(r.get("candidate_cas")) or clean(r.get("component_cas")), axis=1),
        "chemical_name": cases.apply(lambda r: clean(r.get("candidate_name")) or clean(r.get("component_name")), axis=1),
    }).drop_duplicates("cas").reset_index(drop=True)
    inventory["smiles"] = ""
    t0 = time.perf_counter()
    resolved, audit = runtime.enrich_inventory_with_pubchem(inventory, INTER / "14_gm_pubchem_cache.csv")
    elapsed = float(time.perf_counter() - t0)
    mapping = resolved.drop_duplicates("cas").set_index("cas")
    out = cases.copy()
    keys = out.apply(lambda r: clean(r.get("candidate_cas")) or clean(r.get("component_cas")), axis=1)
    for c in ["smiles", "structure_status", "pubchem_cid", "pubchem_record_url"]:
        if c in mapping.columns:
            out[c] = keys.map(mapping[c])
    audit_map = audit.drop_duplicates("cas").set_index("cas")
    out["pubchem_title"] = keys.map(audit_map["pubchem_title"]) if "pubchem_title" in audit_map.columns else ""
    return out, audit, elapsed


def apply_gates(df: pd.DataFrame):
    rows = []
    total = 0.0
    for _, r in df.iterrows():
        t0 = time.perf_counter()
        if clean(r.get("task_type")) == "CHEMICAL_GROUP":
            d, why = engine.group_gate(clean(r.get("rule_id")), clean(r.get("candidate_cas")), clean(r.get("candidate_name")), clean(r.get("smiles")))
        else:
            d, why = engine.mixture_gate(clean(r.get("rule_id")), clean(r.get("component_cas")), clean(r.get("component_name")), float(r.get("component_concentration_pct")), clean(r.get("smiles")))
        sec = float(time.perf_counter() - t0)
        total += sec
        rows.append({"case_id": clean(r.get("case_id")), "gate_decision": d, "gate_reason": why, "gate_elapsed_sec": sec})
    return pd.DataFrame(rows), total


def call_rows(rows: pd.DataFrame, condition: str) -> pd.DataFrame:
    out = []
    for rep in range(1, int(runtime.N_REPEATS) + 1):
        for _, r in rows.iterrows():
            ans = runtime.run_one_llm(r.to_dict(), condition, rep)
            ans["arm"] = condition
            out.append(ans)
            print(f"[{condition}] rep={rep} case={clean(r.get('case_id'))} decision={clean(ans.get('decision'))}", flush=True)
    return pd.DataFrame(out)


def call_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "api_cost_usd": 0.0, "llm_api_elapsed_sec": 0.0}
    ok = df[df["call_status"].astype(str).eq("OK")].copy()
    it = pd.to_numeric(ok.get("input_tokens", 0), errors="coerce").fillna(0).sum()
    ot = pd.to_numeric(ok.get("output_tokens", 0), errors="coerce").fillna(0).sum()
    sec = pd.to_numeric(ok.get("elapsed_sec", 0), errors="coerce").fillna(0).sum()
    cost = float(it) / 1_000_000 * INPUT_USD_PER_MTOK + float(ot) / 1_000_000 * OUTPUT_USD_PER_MTOK
    return {"llm_calls": int(len(ok)), "input_tokens": int(it), "output_tokens": int(ot), "api_cost_usd": float(cost), "llm_api_elapsed_sec": float(sec)}


def main():
    freeze_checks = verify_freeze()
    g = pd.read_csv(GROUP_FILE)
    m = pd.read_csv(MIX_FILE)
    cases = pd.concat([g, m], ignore_index=True, sort=False).fillna("")
    enriched, audit, pubchem_sec = enrich(cases)
    gates, gate_sec = apply_gates(enriched)
    enriched = enriched.merge(gates, on="case_id", how="left")
    enriched.to_csv(INTER / "14_gm_hybrid_gate_preflight.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(INTER / "14_gm_pubchem_audit.csv", index=False, encoding="utf-8-sig")

    manifest = []
    for condition, rows in [("GM_LLM_FULL", enriched), ("GM_HYBRID_REVIEW", enriched[enriched["gate_decision"].eq("REVIEW")])]:
        for _, r in rows.iterrows():
            p = prompt_for(r, condition)
            manifest.append({"condition": condition, "case_id": r["case_id"], "task_type": r["task_type"], "prompt_chars": len(p), "prompt_sha256": hashlib.sha256(p.encode()).hexdigest()})
    pd.DataFrame(manifest).to_csv(INTER / "14_gm_prompt_manifest.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "execute_claude": EXECUTE,
        "n_cases": int(len(enriched)),
        "n_group_cases": int((enriched.task_type == "CHEMICAL_GROUP").sum()),
        "n_mixture_cases": int((enriched.task_type == "MIXTURE_THRESHOLD").sum()),
        "hybrid_gate_match": int((enriched.gate_decision == "MATCH").sum()),
        "hybrid_gate_no_match": int((enriched.gate_decision == "NO_MATCH").sum()),
        "hybrid_gate_review": int((enriched.gate_decision == "REVIEW").sum()),
        "expected_llm_full_calls": int(len(enriched) * runtime.N_REPEATS),
        "expected_hybrid_review_calls": int((enriched.gate_decision == "REVIEW").sum() * runtime.N_REPEATS),
        "pubchem_lookup_elapsed_sec": pubchem_sec,
        "gate_elapsed_sec": gate_sec,
        "freeze_verification": freeze_checks,
    }
    (INTER / "14_gm_run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if not EXECUTE:
        print("[DRY RUN] No Claude calls. Inspect preflight + prompt manifest, then set GROUP_MIX_V1_EXECUTE_CLAUDE=1.")
        return

    full = call_rows(enriched, "GM_LLM_FULL")
    review_rows = enriched[enriched["gate_decision"].eq("REVIEW")].copy()
    hy = call_rows(review_rows, "GM_HYBRID_REVIEW") if len(review_rows) else pd.DataFrame()
    full.to_csv(INTER / "14_gm_llm_full_calls.csv", index=False, encoding="utf-8-sig")
    hy.to_csv(INTER / "14_gm_hybrid_review_calls.csv", index=False, encoding="utf-8-sig")

    full_map = {cid: consensus(x.decision.tolist()) for cid, x in full.groupby("case_id")} if len(full) else {}
    hy_map = {cid: consensus(x.decision.tolist()) for cid, x in hy.groupby("case_id")} if len(hy) else {}
    rows = []
    for _, r in enriched.iterrows():
        cid = clean(r.case_id)
        gd = clean(r.gate_decision).upper()
        llm_d = full_map.get(cid, "REVIEW")
        if gd in {"MATCH", "NO_MATCH"}:
            hy_d, src = gd, "DETERMINISTIC_GATE"
        else:
            hy_d, src = hy_map.get(cid, "REVIEW"), "LLM_AFTER_REVIEW"
        for system, d, source in [("LLM_PUBCHEM", llm_d, "LLM_ALL_CASES"), ("HYBRID_PUBCHEM", hy_d, src)]:
            rows.append({
                "case_id": cid,
                "task_type": r.task_type,
                "rule_id": r.rule_id,
                "system": system,
                "decision": d,
                "decision_source": source,
                "reference_membership": r.reference_membership,
                "correct": d == r.reference_membership,
                "resolved": d in {"MATCH", "NO_MATCH"},
                "gate_decision": gd,
                "gate_reason": r.gate_reason,
                "boundary_class": clean(r.get("boundary_class")),
                "difficulty": clean(r.get("difficulty")),
            })
    caselevel = pd.DataFrame(rows)
    caselevel.to_csv(INTER / "14_gm_system_predictions_caselevel.csv", index=False, encoding="utf-8-sig")

    eff = []
    for system, calls in [("LLM_PUBCHEM", full), ("HYBRID_PUBCHEM", hy)]:
        s = call_stats(calls)
        s.update({
            "system": system,
            "n_cases": len(enriched),
            "llm_case_rate": 1.0 if system == "LLM_PUBCHEM" else len(review_rows) / len(enriched),
            "shared_pubchem_elapsed_sec": pubchem_sec,
            "deterministic_gate_elapsed_sec": 0.0 if system == "LLM_PUBCHEM" else gate_sec,
        })
        eff.append(s)
    pd.DataFrame(eff).to_csv(INTER / "14_gm_efficiency.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
