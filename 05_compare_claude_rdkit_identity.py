# -*- coding: utf-8 -*-
"""Study 2 / Step 05: Claude vs deterministic RDKit identity vs Hybrid.

Main comparisons
----------------
A) OPERATIONAL / CAS-only
   Claude receives: regulatory scope + candidate CAS only.
   RDKit system receives the same CAS, resolves a structure through an exact-CAS-
   verified PubChem lookup, then performs deterministic parent/salt normalization.
   Hybrid uses deterministic RDKit when it can decide and Claude only as fallback.

B) CONTROLLED / structure-informed
   Claude and RDKit both receive the same source-supported candidate SMILES and the
   same frozen reference-parent SMILES.  This separates tool-access advantage from
   the ability to execute structural identity logic reliably.

Sensitivity
-----------
Claude CAS+name is also evaluated because real enterprise inventories often contain
both fields.  It is not the primary operational comparison.

The reference labels are source-supported chemical-identity references, not expert
legal gold and not final compliance determinations.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
INTERMEDIATE = ROOT / "intermediate"
INTERMEDIATE.mkdir(parents=True, exist_ok=True)
ENGINE_DIR = ROOT / "engine"
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

try:
    import broad_salt_identity as bsi
except Exception as exc:
    raise ImportError("Could not import engine/broad_salt_identity.py. Run this script from the project root.") from exc

BENCHMARK_FILE = INTERMEDIATE / "04_identity_challenge_benchmark.csv"
RULE_SNAPSHOT_FILE = INTERMEDIATE / "04_identity_challenge_rules.csv"
CACHE_DIR = INTERMEDIATE / "identity_llm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = os.getenv("ANTHROPIC_VERSION", "2023-06-01").strip()
ANTHROPIC_MODEL = os.getenv("IDENTITY_ANTHROPIC_MODEL", "claude-sonnet-5").strip()
N_REPEATS = max(1, int(os.getenv("IDENTITY_LLM_REPEATS", "3")))
WORKERS = max(1, int(os.getenv("IDENTITY_ANTHROPIC_WORKERS", os.getenv("ANTHROPIC_WORKERS", "2"))))
TIMEOUT = max(30, int(os.getenv("IDENTITY_ANTHROPIC_TIMEOUT", "180")))
MAX_TOKENS = max(256, int(os.getenv("IDENTITY_ANTHROPIC_MAX_TOKENS", "700")))
EFFORT = os.getenv("IDENTITY_ANTHROPIC_EFFORT", "medium").strip().lower() or "medium"
FORCE = os.getenv("IDENTITY_FORCE_LLM", "0").strip().lower() in {"1", "true", "yes", "on"}
RUN_CAS_NAME = os.getenv("IDENTITY_RUN_CAS_NAME_SENSITIVITY", "1").strip().lower() not in {"0", "false", "no", "off"}
MAX_CASES = max(0, int(os.getenv("IDENTITY_MAX_CASES", "0")))
PROMPT_VERSION = "study2-chemical-identity-v1-20260911"
CACHE_VERSION = "claude-structured-identity-v1"

DECISIONS = {"MATCH", "NO_MATCH", "REVIEW"}
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "case_id": {"type": "string"},
        "decision": {"type": "string", "enum": ["MATCH", "NO_MATCH", "REVIEW"]},
        "reason": {"type": "string"},
        "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
    },
    "required": ["case_id", "decision", "reason", "confidence"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are evaluating chemical identity membership for a controlled research benchmark.
The regulatory scope has ALREADY been interpreted as a broad parent-and-salts scope.
Your task is only chemical identity resolution: decide whether the candidate is the same parent chemical or one of its counterion salts.
Do not make a legal compliance determination. Do not infer concentration, exemptions, dates, or other regulations.
Return REVIEW when the supplied information is insufficient to support a chemical-identity decision.
You have no external tools or database access in this experiment; use only the fields in the prompt.
Return only the requested JSON object."""


def clean(x) -> str:
    if x is None:
        return ""
    try:
        if bool(pd.isna(x)):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    return "" if s.lower() in {"nan", "none", "null", "<na>"} else s


def load_dotenv() -> None:
    for p in [ROOT / ".env", Path.cwd() / ".env"]:
        if not p.exists():
            continue
        try:
            for line in p.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
        except Exception:
            pass


def api_key() -> str:
    return clean(os.getenv("ANTHROPIC_API_KEY", ""))


def extract_json(content: Any) -> dict:
    if isinstance(content, dict):
        return content
    text = clean(content)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        decoder = json.JSONDecoder()
        for m in re.finditer(r"\{", text):
            try:
                obj, _ = decoder.raw_decode(text[m.start():])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    raise ValueError("No JSON object in Claude response")


def prompt_for(row: pd.Series, condition: str) -> str:
    base = {
        "case_id": clean(row.get("case_id")),
        "regulatory_scope_text": clean(row.get("regulatory_scope_text")),
        "candidate_cas": clean(row.get("candidate_cas")),
    }
    if condition == "CAS_NAME":
        base["candidate_name"] = clean(row.get("candidate_name"))
    elif condition == "STRUCTURE_INFORMED":
        base.update({
            "candidate_name": clean(row.get("candidate_name")),
            "candidate_smiles": clean(row.get("candidate_source_smiles")),
            "reference_parent_name": clean(row.get("reference_parent_name")),
            "reference_parent_smiles": clean(row.get("reference_parent_smiles")),
            "structural_rule": (
                "Treat disconnected counterion/solvate fragments as removable salt components. "
                "After parent-fragment normalization, MATCH only when candidate and reference represent the same parent identity; "
                "otherwise NO_MATCH. If stereochemical scope cannot be resolved from the supplied information, REVIEW."
            ),
        })
    return (
        f"INPUT_CONDITION={condition}\n"
        "Analyze this one case using only the supplied fields.\n"
        f"INPUT={json.dumps(base, ensure_ascii=False)}\n"
        "Return decision=MATCH, NO_MATCH, or REVIEW."
    )


def call_claude(row: pd.Series, condition: str) -> dict:
    key = api_key()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured")
    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt_for(row, condition)}],
        "thinking": {"type": "disabled"},
        "output_config": {
            "effort": EFFORT,
            "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
        },
    }
    headers = {
        "x-api-key": key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    last = None
    for attempt in range(1, 5):
        try:
            r = requests.post(ANTHROPIC_ENDPOINT, headers=headers, json=body, timeout=TIMEOUT)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                if attempt < 4:
                    retry_after = clean(r.headers.get("retry-after"))
                    try:
                        wait = max(float(retry_after), min(30.0, 2 ** attempt)) if retry_after else min(30.0, 2 ** attempt)
                    except Exception:
                        wait = min(30.0, 2 ** attempt)
                    time.sleep(wait)
                    continue
            if r.status_code >= 400:
                raise requests.HTTPError(f"Anthropic HTTP {r.status_code}: {r.text[:800]}", response=r)
            payload = r.json()
            if clean(payload.get("stop_reason")).lower() == "refusal":
                raise ValueError("Anthropic refusal")
            text = "".join(
                str(part.get("text", "")) for part in (payload.get("content") or [])
                if isinstance(part, dict) and part.get("type") == "text"
            ).strip()
            raw = extract_json(text)
            usage = payload.get("usage", {}) or {}
            raw["_model_returned"] = clean(payload.get("model")) or ANTHROPIC_MODEL
            raw["_input_tokens"] = usage.get("input_tokens", "")
            raw["_output_tokens"] = usage.get("output_tokens", "")
            return raw
        except Exception as exc:
            last = exc
            if attempt < 4:
                time.sleep(min(16.0, 2 ** attempt))
    raise last if last is not None else RuntimeError("Anthropic request failed")


def cache_file(row: pd.Series, condition: str, repeat: int) -> Path:
    payload = "|".join([
        CACHE_VERSION, PROMPT_VERSION, ANTHROPIC_MODEL, condition, str(repeat),
        clean(row.get("case_id")), hashlib.sha256(prompt_for(row, condition).encode("utf-8")).hexdigest(),
        EFFORT, str(MAX_TOKENS),
    ])
    key = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", ANTHROPIC_MODEL)
    d = CACHE_DIR / safe_model / condition.lower()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"r{repeat}_{clean(row.get('case_id'))}_{key}.json"


def normalize_llm(raw: dict, row: pd.Series, condition: str, repeat: int, status: str = "OK") -> dict:
    decision = clean(raw.get("decision")).upper()
    valid = decision in DECISIONS
    if not valid:
        decision = "REVIEW"
    return {
        "case_id": clean(row.get("case_id")),
        "identity_benchmark_id": clean(row.get("identity_benchmark_id")),
        "input_condition": condition,
        "llm_model": ANTHROPIC_MODEL,
        "repeat": repeat,
        "decision": decision,
        "classification_valid": bool(valid and status == "OK"),
        "reason": clean(raw.get("reason")),
        "confidence": clean(raw.get("confidence")).upper(),
        "call_status": status,
        "model_returned": clean(raw.get("_model_returned")),
        "input_tokens": raw.get("_input_tokens", ""),
        "output_tokens": raw.get("_output_tokens", ""),
    }


def run_one_llm(row_dict: dict, condition: str, repeat: int) -> dict:
    row = pd.Series(row_dict)
    p = cache_file(row, condition, repeat)
    if p.exists() and not FORCE:
        try:
            saved = json.loads(p.read_text(encoding="utf-8"))
            if saved.get("call_status") == "OK":
                return saved
        except Exception:
            pass
    try:
        raw = call_claude(row, condition)
        out = normalize_llm(raw, row, condition, repeat, "OK")
        # Cache only successful transport/model responses.
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return out
    except Exception as exc:
        return normalize_llm({}, row, condition, repeat, f"LLM_ERROR:{type(exc).__name__}:{str(exc)[:300]}")


def normalize_rule_scope(x) -> str:
    return bsi.normalize_isomer_scope(x) if hasattr(bsi, "normalize_isomer_scope") else "UNSPECIFIED_REVIEW"


def compare_rdkit(candidate_smiles: str, reference_smiles: str, isomer_scope: str = "") -> tuple[str, str]:
    ck = bsi.parent_key(candidate_smiles)
    cc = bsi.parent_connectivity_key(candidate_smiles)
    rk = bsi.parent_key(reference_smiles)
    rc = bsi.parent_connectivity_key(reference_smiles)
    scope = normalize_rule_scope(isomer_scope)
    if not rk or not rc:
        return "REVIEW", "INVALID_REFERENCE_STRUCTURE"
    if not ck or not cc:
        return "REVIEW", "MISSING_OR_INVALID_CANDIDATE_STRUCTURE"
    if ck == rk:
        return "MATCH", "PARENT_MATCH_EXACT_STEREO"
    if cc != rc:
        return "NO_MATCH", "PARENT_NO_MATCH_CONNECTIVITY"
    if scope == "ALL_STEREOISOMERS":
        return "MATCH", "PARENT_MATCH_ALL_STEREOISOMERS_RULE"
    if scope == "EXACT_STEREO_ONLY":
        return "NO_MATCH", "PARENT_NO_MATCH_EXACT_STEREO_RULE"
    return "REVIEW", "STEREO_SCOPE_UNSPECIFIED"


def truth_bool(x) -> bool:
    return clean(x).upper() == "MATCH"


def flag_bool(x) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return clean(x).lower() in {"1", "true", "yes", "y"}


def pred_bool(decision: str) -> Optional[bool]:
    d = clean(decision).upper()
    if d == "MATCH":
        return True
    if d == "NO_MATCH":
        return False
    return None


def performance(df: pd.DataFrame) -> dict:
    n = len(df)
    decided = df[df["decided"].astype(bool)].copy()
    coverage = len(decided) / n if n else np.nan
    if decided.empty:
        return {"n_total": n, "n_decided": 0, "coverage": coverage, "accuracy": np.nan, "precision": np.nan,
                "recall": np.nan, "specificity": np.nan, "false_safe_rate": np.nan, "false_positive_rate": np.nan}
    t = decided["truth_bool"].astype(bool)
    p = decided["pred_bool"].astype(bool)
    tp = int((t & p).sum()); tn = int((~t & ~p).sum()); fp = int((~t & p).sum()); fn = int((t & ~p).sum())
    return {
        "n_total": n, "n_decided": len(decided), "coverage": coverage,
        "accuracy": (tp + tn) / len(decided),
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "false_safe_rate": fn / (tp + fn) if tp + fn else np.nan,
        "false_positive_rate": fp / (tn + fp) if tn + fp else np.nan,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


def exact_mcnemar_by_repeat(a: pd.DataFrame, system_a: str, system_b: str) -> list[dict]:
    rows = []
    repeats = sorted(set(pd.to_numeric(a.get("repeat", pd.Series(dtype=int)), errors="coerce").dropna().astype(int)))
    for rep in repeats:
        x = a[(a["system"].eq(system_a)) & a["decided"] & (pd.to_numeric(a["repeat"], errors="coerce") == rep)][["case_id", "correct"]].copy()
        y = a[(a["system"].eq(system_b)) & a["decided"] & (pd.to_numeric(a["repeat"], errors="coerce") == rep)][["case_id", "correct"]].copy()
        m = x.merge(y, on="case_id", suffixes=("_a", "_b"))
        if m.empty:
            rows.append({"system_a": system_a, "system_b": system_b, "repeat": rep, "n_common": 0, "a_only_correct": 0, "b_only_correct": 0, "p_exact": np.nan})
            continue
        b = int((m["correct_a"].astype(bool) & ~m["correct_b"].astype(bool)).sum())
        c = int((~m["correct_a"].astype(bool) & m["correct_b"].astype(bool)).sum())
        n = b + c
        if n == 0:
            pval = 1.0
        else:
            k = min(b, c)
            prob = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
            pval = min(1.0, 2 * prob)
        rows.append({"system_a": system_a, "system_b": system_b, "repeat": rep, "n_common": len(m), "a_only_correct": b, "b_only_correct": c, "p_exact": pval})
    return rows


def main() -> None:
    load_dotenv()
    print("=" * 84)
    print("05 / STUDY 2 - CLAUDE vs RDKIT IDENTITY vs HYBRID")
    print("=" * 84)

    if not BENCHMARK_FILE.exists():
        raise FileNotFoundError("Run 04_build_identity_challenge.py first")
    bench = pd.read_csv(BENCHMARK_FILE).fillna("")
    if MAX_CASES:
        bench = bench.head(MAX_CASES).copy()
    if bench.empty:
        raise RuntimeError("Study 2 benchmark is empty")
    if "operational_cas_eligible" not in bench.columns:
        bench["operational_cas_eligible"] = bench["candidate_cas"].map(clean).ne("")
    if "controlled_structure_eligible" not in bench.columns:
        bench["controlled_structure_eligible"] = bench["candidate_source_smiles"].map(clean).ne("") & bench["reference_parent_smiles"].map(clean).ne("")
    bench["operational_cas_eligible"] = bench["operational_cas_eligible"].map(flag_bool)
    bench["controlled_structure_eligible"] = bench["controlled_structure_eligible"].map(flag_bool)
    op_bench = bench[bench["operational_cas_eligible"]].copy()
    ct_bench = bench[bench["controlled_structure_eligible"]].copy()
    if op_bench.empty or ct_bench.empty:
        raise RuntimeError("No eligible rows for one of the Study 2 settings")

    # Frozen rule metadata for isomer policy. Current CSVs may not contain the field.
    rules = pd.read_csv(RULE_SNAPSHOT_FILE).fillna("") if RULE_SNAPSHOT_FILE.exists() else pd.DataFrame()
    isomer_by_rule = {}
    if not rules.empty:
        if "isomer_scope" not in rules.columns:
            rules["isomer_scope"] = ""
        isomer_by_rule = dict(zip(rules["rule_id"].map(clean), rules["isomer_scope"].map(clean)))

    # ----- deterministic structure arms -----
    # OPERATIONAL: blank supplied SMILES so resolution really begins from CAS.
    inv = pd.DataFrame({
        "chemical_name": ["" for _ in range(len(op_bench))],  # name intentionally withheld from resolver input
        "cas": op_bench["candidate_cas"].tolist(),
        "smiles": ["" for _ in range(len(op_bench))],
    }, index=op_bench.index)
    resolved, pubchem_audit = bsi.enrich_inventory_with_pubchem(
        inv,
        cache_file=INTERMEDIATE / "05_pubchem_operational_identity_cache.csv",
        auto_fetch=True,
    )
    for col in ["operational_resolved_smiles", "operational_structure_status", "operational_structure_source", "operational_pubchem_cid", "operational_pubchem_record_url"]:
        bench[col] = ""
    bench.loc[op_bench.index, "operational_resolved_smiles"] = resolved["smiles"].tolist()
    bench.loc[op_bench.index, "operational_structure_status"] = resolved["structure_status"].tolist()
    bench.loc[op_bench.index, "operational_structure_source"] = resolved["structure_smiles_source"].tolist()
    bench.loc[op_bench.index, "operational_pubchem_cid"] = resolved["pubchem_cid"].tolist()
    bench.loc[op_bench.index, "operational_pubchem_record_url"] = resolved["pubchem_record_url"].tolist()

    rdkit_rows = []
    for _, r in bench.iterrows():
        scope = isomer_by_rule.get(clean(r.get("rule_id")), "")
        op_dec, op_reason = compare_rdkit(r.get("operational_resolved_smiles"), r.get("reference_parent_smiles"), scope)
        ct_dec, ct_reason = compare_rdkit(r.get("candidate_source_smiles"), r.get("reference_parent_smiles"), scope)
        rdkit_rows.append({
            "case_id": clean(r.get("case_id")),
            "rdkit_operational_decision": op_dec,
            "rdkit_operational_reason": op_reason,
            "rdkit_structure_decision": ct_dec,
            "rdkit_structure_reason": ct_reason,
        })
    rdkit = pd.DataFrame(rdkit_rows)
    bench = bench.merge(rdkit, on="case_id", how="left")

    # ----- Claude arms -----
    if not api_key():
        raise RuntimeError("ANTHROPIC_API_KEY is missing. Put it in .env or the PowerShell environment.")
    conditions = ["CAS_ONLY", "STRUCTURE_INFORMED"] + (["CAS_NAME"] if RUN_CAS_NAME else [])

    # One-case preflight catches schema/API problems before the full run.
    print(f"Claude model: {ANTHROPIC_MODEL} | repeats={N_REPEATS} | workers={WORKERS}")
    print("[PREFLIGHT] Claude structured-output identity check ...")
    try:
        _ = call_claude(bench.iloc[0], "CAS_ONLY")
        print("[PREFLIGHT] OK")
    except Exception as exc:
        raise RuntimeError(f"Claude preflight failed: {type(exc).__name__}: {exc}") from exc

    llm_rows = []
    for condition in conditions:
        condition_df = ct_bench if condition == "STRUCTURE_INFORMED" else op_bench
        records = condition_df.to_dict("records")
        for rep in range(1, N_REPEATS + 1):
            print(f"\n[{condition}] repeat {rep}/{N_REPEATS} | n={len(records)}")
            t0 = time.time(); done = 0
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                futs = {ex.submit(run_one_llm, row, condition, rep): row for row in records}
                for fut in as_completed(futs):
                    llm_rows.append(fut.result()); done += 1
                    if done % 10 == 0 or done == len(records):
                        elapsed = max(0.001, time.time() - t0)
                        print(f"  {done}/{len(records)} | {elapsed/done:.1f} s/case")

    llm = pd.DataFrame(llm_rows)
    llm.to_csv(INTERMEDIATE / "05_identity_claude_outputs.csv", index=False, encoding="utf-8-sig")
    pubchem_audit.to_csv(INTERMEDIATE / "05_pubchem_operational_identity_audit.csv", index=False, encoding="utf-8-sig")
    bench.to_csv(INTERMEDIATE / "05_identity_benchmark_enriched.csv", index=False, encoding="utf-8-sig")

    # ----- system-level predictions -----
    systems = []
    truth_map = dict(zip(bench["case_id"].map(clean), bench["reference_membership_bool"].map(lambda x: str(x).lower() in {"true", "1"})))
    bench_by_case = bench.set_index(bench["case_id"].map(clean), drop=False)

    # deterministic systems are repeated to align paired comparisons with Claude repeats
    for rep in range(1, N_REPEATS + 1):
        for _, r in bench.iterrows():
            cid = clean(r.get("case_id")); truth = truth_map[cid]
            det_specs = []
            if flag_bool(r.get("operational_cas_eligible")):
                det_specs.append(("RDKIT_CAS_LOOKUP", clean(r.get("rdkit_operational_decision")), "OPERATIONAL_CAS_ONLY", "PUBCHEM_EXACT_CAS_TO_RDKIT"))
            if flag_bool(r.get("controlled_structure_eligible")):
                det_specs.append(("RDKIT_STRUCTURE", clean(r.get("rdkit_structure_decision")), "CONTROLLED_STRUCTURE", "SOURCE_SMILES_TO_RDKIT"))
            for system, decision, setting, src in det_specs:
                pb = pred_bool(decision); decided = pb is not None
                systems.append({"case_id": cid, "repeat": rep, "system": system, "setting": setting,
                                "decision": decision or "REVIEW", "decision_source": src,
                                "truth_bool": truth, "pred_bool": pb, "decided": decided,
                                "correct": bool(pb == truth) if decided else False,
                                "false_safe": bool(decided and truth and pb is False),
                                "false_positive": bool(decided and (not truth) and pb is True)})

    for _, o in llm.iterrows():
        cid = clean(o.get("case_id")); rep = int(o.get("repeat", 1)); cond = clean(o.get("input_condition"))
        decision = clean(o.get("decision")) if bool(o.get("classification_valid")) else "REVIEW"
        pb = pred_bool(decision); decided = pb is not None; truth = truth_map[cid]
        system = {"CAS_ONLY": "CLAUDE_CAS_ONLY", "CAS_NAME": "CLAUDE_CAS_NAME", "STRUCTURE_INFORMED": "CLAUDE_STRUCTURE"}[cond]
        setting = "CONTROLLED_STRUCTURE" if cond == "STRUCTURE_INFORMED" else "OPERATIONAL_CAS_ONLY" if cond == "CAS_ONLY" else "PRACTICAL_CAS_NAME_SENSITIVITY"
        systems.append({"case_id": cid, "repeat": rep, "system": system, "setting": setting,
                        "decision": decision, "decision_source": f"CLAUDE_{cond}", "truth_bool": truth,
                        "pred_bool": pb, "decided": decided, "correct": bool(pb == truth) if decided else False,
                        "false_safe": bool(decided and truth and pb is False),
                        "false_positive": bool(decided and (not truth) and pb is True)})

    # Hybrid = deterministic-first; Claude is used only where RDKit cannot decide.
    llm_index = {(clean(r["case_id"]), int(r["repeat"]), clean(r["input_condition"])): r for _, r in llm.iterrows()}
    for rep in range(1, N_REPEATS + 1):
        for _, r in bench.iterrows():
            cid = clean(r.get("case_id")); truth = truth_map[cid]
            hybrid_specs = []
            if flag_bool(r.get("operational_cas_eligible")):
                hybrid_specs.append(("HYBRID_OPERATIONAL", "rdkit_operational_decision", "CAS_ONLY", "OPERATIONAL_CAS_ONLY"))
            if flag_bool(r.get("controlled_structure_eligible")):
                hybrid_specs.append(("HYBRID_STRUCTURE", "rdkit_structure_decision", "STRUCTURE_INFORMED", "CONTROLLED_STRUCTURE"))
            for system, rd_col, llm_cond, setting in hybrid_specs:
                rd = clean(r.get(rd_col)).upper(); source = "RDKIT_PRIMARY"
                if rd not in {"MATCH", "NO_MATCH"}:
                    lo = llm_index.get((cid, rep, llm_cond), {})
                    rd = clean(lo.get("decision")).upper() if bool(lo.get("classification_valid")) else "REVIEW"
                    source = f"CLAUDE_{llm_cond}_FALLBACK"
                pb = pred_bool(rd); decided = pb is not None
                systems.append({"case_id": cid, "repeat": rep, "system": system, "setting": setting,
                                "decision": rd or "REVIEW", "decision_source": source,
                                "truth_bool": truth, "pred_bool": pb, "decided": decided,
                                "correct": bool(pb == truth) if decided else False,
                                "false_safe": bool(decided and truth and pb is False),
                                "false_positive": bool(decided and (not truth) and pb is True)})

    pred = pd.DataFrame(systems)
    pred = pred.merge(bench[["case_id", "designation_id", "candidate_cas", "candidate_name", "reference_parent_name"]], on="case_id", how="left")
    pred.to_csv(INTERMEDIATE / "05_identity_system_predictions.csv", index=False, encoding="utf-8-sig")

    perf_rows = []
    for (system, setting, rep), g in pred.groupby(["system", "setting", "repeat"], sort=False):
        perf_rows.append({"system": system, "setting": setting, "repeat": rep, **performance(g)})
    perf = pd.DataFrame(perf_rows)
    perf.to_csv(INTERMEDIATE / "05_identity_performance_by_repeat.csv", index=False, encoding="utf-8-sig")

    mean = perf.groupby(["system", "setting"], as_index=False).agg(
        n_repeats=("repeat", "nunique"), n_total=("n_total", "max"),
        coverage_mean=("coverage", "mean"), coverage_sd=("coverage", "std"),
        accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        recall_mean=("recall", "mean"), recall_sd=("recall", "std"),
        specificity_mean=("specificity", "mean"), specificity_sd=("specificity", "std"),
        false_safe_rate_mean=("false_safe_rate", "mean"), false_safe_rate_sd=("false_safe_rate", "std"),
        false_positive_rate_mean=("false_positive_rate", "mean"), false_positive_rate_sd=("false_positive_rate", "std"),
    )
    mean.to_csv(INTERMEDIATE / "05_identity_performance_mean.csv", index=False, encoding="utf-8-sig")

    by_rule_rows = []
    for (system, setting, rep, did), g in pred.groupby(["system", "setting", "repeat", "designation_id"], sort=False):
        by_rule_rows.append({"system": system, "setting": setting, "repeat": rep, "designation_id": did, **performance(g)})
    pd.DataFrame(by_rule_rows).to_csv(INTERMEDIATE / "05_identity_performance_by_rule.csv", index=False, encoding="utf-8-sig")

    # Claude/RDKit conflicts on common decided rows.
    conflict_rows = []
    pairs = [
        ("CLAUDE_CAS_ONLY", "RDKIT_CAS_LOOKUP", "OPERATIONAL_CAS_ONLY"),
        ("CLAUDE_STRUCTURE", "RDKIT_STRUCTURE", "CONTROLLED_STRUCTURE"),
    ]
    for a, b, setting in pairs:
        xa = pred[(pred.system == a) & pred.decided][["case_id", "repeat", "decision", "correct"]]
        xb = pred[(pred.system == b) & pred.decided][["case_id", "repeat", "decision", "correct"]]
        m = xa.merge(xb, on=["case_id", "repeat"], suffixes=("_claude", "_rdkit"))
        if not m.empty:
            m["setting"] = setting; m["claude_system"] = a; m["rdkit_system"] = b
            m["conflict"] = m["decision_claude"] != m["decision_rdkit"]
            conflict_rows.append(m)
    conflicts = pd.concat(conflict_rows, ignore_index=True) if conflict_rows else pd.DataFrame()
    conflicts.to_csv(INTERMEDIATE / "05_identity_claude_rdkit_conflicts.csv", index=False, encoding="utf-8-sig")

    tests = []
    for a, b in [
        ("CLAUDE_CAS_ONLY", "RDKIT_CAS_LOOKUP"), ("CLAUDE_CAS_ONLY", "HYBRID_OPERATIONAL"),
        ("RDKIT_CAS_LOOKUP", "HYBRID_OPERATIONAL"), ("CLAUDE_STRUCTURE", "RDKIT_STRUCTURE"),
        ("CLAUDE_STRUCTURE", "HYBRID_STRUCTURE"), ("RDKIT_STRUCTURE", "HYBRID_STRUCTURE"),
    ]:
        tests.extend(exact_mcnemar_by_repeat(pred, a, b))
    pd.DataFrame(tests).to_csv(INTERMEDIATE / "05_identity_paired_mcnemar.csv", index=False, encoding="utf-8-sig")

    # Repeat consistency for Claude arms.
    cons_rows = []
    if N_REPEATS > 1:
        for cond, g in llm[llm["classification_valid"]].groupby("input_condition"):
            pivot = g.pivot_table(index="case_id", columns="repeat", values="decision", aggfunc="first")
            valid = pivot.dropna()
            if not valid.empty:
                row_cons = valid.nunique(axis=1).eq(1)
                cons_rows.append({"input_condition": cond, "n_cases_all_repeats_valid": len(valid), "decision_consistency": float(row_cons.mean())})
    pd.DataFrame(cons_rows).to_csv(INTERMEDIATE / "05_identity_repeat_consistency.csv", index=False, encoding="utf-8-sig")

    # Failure cases for interpretation.
    failures = pred[pred["decided"] & (~pred["correct"])].copy()
    failures.to_csv(INTERMEDIATE / "05_identity_failure_cases.csv", index=False, encoding="utf-8-sig")

    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "claude_model": ANTHROPIC_MODEL,
        "repeats": N_REPEATS,
        "workers": WORKERS,
        "effort": EFFORT,
        "prompt_version": PROMPT_VERSION,
        "operational_comparison": "Claude CAS-only vs exact-CAS PubChem lookup + RDKit parent normalization vs deterministic-first hybrid",
        "controlled_comparison": "Claude and RDKit receive same candidate/reference SMILES",
        "hybrid_policy": "RDKit-primary; Claude fallback only when RDKit returns REVIEW",
        "reference_type": "source-supported chemical identity operational reference; not expert legal gold",
    }
    (INTERMEDIATE / "05_identity_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[Mean performance]")
    cols = ["system", "setting", "coverage_mean", "accuracy_mean", "recall_mean", "specificity_mean", "false_safe_rate_mean"]
    print(mean[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved to: intermediate/")
    print("Next: python 06_make_identity_results.py")


if __name__ == "__main__":
    main()
