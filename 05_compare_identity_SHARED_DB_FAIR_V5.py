# -*- coding: utf-8 -*-
"""Study 2 / Step 05 FAIR V5: equal-information identity comparison.

Primary systems
---------------
1) LLM + shared PubChem structure
2) Salt-aware RDKit + the SAME PubChem structure
3) Hybrid: salt-aware RDKit first, SAME LLM result only when RDKit returns REVIEW

Methodological guardrails
-------------------------
* Candidate names and CAS are used only to perform the shared exact-CAS PubChem lookup;
  neither model receives lexical identity cues at decision time.
* LLM and RDKit receive the same candidate structure, reference structure, task scope,
  and stereochemistry policy.
* The deterministic salt rules are generic chemistry-class rules frozen in
  cheminformatics_identity_V5_FAIR.py; no compound-specific exceptions are allowed.
* Current 60/134-case sets are development/exploratory after V5 refinement.
* Confirmatory claims require an independently frozen holdout and a valid 04D freeze.
* Primary inferential comparisons are case-level; repeated LLM calls are collapsed to
  a consensus so deterministic predictions are not pseudo-replicated.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
INTER = ROOT / "intermediate"
INTER.mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_FILE = ROOT / "05_compare_identity_SHARED_DB_METHODSAFE_V4.py"
ENGINE_FILE = ROOT / "cheminformatics_identity_V5_FAIR.py"
FREEZE_FILE = INTER / "04D_fair_v5_protocol_freeze.json"
DEV_BENCHMARK_FILE = INTER / "04_identity_challenge_benchmark.csv"
RULE_SNAPSHOT_FILE = INTER / "04_identity_challenge_rules.csv"
FINAL_HOLDOUT_ENV = os.getenv("IDENTITY_V5_FINAL_HOLDOUT_FILE", "").strip()
EXECUTE_CLAUDE = os.getenv("IDENTITY_EXECUTE_CLAUDE", "0").strip().lower() in {"1", "true", "yes", "on"}

PROMPT_POLICY_VERSION = "study2-fair-v5.2-equal-structure-policy-20260930"
SYSTEMS = ["CLAUDE_DB", "RDKIT_SALT_AWARE_V5", "HYBRID_SALT_AWARE_V5"]

STRUCTURAL_POLICY = (
    "The scope is the reference parent chemical and its counterion salts. Split the candidate into all "
    "disconnected components. Standardize ordinary charge/protonation and tautomer representation before "
    "comparing parent identity. One or more stoichiometric copies of the parent fragment may occur. Every "
    "remaining component must be a chemically compatible counterion or a common solvate; otherwise REVIEW. "
    "Counterions may be drawn as ions, as neutral acids (for example hydrogen chloride written as Cl), or as "
    "neutral metal atoms. "
    "Common solvates are water, methanol, ethanol, isopropanol, acetone, acetonitrile, carbon dioxide, "
    "dimethyl sulfoxide and 1,4-dioxane. "
    "Covalent derivatives or close analogs that do not contain the same parent connectivity are NO_MATCH. "
    "If the reference parent SMILES specifies no stereochemistry, every stereoisomer of the parent is in scope. "
    "Otherwise stereo-only differences follow the supplied isomer_scope; when that scope is unspecified, REVIEW."
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path.name}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_dotenv_early() -> None:
    """Load .env BEFORE the base module reads IDENTITY_ANTHROPIC_* settings at import.

    Existing environment variables win, matching base.load_dotenv().
    """
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


_load_dotenv_early()
if not BASE_FILE.exists():
    raise FileNotFoundError(BASE_FILE)
if not ENGINE_FILE.exists():
    raise FileNotFoundError(ENGINE_FILE)
base = _load_module(BASE_FILE, "study2_v4_base")
engine = _load_module(ENGINE_FILE, "study2_v5_engine")
base.CACHE_DIR = INTER / "identity_llm_cache_v5"
base.CACHE_DIR.mkdir(parents=True, exist_ok=True)
base.DB_PROMPT_VERSION = PROMPT_POLICY_VERSION


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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_scope(x: str) -> str:
    return engine.normalize_isomer_scope(x)


def prompt_for_v5(row: pd.Series, condition: str) -> str:
    if condition != "DB_INFORMED":
        raise ValueError("FAIR V5 uses DB_INFORMED only")
    payload = {
        "case_id": clean(row.get("case_id")),
        "candidate_pubchem_smiles": clean(row.get("operational_resolved_smiles")),
        "candidate_structure_status": clean(row.get("operational_structure_status")),
        "reference_parent_smiles": clean(row.get("reference_parent_smiles")),
        "isomer_scope": normalize_scope(clean(row.get("isomer_scope"))),
        "scope_definition": "The regulated identity scope is the reference parent chemical and its counterion salts.",
        "decision_policy": STRUCTURAL_POLICY,
    }
    return (
        "INPUT_CONDITION=DB_INFORMED\n"
        "Chemical-identity benchmark only; use only the supplied fields.\n"
        f"INPUT={json.dumps(payload, ensure_ascii=False)}\n"
        "Return decision=MATCH, NO_MATCH, or REVIEW."
    )


base.prompt_for = prompt_for_v5


def required_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")


def load_benchmark() -> tuple[pd.DataFrame, str, Path]:
    if FINAL_HOLDOUT_ENV:
        p = Path(FINAL_HOLDOUT_ENV).expanduser()
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"Final holdout not found: {p}")
        df = pd.read_csv(p, dtype=str).fillna("")
        required_columns(df, ["case_id", "rule_id", "candidate_cas", "reference_parent_smiles", "reference_membership", "reference_source", "holdout_frozen_before_v5_evaluation"], "FINAL HOLDOUT")
        if not df["holdout_frozen_before_v5_evaluation"].map(as_bool).all():
            raise RuntimeError("Every final-holdout row must have holdout_frozen_before_v5_evaluation=true")
        if not df["reference_source"].map(clean).ne("").all():
            raise RuntimeError("Every final-holdout row must have a non-empty independent reference_source")
        return df, "FINAL_INDEPENDENT_HOLDOUT_CONFIRMATORY", p
    if not DEV_BENCHMARK_FILE.exists():
        raise FileNotFoundError("Run 04C_prepare_dual_benchmark_METHODSAFE.py first")
    return pd.read_csv(DEV_BENCHMARK_FILE, dtype=str).fillna(""), "DEVELOPMENT_REANALYSIS_NOT_CONFIRMATORY", DEV_BENCHMARK_FILE


def validate_holdout_independence(holdout: pd.DataFrame) -> dict:
    if not DEV_BENCHMARK_FILE.exists():
        raise RuntimeError("Development benchmark is required for holdout-overlap audit")
    dev = pd.read_csv(DEV_BENCHMARK_FILE, dtype=str).fillna("")
    dev_cas = {clean(x) for x in dev.get("candidate_cas", pd.Series(dtype=str)) if clean(x)}
    hold_cas = {clean(x) for x in holdout["candidate_cas"] if clean(x)}
    cas_overlap = sorted(dev_cas & hold_cas)
    dev_pairs = set(zip(dev.get("rule_id", pd.Series(dtype=str)).map(clean), dev.get("candidate_cas", pd.Series(dtype=str)).map(clean)))
    hold_pairs = set(zip(holdout["rule_id"].map(clean), holdout["candidate_cas"].map(clean)))
    pair_overlap = sorted(dev_pairs & hold_pairs)
    case_overlap = sorted(set(dev.get("case_id", pd.Series(dtype=str)).map(clean)) & set(holdout["case_id"].map(clean)))
    audit = {"candidate_cas_overlap_n": len(cas_overlap), "rule_cas_pair_overlap_n": len(pair_overlap), "case_id_overlap_n": len(case_overlap), "candidate_cas_overlap": cas_overlap, "rule_cas_pair_overlap": pair_overlap, "case_id_overlap": case_overlap}
    if cas_overlap or pair_overlap or case_overlap:
        raise RuntimeError(f"Final holdout is not independent of development data: {audit}")
    return audit


def llm_protocol_config() -> dict:
    """Everything that defines the LLM arm besides this file and the engine.

    Recorded by Step 04D and re-checked before a final-holdout run, so a change
    to the base module (system prompt, API request, PubChem lookup) or to the
    model settings after the freeze is detected.
    """
    return {
        "base_file": BASE_FILE.name,
        "base_sha256": sha256_file(BASE_FILE),
        "anthropic_model": base.ANTHROPIC_MODEL,
        "anthropic_effort": base.EFFORT,
        "anthropic_max_tokens": int(base.MAX_TOKENS),
        "llm_repeats": int(base.N_REPEATS),
        "system_prompt_sha256": hashlib.sha256(base.SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "prompt_policy_version": PROMPT_POLICY_VERSION,
        "prompt_policy_sha256": hashlib.sha256(STRUCTURAL_POLICY.encode("utf-8")).hexdigest(),
    }


def verify_protocol_freeze() -> dict:
    if not FREEZE_FILE.exists():
        raise RuntimeError("Final holdout execution requires 04D_fair_v5_protocol_freeze.json. Run Step 04D before opening/evaluating the holdout.")
    freeze = json.loads(FREEZE_FILE.read_text(encoding="utf-8"))
    checks = {
        "engine_sha256_match": clean(freeze.get("engine_sha256")) == sha256_file(ENGINE_FILE),
        "step05_sha256_match": clean(freeze.get("step05_sha256")) == sha256_file(Path(__file__).resolve()),
        "engine_version_match": clean(freeze.get("engine_version")) == clean(engine.ENGINE_VERSION),
        "policy_version_match": clean(freeze.get("policy_version")) == clean(engine.POLICY_VERSION),
        "llm_protocol_match": freeze.get("llm_protocol") == llm_protocol_config(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"FAIR V5 protocol changed after freeze: {checks}. Re-freezing after viewing the holdout would invalidate confirmatory status.")
    freeze["verification_checks"] = checks
    return freeze


def consensus_decision(values: list[str]) -> str:
    ds = [clean(v).upper() if clean(v).upper() in {"MATCH", "NO_MATCH", "REVIEW"} else "REVIEW" for v in values]
    if not ds:
        return "REVIEW"
    counts = {d: ds.count(d) for d in {"MATCH", "NO_MATCH", "REVIEW"}}
    winner, n = max(counts.items(), key=lambda kv: kv[1])
    return winner if n > len(ds) / 2 else "REVIEW"


def pred_bool(decision: str) -> Optional[bool]:
    d = clean(decision).upper()
    if d == "MATCH": return True
    if d == "NO_MATCH": return False
    return None


def exact_mcnemar(a: pd.DataFrame, system_a: str, system_b: str) -> dict:
    x = a[a["system"].eq(system_a)][["case_id", "correct"]].copy()
    y = a[a["system"].eq(system_b)][["case_id", "correct"]].copy()
    m = x.merge(y, on="case_id", suffixes=("_a", "_b"))
    b = int((m["correct_a"].map(as_bool) & ~m["correct_b"].map(as_bool)).sum())
    c = int((~m["correct_a"].map(as_bool) & m["correct_b"].map(as_bool)).sum())
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2.0 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2 ** n))
    return {"system_a": system_a, "system_b": system_b, "n_common": len(m), "a_only_correct": b, "b_only_correct": c, "p_exact": p}


def performance(g: pd.DataFrame) -> dict:
    n = len(g); d = g[g["decided"].map(as_bool)].copy(); coverage = len(d) / n if n else np.nan
    overall = float((g["decided"].map(as_bool) & g["correct"].map(as_bool)).mean()) if n else np.nan
    if d.empty:
        return {"n_total": n, "n_decided": 0, "coverage": coverage, "accuracy": np.nan, "precision": np.nan, "recall": np.nan, "specificity": np.nan, "balanced_accuracy": np.nan, "false_safe_rate": np.nan, "false_positive_rate": np.nan, "overall_correct_resolution_rate": overall, "TP":0,"TN":0,"FP":0,"FN":0}
    t, p = d["truth_bool"].map(as_bool), d["pred_bool"].map(as_bool)
    tp = int((t & p).sum()); tn = int((~t & ~p).sum()); fp = int((~t & p).sum()); fn = int((t & ~p).sum())
    rec = tp/(tp+fn) if tp+fn else np.nan; spec = tn/(tn+fp) if tn+fp else np.nan
    ba = np.nanmean([rec, spec]) if np.isfinite(rec) or np.isfinite(spec) else np.nan
    return {"n_total": n, "n_decided": len(d), "coverage": coverage, "accuracy": (tp+tn)/len(d), "precision": tp/(tp+fp) if tp+fp else np.nan, "recall": rec, "specificity": spec, "balanced_accuracy": ba, "false_safe_rate": fn/(tp+fn) if tp+fn else np.nan, "false_positive_rate": fp/(tn+fp) if tn+fp else np.nan, "overall_correct_resolution_rate": overall, "TP":tp,"TN":tn,"FP":fp,"FN":fn}


def main() -> None:
    base.load_dotenv()
    bench, analysis_status, benchmark_path = load_benchmark()
    required_columns(bench, ["case_id", "rule_id", "candidate_cas", "reference_parent_smiles", "reference_membership"], "benchmark")
    bench = bench.copy()
    if bench["case_id"].map(clean).duplicated().any(): raise RuntimeError("case_id must be unique")
    if bench[["rule_id", "candidate_cas"]].astype(str).apply(lambda s: s.str.strip()).duplicated().any(): raise RuntimeError("rule_id/candidate_cas pairs must be unique")
    bench["reference_membership_bool"] = bench["reference_membership"].map(lambda x: clean(x).upper() == "MATCH")
    if "operational_cas_eligible" not in bench.columns: bench["operational_cas_eligible"] = bench["candidate_cas"].map(clean).ne("")
    bench["operational_cas_eligible"] = bench["operational_cas_eligible"].map(as_bool)

    rules = pd.read_csv(RULE_SNAPSHOT_FILE, dtype=str).fillna("") if RULE_SNAPSHOT_FILE.exists() else pd.DataFrame(); isomer_map = {}
    if len(rules) and "rule_id" in rules.columns:
        if "isomer_scope" not in rules.columns: rules["isomer_scope"] = ""
        isomer_map = dict(zip(rules["rule_id"].map(clean), rules["isomer_scope"].map(clean)))
    if "isomer_scope" not in bench.columns: bench["isomer_scope"] = bench["rule_id"].map(lambda x: isomer_map.get(clean(x), ""))
    bench["isomer_scope"] = bench["isomer_scope"].map(normalize_scope)

    holdout_audit = {}; freeze = {}
    if analysis_status.startswith("FINAL_"):
        holdout_audit = validate_holdout_independence(bench); freeze = verify_protocol_freeze()
    self_tests = engine.generic_self_tests()
    if not all(bool(r.get("pass")) for r in self_tests): raise RuntimeError("FAIR V5 generic engine self-tests failed")

    op = bench[bench["operational_cas_eligible"]].copy()
    if op.empty: raise RuntimeError("No CAS-eligible cases")
    inv = pd.DataFrame({"chemical_name": [""] * len(op), "cas": op["candidate_cas"].tolist(), "smiles": [""] * len(op)}, index=op.index)
    resolved, audit = base.enrich_inventory_with_pubchem(inv, INTER / "05_v5_pubchem_shared_cache.csv")
    if (audit["structure_status"] == "PUBCHEM_REQUEST_ERROR").any(): raise RuntimeError("Transient PubChem errors are not allowed in a completed FAIR V5 run")

    for col in ["operational_resolved_smiles", "operational_structure_status", "operational_pubchem_cid", "operational_pubchem_record_url"]: bench[col] = ""
    bench["operational_structure_lookup_elapsed_sec"] = np.nan
    bench.loc[op.index, "operational_resolved_smiles"] = resolved["smiles"].tolist(); bench.loc[op.index, "operational_structure_status"] = resolved["structure_status"].tolist(); bench.loc[op.index, "operational_pubchem_cid"] = resolved["pubchem_cid"].tolist(); bench.loc[op.index, "operational_pubchem_record_url"] = resolved["pubchem_record_url"].tolist(); bench.loc[op.index, "operational_structure_lookup_elapsed_sec"] = resolved["structure_lookup_elapsed_sec"].to_numpy(dtype=float)

    rd_rows = []
    for _, r in bench[bench["operational_cas_eligible"]].iterrows():
        t0 = time.perf_counter(); decision, reason = engine.compare_salt_parent_v5(clean(r.get("operational_resolved_smiles")), clean(r.get("reference_parent_smiles")), clean(r.get("isomer_scope"))); compare_sec = float(time.perf_counter() - t0)
        lookup = pd.to_numeric(pd.Series([r.get("operational_structure_lookup_elapsed_sec")]), errors="coerce").iloc[0]; total = float(lookup + compare_sec) if pd.notna(lookup) else np.nan
        rd_rows.append({"case_id": clean(r.get("case_id")), "rdkit_v5_decision": decision, "rdkit_v5_reason": reason, "rdkit_v5_compare_elapsed_sec": compare_sec, "rdkit_v5_total_elapsed_sec": total})
    bench = bench.merge(pd.DataFrame(rd_rows), on="case_id", how="left")
    audit.to_csv(INTER / "05_v5_pubchem_shared_audit.csv", index=False, encoding="utf-8-sig"); bench.to_csv(INTER / "05_v5_identity_benchmark_enriched.csv", index=False, encoding="utf-8-sig")
    preflight_cols = [c for c in ["case_id", "candidate_cas", "reference_parent_name", "candidate_name", "challenge_class", "difficulty", "operational_structure_status", "operational_resolved_smiles", "reference_parent_smiles", "isomer_scope", "rdkit_v5_decision", "rdkit_v5_reason"] if c in bench.columns]
    bench[preflight_cols].to_csv(INTER / "05_v5_rdkit_preflight_predictions.csv", index=False, encoding="utf-8-sig")

    manifests = []
    for _, r in bench[bench["operational_cas_eligible"]].iterrows():
        text = prompt_for_v5(r, "DB_INFORMED"); manifests.append({"case_id": clean(r.get("case_id")), "candidate_structure_status": clean(r.get("operational_structure_status")), "paid_api_eligible": clean(r.get("operational_structure_status")) == "PUBCHEM_EXACT_CAS_VERIFIED", "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "prompt_text": text})
    manifest = pd.DataFrame(manifests); manifest.to_csv(INTER / "05_v5_claude_prompt_manifest.csv", index=False, encoding="utf-8-sig")

    run_meta = {"created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "analysis_status": analysis_status, "benchmark_file": str(benchmark_path), "benchmark_sha256": sha256_file(benchmark_path), "engine_version": engine.ENGINE_VERSION, "engine_policy_version": engine.POLICY_VERSION, "engine_sha256": sha256_file(ENGINE_FILE), "step05_sha256": sha256_file(Path(__file__).resolve()), "prompt_policy_version": PROMPT_POLICY_VERSION, "prompt_policy_sha256": hashlib.sha256(STRUCTURAL_POLICY.encode("utf-8")).hexdigest(), "same_candidate_structure_all_systems": True, "same_reference_structure_all_systems": True, "same_isomer_scope_all_systems": True, "candidate_name_withheld_from_decision_models": True, "candidate_cas_withheld_after_shared_lookup": True, "source_smiles_not_used_for_operational_decision": True, "compound_specific_rdkit_exceptions": False, "llm_repeats": int(base.N_REPEATS), "llm_protocol": llm_protocol_config(), "current_60_134_status": "development/exploratory only" if not analysis_status.startswith("FINAL_") else "not used for confirmatory scoring", "holdout_overlap_audit": holdout_audit, "protocol_freeze": freeze, "generic_engine_self_tests": self_tests}
    (INTER / "05_v5_run_metadata.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    n_paid = int(manifest["paid_api_eligible"].sum())
    print("=" * 96); print("STUDY 2 FAIR V5"); print("=" * 96); print(f"Analysis status: {analysis_status}"); print(f"Cases: {len(manifest)} | exact-CAS resolved for paid arm: {n_paid}"); print(f"Planned Claude calls: {n_paid * base.N_REPEATS}"); print(f"Paid Claude execution enabled: {EXECUTE_CLAUDE}")
    if not EXECUTE_CLAUDE:
        print("[STOP BEFORE PAID API] Review 05_v5_rdkit_preflight_predictions.csv and 05_v5_claude_prompt_manifest.csv."); return

    base.validate_api_key_for_http_header(); op_rows = bench[bench["operational_cas_eligible"]].copy(); paid = op_rows[op_rows["operational_structure_status"].eq("PUBCHEM_EXACT_CAS_VERIFIED")].copy(); unresolved = op_rows[~op_rows["operational_structure_status"].eq("PUBCHEM_EXACT_CAS_VERIFIED")].copy()
    if len(paid):
        # One real call before the batch: a bad model id or an unsupported request
        # parameter must stop the run instead of turning every case into REVIEW.
        # The result is cached, so the batch below reuses it.
        probe = base.run_one_llm(paid.iloc[0].to_dict(), "DB_INFORMED", 1)
        if clean(probe.get("call_status")) != "OK":
            raise RuntimeError(f"Claude preflight call failed; no batch was run. Detail: {clean(probe.get('call_status'))}")
        print(f"[CLAUDE PREFLIGHT OK] model={base.ANTHROPIC_MODEL} effort={base.EFFORT} max_tokens={base.MAX_TOKENS}")
    llm_rows = []
    for rep in range(1, base.N_REPEATS + 1):
        for _, r in unresolved.iterrows(): llm_rows.append({"case_id": clean(r.get("case_id")), "repeat": rep, "decision": "REVIEW", "classification_valid": False, "reason": "SHARED_DB_UNRESOLVED_NO_MODEL_CALL", "call_status": "DB_UNRESOLVED_NO_CALL", "elapsed_sec": 0.0})
        for row in paid.to_dict("records"): llm_rows.append(base.run_one_llm(row, "DB_INFORMED", rep))
    llm = pd.DataFrame(llm_rows); llm.to_csv(INTER / "05_v5_identity_claude_outputs.csv", index=False, encoding="utf-8-sig")
    # A transport/API failure is not a model decision. Scoring it as REVIEW would
    # silently lower CLAUDE_DB and HYBRID coverage, so stop instead. Successful
    # calls are cached; rerunning retries only the failed ones.
    failed = llm[~llm["call_status"].map(clean).isin({"OK", "DB_UNRESOLVED_NO_CALL"})]
    if len(failed):
        raise RuntimeError(
            f"{len(failed)} of {len(llm)} Claude calls failed; results were NOT scored. "
            "See call_status in 05_v5_identity_claude_outputs.csv and rerun Step 05 to retry the failed calls. "
            f"First error: {clean(failed['call_status'].iloc[0])[:300]}"
        )

    truth_map = dict(zip(bench["case_id"].map(clean), bench["reference_membership_bool"].map(as_bool))); br = bench.set_index(bench["case_id"].map(clean), drop=False); consensus = {}; consistency_rows = []
    for cid, g in llm.groupby(llm["case_id"].map(clean)):
        valid_series = g["classification_valid"] if "classification_valid" in g.columns else pd.Series([True] * len(g), index=g.index)
        ds = [clean(x).upper() if as_bool(v) else "REVIEW" for x, v in zip(g["decision"], valid_series)]; con = consensus_decision(ds); consensus[cid] = con; consistency_rows.append({"case_id": cid, "n_repeats": len(ds), "decisions": ";".join(ds), "consensus_decision": con, "all_repeats_agree": len(set(ds)) == 1})
    pd.DataFrame(consistency_rows).to_csv(INTER / "05_v5_llm_repeat_consistency.csv", index=False, encoding="utf-8-sig")

    systems = []
    for cid in br.index:
        r = br.loc[cid]
        if not as_bool(r.get("operational_cas_eligible")): continue
        truth = truth_map[cid]; llm_dec = consensus.get(cid, "REVIEW"); rd_dec = clean(r.get("rdkit_v5_decision")).upper() or "REVIEW"; hyb_dec = rd_dec if rd_dec in {"MATCH", "NO_MATCH"} else llm_dec
        for system, dec, source in [("CLAUDE_DB", llm_dec, "LLM_CONSENSUS_FROM_SHARED_STRUCTURE"), ("RDKIT_SALT_AWARE_V5", rd_dec, "FROZEN_GENERIC_SALT_AWARE_ENGINE"), ("HYBRID_SALT_AWARE_V5", hyb_dec, "RDKIT_PRIMARY_ELSE_LLM_CONSENSUS")]:
            pb = pred_bool(dec); decided = pb is not None; row = {"case_id": cid, "system": system, "decision": dec, "decision_source": source, "truth_bool": truth, "pred_bool": pb, "decided": decided, "correct": bool(pb == truth) if decided else False, "false_safe": bool(decided and truth and pb is False), "false_positive": bool(decided and (not truth) and pb is True), "overall_correct_resolution": bool(decided and pb == truth), "analysis_status": analysis_status}
            for c in ["rule_id", "designation_id", "candidate_cas", "candidate_name", "reference_parent_name", "challenge_class", "difficulty", "in_primary_source_curated", "in_secondary_structure_anchored", "benchmark_sets", "isomer_scope"]:
                if c in r.index: row[c] = r.get(c)
            systems.append(row)
    pred = pd.DataFrame(systems); pred.to_csv(INTER / "05_v5_identity_system_predictions_caselevel.csv", index=False, encoding="utf-8-sig")

    perf_rows = []
    for s, g in pred.groupby("system", sort=False): perf_rows.append({"system": s, **performance(g)})
    pd.DataFrame(perf_rows).to_csv(INTER / "05_v5_performance_caselevel.csv", index=False, encoding="utf-8-sig")
    tests = [exact_mcnemar(pred, a, b) for a, b in [("CLAUDE_DB", "RDKIT_SALT_AWARE_V5"), ("CLAUDE_DB", "HYBRID_SALT_AWARE_V5"), ("RDKIT_SALT_AWARE_V5", "HYBRID_SALT_AWARE_V5")]]; tdf = pd.DataFrame(tests); tdf["confirmatory_interpretation_allowed"] = analysis_status.startswith("FINAL_"); tdf.to_csv(INTER / "05_v5_paired_mcnemar_caselevel.csv", index=False, encoding="utf-8-sig")
    pred[pred["decided"].map(as_bool) & ~pred["correct"].map(as_bool)].to_csv(INTER / "05_v5_failure_cases.csv", index=False, encoding="utf-8-sig"); pred[~pred["decided"].map(as_bool)].to_csv(INTER / "05_v5_review_cases.csv", index=False, encoding="utf-8-sig")
    print("\n[FAIR V5 case-level performance]"); print(pd.DataFrame(perf_rows).to_string(index=False))
    if not analysis_status.startswith("FINAL_"): print("\n[IMPORTANT] Current benchmark results are development/exploratory after method refinement; do not present McNemar p-values as confirmatory.")
    print("Next: python 06_make_identity_results_FAIR_V5.py")


if __name__ == "__main__":
    main()
