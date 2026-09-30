# -*- coding: utf-8 -*-
"""Study 2 / Step 05 FINAL: shared-DB comparison with paid-API safety gate.

Final systems only:
  1) LLM + DB
  2) DB + RDKit
  3) Hybrid (RDKit first; reuses the same LLM+DB result only on REVIEW)

Closed-LLM, CAS+name, and separate controlled-structure Claude arms are removed
from the final experiment. All systems share the same exact-CAS PubChem
resolution. By default this script performs only PubChem/RDKit preprocessing and
writes the exact Claude prompt manifest; it makes ZERO Claude API calls. Set
IDENTITY_EXECUTE_CLAUDE=1 only after Step 04 readiness is true.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from urllib.parse import quote
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import cheminformatics_identity as ci42
except Exception as exc:
    raise ImportError(
        "Could not import the frozen cheminformatics_identity.py comparator from the repository root."
    ) from exc

try:
    import cheminformatics_identity_V4_3_SALT_AWARE as ci43
except Exception as exc:
    raise ImportError(
        "Could not import cheminformatics_identity_V4_3_SALT_AWARE.py from the repository root."
    ) from exc

# Backward-compatible alias: original V3 helper calls remain frozen-V4.2 logic.
ci = ci42

try:
    from rdkit import Chem
except Exception as exc:
    raise ImportError("RDKit is required for Study 2 deterministic identity resolution.") from exc

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
RUN_LLM_DB = os.getenv("IDENTITY_RUN_LLM_DB", "1").strip().lower() not in {"0", "false", "no", "off"}
MAX_CASES = max(0, int(os.getenv("IDENTITY_MAX_CASES", "0")))
# Runtime benchmark controls.
# - IDENTITY_FORCE_LLM=1: bypass Claude cache and make fresh API calls.
# - IDENTITY_PUBCHEM_COLD_START=1: ignore the pre-existing PubChem cache at the
#   start of this run, while still caching repeated CAS values within the run.
PUBCHEM_COLD_START = os.getenv("IDENTITY_PUBCHEM_COLD_START", "0").strip().lower() in {"1", "true", "yes", "on"}
RUNTIME_PER_CASE_FILE = INTERMEDIATE / "05_identity_runtime_per_case.csv"
RUNTIME_SUMMARY_FILE = INTERMEDIATE / "05_identity_runtime_summary.csv"
RUNTIME_BATCH_FILE = INTERMEDIATE / "05_identity_runtime_batch_wall.csv"
RUNTIME_PPT_FILE = INTERMEDIATE / "05_identity_runtime_ppt_operational.csv"
PROMPT_VERSION = "study2-chemical-identity-v1-20260911"
DB_PROMPT_VERSION = "study2-shared-pubchem-v2-20260929"
STRUCTURE_PROMPT_VERSION = "study2-structure-blinded-v2-20260911"
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
You cannot independently browse or call tools. In DB-informed conditions, the prompt contains the exact external PubChem information made available to the system; use only those supplied fields.
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


PUBCHEM_API_ROOT = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_RETRIES = max(3, int(os.getenv("IDENTITY_PUBCHEM_RETRIES", "8")))
PUBCHEM_MIN_INTERVAL_SEC = max(0.0, float(os.getenv("IDENTITY_PUBCHEM_MIN_INTERVAL_SEC", "0.35")))
_LAST_PUBCHEM_REQUEST_AT = 0.0
PUBCHEM_CACHE_COLUMNS = [
    "cas", "structure_status", "pubchem_cid", "pubchem_title",
    "pubchem_isomeric_smiles", "pubchem_connectivity_smiles", "pubchem_inchikey",
    "pubchem_cas_verified", "pubchem_retrieval_date", "pubchem_record_url",
    "pubchem_query_url", "pubchem_error",
]


def normalize_cas(x) -> str:
    return re.sub(r"\s+", "", clean(x))


def valid_cas_checksum(cas: str) -> bool:
    value = normalize_cas(cas)
    if not re.fullmatch(r"\d{2,7}-\d{2}-\d", value):
        return False
    body, check = value.rsplit("-", 1)
    digits = body.replace("-", "")
    checksum = sum((i + 1) * int(d) for i, d in enumerate(reversed(digits))) % 10
    return checksum == int(check)


def _pubchem_json(url: str, timeout: int = 30, retries: int = PUBCHEM_RETRIES) -> dict:
    """Rate-safe PubChem request with exponential backoff.

    A transient 429/5xx/network failure must never silently become an
    unresolved benchmark case.  The caller will block the preflight if all
    retries are exhausted.
    """
    global _LAST_PUBCHEM_REQUEST_AT
    last = None
    for attempt in range(max(1, retries)):
        try:
            elapsed = time.monotonic() - _LAST_PUBCHEM_REQUEST_AT
            wait = PUBCHEM_MIN_INTERVAL_SEC - elapsed
            if wait > 0:
                time.sleep(wait)
            _LAST_PUBCHEM_REQUEST_AT = time.monotonic()

            r = requests.get(url, timeout=timeout, headers={
                "User-Agent": "chemical-regulatory-study2-presentation/1.1",
                "Accept": "application/json",
            })
            if r.status_code == 404:
                raise LookupError("PUBCHEM_NOT_FOUND")
            if r.status_code == 429 or 500 <= r.status_code < 600:
                if attempt + 1 < retries:
                    retry_after = r.headers.get("Retry-After")
                    try:
                        retry_after = float(retry_after) if retry_after else 0.0
                    except Exception:
                        retry_after = 0.0
                    delay = max(retry_after, min(30.0, 1.0 * (2 ** attempt)))
                    print(f"[PUBCHEM RETRY {attempt+1}/{retries}] HTTP {r.status_code}; sleep {delay:.1f}s")
                    time.sleep(delay)
                    continue
            r.raise_for_status()
            return r.json()
        except LookupError:
            raise
        except Exception as exc:
            last = exc
            if attempt + 1 < retries:
                delay = min(30.0, 1.0 * (2 ** attempt))
                print(f"[PUBCHEM RETRY {attempt+1}/{retries}] {type(exc).__name__}; sleep {delay:.1f}s")
                time.sleep(delay)

    raise RuntimeError(f"{type(last).__name__}: {last}") from last


def fetch_pubchem_structure_by_cas(cas: str) -> dict:
    cas_norm = normalize_cas(cas)
    rec = {c: "" for c in PUBCHEM_CACHE_COLUMNS}
    rec["cas"] = cas_norm
    rec["pubchem_retrieval_date"] = datetime.now(timezone.utc).date().isoformat()
    if not valid_cas_checksum(cas_norm):
        rec["structure_status"] = "PUBCHEM_INVALID_CAS"
        rec["pubchem_error"] = "CAS format/checksum invalid"
        return rec

    encoded = quote(cas_norm, safe="")
    url = (
        f"{PUBCHEM_API_ROOT}/compound/name/{encoded}/property/"
        "Title,CanonicalSMILES,IsomericSMILES,InChIKey/JSON"
    )
    rec["pubchem_query_url"] = url
    try:
        payload = _pubchem_json(url)
        props = payload.get("PropertyTable", {}).get("Properties", []) or []
        verified = []
        for prop in props:
            cid = clean(prop.get("CID"))
            if not cid:
                continue
            syn_url = f"{PUBCHEM_API_ROOT}/compound/cid/{quote(cid, safe='')}/synonyms/JSON"
            syn = _pubchem_json(syn_url)
            info = syn.get("InformationList", {}).get("Information", []) or []
            synonyms = info[0].get("Synonym", []) if info else []
            if cas_norm in {normalize_cas(v) for v in synonyms}:
                verified.append(prop)

        if not verified:
            rec["structure_status"] = "PUBCHEM_CAS_SYNONYM_MISMATCH"
            rec["pubchem_error"] = "Exact CAS not confirmed in PubChem synonyms"
            return rec
        if len(verified) > 1:
            rec["structure_status"] = "PUBCHEM_MULTIPLE_CANDIDATES_REVIEW"
            rec["pubchem_cid"] = ";".join(clean(v.get("CID")) for v in verified)
            rec["pubchem_error"] = "Multiple PubChem records carry exact CAS synonym"
            return rec

        prop = verified[0]
        cid = clean(prop.get("CID"))
        iso = clean(prop.get("IsomericSMILES")) or clean(prop.get("SMILES"))
        conn = clean(prop.get("CanonicalSMILES")) or clean(prop.get("ConnectivitySMILES")) or iso
        if not iso and not conn:
            rec["structure_status"] = "PUBCHEM_MISSING_SMILES_REVIEW"
            rec["pubchem_cid"] = cid
            return rec
        rec.update({
            "structure_status": "PUBCHEM_EXACT_CAS_VERIFIED",
            "pubchem_cid": cid,
            "pubchem_title": clean(prop.get("Title")),
            "pubchem_isomeric_smiles": iso or conn,
            "pubchem_connectivity_smiles": conn or iso,
            "pubchem_inchikey": clean(prop.get("InChIKey")),
            "pubchem_cas_verified": True,
            "pubchem_record_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
        })
        return rec
    except LookupError:
        rec["structure_status"] = "PUBCHEM_NOT_FOUND"
        rec["pubchem_error"] = "No PubChem compound found for CAS"
        return rec
    except Exception as exc:
        rec["structure_status"] = "PUBCHEM_REQUEST_ERROR"
        rec["pubchem_error"] = f"{type(exc).__name__}: {exc}"
        return rec


def _load_pubchem_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return {}
    out = {}
    for _, r in df.iterrows():
        cas = normalize_cas(r.get("cas"))
        if cas:
            out[cas] = {c: clean(r.get(c)) for c in PUBCHEM_CACHE_COLUMNS}
    return out


def _save_pubchem_cache(path: Path, cache: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{c: clean(v.get(c)) for c in PUBCHEM_CACHE_COLUMNS} for v in cache.values()]
    pd.DataFrame(rows, columns=PUBCHEM_CACHE_COLUMNS).sort_values("cas").to_csv(
        path, index=False, encoding="utf-8-sig"
    )


def enrich_inventory_with_pubchem(inventory: pd.DataFrame, cache_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve CAS identifiers to structures and record per-row lookup time.

    Timing semantics
    ----------------
    structure_lookup_elapsed_sec measures the wall-clock time spent resolving the
    structure for that inventory row. It therefore includes PubChem network latency
    for a cold lookup, but only local dictionary/cache access for a cache hit.
    Set IDENTITY_PUBCHEM_COLD_START=1 to ignore a pre-existing cache at the start
    of the run while still reusing structures for repeated CAS values within the
    same run.
    """
    out = inventory.copy()
    if "smiles" not in out.columns:
        out["smiles"] = ""
    cache = {} if PUBCHEM_COLD_START else _load_pubchem_cache(cache_file)
    audit_rows = []
    chosen_smiles = []
    statuses = []
    sources = []
    cids = []
    urls = []
    lookup_seconds = []
    lookup_modes = []

    for idx, row in out.iterrows():
        lookup_t0 = time.perf_counter()
        cas = normalize_cas(row.get("cas"))
        supplied = clean(row.get("smiles"))
        lookup_mode = ""
        if supplied:
            rec = {c: "" for c in PUBCHEM_CACHE_COLUMNS}
            rec.update({"cas": cas, "structure_status": "USER_SMILES_PRESENT"})
            chosen, source = supplied, "USER_SUPPLIED"
            lookup_mode = "USER_SUPPLIED"
        else:
            rec = cache.get(cas)
            cache_usable = rec is not None and clean(rec.get("structure_status")) != "PUBCHEM_REQUEST_ERROR"
            if cache_usable:
                lookup_mode = "CACHE_HIT"
            else:
                lookup_mode = "NETWORK_LOOKUP" if rec is None else "NETWORK_RETRY"
                rec = fetch_pubchem_structure_by_cas(cas)
                status_now = clean(rec.get("structure_status"))
                if status_now == "PUBCHEM_REQUEST_ERROR":
                    # Preserve successful work already completed, then STOP.
                    # A transient network/server error is not a chemical-identity result.
                    if cache:
                        _save_pubchem_cache(cache_file, cache)
                    raise RuntimeError(
                        f"Transient PubChem request failure for CAS {cas}. "
                        "Preflight stopped so the same CAS can never receive different "
                        "external-DB information within a completed benchmark run. "
                        f"Detail: {clean(rec.get('pubchem_error'))}"
                    )
                if cas:
                    cache[cas] = rec.copy()
                    _save_pubchem_cache(cache_file, cache)
            if clean(rec.get("structure_status")) == "PUBCHEM_EXACT_CAS_VERIFIED":
                chosen = clean(rec.get("pubchem_isomeric_smiles")) or clean(rec.get("pubchem_connectivity_smiles"))
                source = "PUBCHEM_PUG_REST"
            else:
                chosen, source = "", "UNRESOLVED"

        lookup_elapsed = time.perf_counter() - lookup_t0
        chosen_smiles.append(chosen)
        statuses.append(clean(rec.get("structure_status")))
        sources.append(source)
        cids.append(clean(rec.get("pubchem_cid")))
        urls.append(clean(rec.get("pubchem_record_url")))
        lookup_seconds.append(float(lookup_elapsed))
        lookup_modes.append(lookup_mode)
        audit_rows.append({
            "inventory_row": idx,
            **{c: clean(rec.get(c)) for c in PUBCHEM_CACHE_COLUMNS},
            "lookup_mode": lookup_mode,
            "lookup_elapsed_sec": float(lookup_elapsed),
        })

    out["smiles"] = chosen_smiles
    out["structure_status"] = statuses
    out["structure_smiles_source"] = sources
    out["pubchem_cid"] = cids
    out["pubchem_record_url"] = urls
    out["structure_lookup_elapsed_sec"] = lookup_seconds
    out["structure_lookup_mode"] = lookup_modes
    if cache:
        _save_pubchem_cache(cache_file, cache)
    return out, pd.DataFrame(audit_rows)


def normalize_isomer_scope(x) -> str:
    value = clean(x).upper()
    aliases = {
        "ALL": "ALL_STEREOISOMERS",
        "ALL_ISOMERS": "ALL_STEREOISOMERS",
        "ALL_STEREOISOMERS": "ALL_STEREOISOMERS",
        "EXACT": "EXACT_STEREO_ONLY",
        "EXACT_ONLY": "EXACT_STEREO_ONLY",
        "EXACT_STEREO_ONLY": "EXACT_STEREO_ONLY",
    }
    return aliases.get(value, "UNSPECIFIED_REVIEW")


def parent_key(smiles: str) -> str:
    return ci.parent_smiles(smiles)


def parent_connectivity_key(smiles: str) -> str:
    parent = ci.parent_smiles(smiles)
    if not parent:
        return ""
    try:
        mol = Chem.MolFromSmiles(parent)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False) if mol is not None else ""
    except Exception:
        return ""


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
    # FAIR OPERATIONAL SHARED-DB ARM:
    # Claude gets the same PubChem-resolved candidate structure and the same frozen
    # reference-parent structure used by RDKit. Candidate/reference names are not
    # separately supplied, preventing trivial lexical answer leakage.
    if condition == "DB_INFORMED":
        base = {
            "case_id": clean(row.get("case_id")),
            "candidate_cas": clean(row.get("candidate_cas")),
            "candidate_pubchem_smiles": clean(row.get("operational_resolved_smiles")),
            "candidate_structure_status": clean(row.get("operational_structure_status")),
            "reference_parent_smiles": clean(row.get("reference_parent_smiles")),
            "scope_definition": "The regulated scope is the reference parent chemical and its counterion salts.",
            "structural_rule": (
                "Treat disconnected counterion/solvate fragments as removable salt components. "
                "After parent-fragment normalization, MATCH only when candidate and reference represent the same parent identity; "
                "otherwise NO_MATCH. If the supplied structure is missing or stereochemical scope cannot be resolved, REVIEW."
            ),
        }
    # CONTROLLED arm: structures only; lexical identity cues are withheld.
    elif condition == "STRUCTURE_INFORMED":
        base = {
            "case_id": clean(row.get("case_id")),
            "candidate_smiles": clean(row.get("candidate_source_smiles")),
            "reference_parent_smiles": clean(row.get("reference_parent_smiles")),
            "scope_definition": "The regulated scope is the reference parent chemical and its counterion salts.",
            "structural_rule": (
                "Treat disconnected counterion/solvate fragments as removable salt components. "
                "After parent-fragment normalization, MATCH only when candidate and reference represent the same parent identity; "
                "otherwise NO_MATCH. If stereochemical scope cannot be resolved from the supplied structures, REVIEW."
            ),
        }
    else:
        raise ValueError(f"Unsupported final Study-2 input condition: {condition}")

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
    # Keep the already-completed CAS-only/CAS+name caches valid. Only the controlled
    # structure arm receives a new prompt-version tag after blinding lexical cues.
    prompt_version = (
        STRUCTURE_PROMPT_VERSION if condition == "STRUCTURE_INFORMED" else
        DB_PROMPT_VERSION if condition == "DB_INFORMED" else
        PROMPT_VERSION
    )
    payload = "|".join([
        CACHE_VERSION, prompt_version, ANTHROPIC_MODEL, condition, str(repeat),
        clean(row.get("case_id")), hashlib.sha256(prompt_for(row, condition).encode("utf-8")).hexdigest(),
        EFFORT, str(MAX_TOKENS),
    ])
    key = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", ANTHROPIC_MODEL)
    d = CACHE_DIR / safe_model / condition.lower()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"r{repeat}_{clean(row.get('case_id'))}_{key}.json"


def normalize_llm(
    raw: dict, row: pd.Series, condition: str, repeat: int, status: str = "OK",
    elapsed_sec: float = np.nan, runtime_source: str = ""
) -> dict:
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
        "elapsed_sec": float(elapsed_sec) if np.isfinite(elapsed_sec) else np.nan,
        "runtime_source": runtime_source,
    }


def run_one_llm(row_dict: dict, condition: str, repeat: int) -> dict:
    """Run one Claude case and record true per-case wall-clock latency.

    Fresh API calls are timed around call_claude(), including retry/backoff time.
    Cache hits retain a previously stored API elapsed time when available; old
    caches created before runtime instrumentation may contain no elapsed time.
    Use IDENTITY_FORCE_LLM=1 for a clean fresh timing run.
    """
    row = pd.Series(row_dict)
    p = cache_file(row, condition, repeat)
    cache_t0 = time.perf_counter()
    if p.exists() and not FORCE:
        try:
            saved = json.loads(p.read_text(encoding="utf-8"))
            if saved.get("call_status") == "OK":
                saved.setdefault("runtime_source", "CACHE_HIT_WITH_STORED_API_TIME" if saved.get("elapsed_sec") not in {None, ""} else "CACHE_HIT_NO_STORED_API_TIME")
                saved["cache_read_elapsed_sec"] = float(time.perf_counter() - cache_t0)
                return saved
        except Exception:
            pass
    try:
        t0 = time.perf_counter()
        raw = call_claude(row, condition)
        elapsed = time.perf_counter() - t0
        out = normalize_llm(raw, row, condition, repeat, "OK", elapsed_sec=elapsed, runtime_source="FRESH_API_CALL")
        # Cache only successful transport/model responses, including measured API latency.
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return out
    except Exception as exc:
        elapsed = time.perf_counter() - t0 if 't0' in locals() else np.nan
        return normalize_llm(
            {}, row, condition, repeat,
            f"LLM_ERROR:{type(exc).__name__}:{str(exc)[:300]}",
            elapsed_sec=elapsed, runtime_source="FRESH_API_CALL_ERROR"
        )


def normalize_rule_scope(x) -> str:
    return normalize_isomer_scope(x)


def compare_rdkit(candidate_smiles: str, reference_smiles: str, isomer_scope: str = "") -> tuple[str, str]:
    ck = parent_key(candidate_smiles)
    cc = parent_connectivity_key(candidate_smiles)
    rk = parent_key(reference_smiles)
    rc = parent_connectivity_key(reference_smiles)
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


def timed_compare_rdkit(candidate_smiles: str, reference_smiles: str, isomer_scope: str = "") -> tuple[str, str, float]:
    t0 = time.perf_counter()
    decision, reason = compare_rdkit(candidate_smiles, reference_smiles, isomer_scope)
    return decision, reason, float(time.perf_counter() - t0)


def compare_rdkit_v43(candidate_smiles: str, reference_smiles: str, isomer_scope: str = "") -> tuple[str, str]:
    """Post-preflight salt-aware sensitivity comparator; not primary inference."""
    return ci43.compare_salt_parent_v43(candidate_smiles, reference_smiles, isomer_scope)


def timed_compare_rdkit_v43(candidate_smiles: str, reference_smiles: str, isomer_scope: str = "") -> tuple[str, str, float]:
    t0 = time.perf_counter()
    decision, reason = compare_rdkit_v43(candidate_smiles, reference_smiles, isomer_scope)
    return decision, reason, float(time.perf_counter() - t0)


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


# -----------------------------------------------------------------------------
# FINAL shared-DB execution policy
# -----------------------------------------------------------------------------
READINESS_FILE = INTERMEDIATE / "04_identity_benchmark_readiness.json"
EXECUTE_CLAUDE = os.getenv("IDENTITY_EXECUTE_CLAUDE", "0").strip().lower() in {"1", "true", "yes", "on"}
PROMPT_MANIFEST_FILE = INTERMEDIATE / "05_claude_prompt_manifest.csv"
API_PLAN_FILE = INTERMEDIATE / "05_claude_api_plan.json"
RDKIT_PREFLIGHT_FILE = INTERMEDIATE / "05_rdkit_preflight_predictions.csv"


def _load_readiness() -> dict:
    if not READINESS_FILE.exists():
        return {
            "ready_for_claude_api": False,
            "failed_checks": ["readiness_file_missing"],
            "note": "Run 04_build_identity_challenge_SHARED_DB_EXPANDED.py first.",
        }
    try:
        return json.loads(READINESS_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "ready_for_claude_api": False,
            "failed_checks": [f"readiness_json_error:{type(exc).__name__}"],
        }


def _benchmark_metadata_cols(bench: pd.DataFrame) -> list[str]:
    preferred = [
        "case_id", "benchmark_tier", "designation_id", "rule_id", "candidate_cas", "candidate_name",
        "reference_parent_name", "reference_membership", "challenge_class", "difficulty",
        "difficulty_definition_source", "difficulty_frozen_before_llm",
        "benchmark_sets", "in_primary_source_curated", "in_secondary_structure_anchored",
        "primary_case_id", "secondary_case_id", "secondary_difficulty",
        "secondary_challenge_class", "stereo_scope_sensitivity_flag",
        "rdkit_structure_qc_used_for_secondary_inclusion",
    ]
    return [c for c in preferred if c in bench.columns]


def main() -> None:
    load_dotenv()
    print("=" * 96)
    print("05 / STUDY 2 METHOD-SAFE V4 - DUAL BENCHMARK, FROZEN RDKIT + V4.3 SENSITIVITY")
    print("=" * 96)
    print(f"Paid Claude execution enabled: {EXECUTE_CLAUDE}")
    print(f"Frozen deterministic engine: {Path(ci42.__file__).resolve()}")
    print(f"V4.3 sensitivity engine: {Path(ci43.__file__).resolve()}")

    if not BENCHMARK_FILE.exists():
        raise FileNotFoundError("Run 04_build_identity_challenge_SHARED_DB_EXPANDED.py first")
    bench = pd.read_csv(BENCHMARK_FILE).fillna("")
    if MAX_CASES:
        bench = bench.head(MAX_CASES).copy()
    if bench.empty:
        raise RuntimeError("Study 2 benchmark is empty")

    # FINAL experiment is operational shared-DB only.  No closed LLM and no
    # second Claude structure arm: this prevents duplicated paid calls.
    if "operational_cas_eligible" not in bench.columns:
        bench["operational_cas_eligible"] = bench["candidate_cas"].map(clean).ne("")
    bench["operational_cas_eligible"] = bench["operational_cas_eligible"].map(flag_bool)
    op_bench = bench[bench["operational_cas_eligible"]].copy()
    if op_bench.empty:
        raise RuntimeError("No operational CAS-eligible rows")

    # Frozen rule metadata for stereochemical policy when available.
    rules = pd.read_csv(RULE_SNAPSHOT_FILE).fillna("") if RULE_SNAPSHOT_FILE.exists() else pd.DataFrame()
    isomer_by_rule = {}
    if not rules.empty:
        if "isomer_scope" not in rules.columns:
            rules["isomer_scope"] = ""
        isomer_by_rule = dict(zip(rules["rule_id"].map(clean), rules["isomer_scope"].map(clean)))

    # ------------------------------------------------------------------
    # A. ONE shared external-DB resolution for all three systems
    # ------------------------------------------------------------------
    deterministic_stage_t0 = time.perf_counter()
    inv = pd.DataFrame({
        "chemical_name": ["" for _ in range(len(op_bench))],  # avoid lexical leakage
        "cas": op_bench["candidate_cas"].tolist(),
        "smiles": ["" for _ in range(len(op_bench))],
    }, index=op_bench.index)
    resolved, pubchem_audit = enrich_inventory_with_pubchem(
        inv, cache_file=INTERMEDIATE / "05_pubchem_operational_identity_cache.csv"
    )

    # Completed-run consistency gate:
    # repeated occurrences of the same CAS must receive the same shared-DB status/CID.
    _audit = pubchem_audit.copy()
    inconsistent = []
    for cas_value, g in _audit.groupby("cas", dropna=False):
        signatures = {
            (
                clean(r.get("structure_status")),
                clean(r.get("pubchem_cid")),
                clean(r.get("pubchem_isomeric_smiles")),
                clean(r.get("pubchem_connectivity_smiles")),
            )
            for _, r in g.iterrows()
        }
        if len(signatures) > 1:
            inconsistent.append(clean(cas_value))
    if inconsistent:
        raise RuntimeError(
            "Shared-DB consistency gate failed: identical CAS received different "
            f"PubChem resolution results within the same completed run: {inconsistent}"
        )

    status_counts = _audit["structure_status"].value_counts(dropna=False).to_dict()
    request_errors = int((_audit["structure_status"] == "PUBCHEM_REQUEST_ERROR").sum())
    if request_errors:
        raise RuntimeError(
            f"Shared-DB preflight contains {request_errors} transient PubChem request error(s). "
            "Do not run Claude; rerun the dry preflight."
        )

    shared_db_qc = {
        "n_rows": int(len(_audit)),
        "n_unique_cas": int(_audit["cas"].nunique(dropna=True)),
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "n_request_errors": request_errors,
        "n_inconsistent_repeated_cas": int(len(inconsistent)),
        "inconsistent_cas": inconsistent,
        "prompt_regulatory_url_removed": True,
    }
    (INTERMEDIATE / "05_shared_db_preflight_qc.json").write_text(
        json.dumps(shared_db_qc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Shared-DB preflight QC:", json.dumps(shared_db_qc, ensure_ascii=False))

    for col in [
        "operational_resolved_smiles", "operational_structure_status", "operational_structure_source",
        "operational_pubchem_cid", "operational_pubchem_record_url", "operational_structure_lookup_mode",
    ]:
        bench[col] = ""
    bench["operational_structure_lookup_elapsed_sec"] = np.nan
    bench.loc[op_bench.index, "operational_resolved_smiles"] = resolved["smiles"].tolist()
    bench.loc[op_bench.index, "operational_structure_status"] = resolved["structure_status"].tolist()
    bench.loc[op_bench.index, "operational_structure_source"] = resolved["structure_smiles_source"].tolist()
    bench.loc[op_bench.index, "operational_pubchem_cid"] = resolved["pubchem_cid"].tolist()
    bench.loc[op_bench.index, "operational_pubchem_record_url"] = resolved["pubchem_record_url"].tolist()
    bench.loc[op_bench.index, "operational_structure_lookup_elapsed_sec"] = resolved["structure_lookup_elapsed_sec"].to_numpy(dtype=float)
    bench.loc[op_bench.index, "operational_structure_lookup_mode"] = resolved["structure_lookup_mode"].tolist()

    rdkit_rows = []
    for _, r in bench.iterrows():
        scope = isomer_by_rule.get(clean(r.get("rule_id")), "")
        op_dec, op_reason, op_compare_sec = timed_compare_rdkit(
            r.get("operational_resolved_smiles"), r.get("reference_parent_smiles"), scope
        )
        v43_dec, v43_reason, v43_compare_sec = timed_compare_rdkit_v43(
            r.get("operational_resolved_smiles"), r.get("reference_parent_smiles"), scope
        )
        lookup_sec = pd.to_numeric(pd.Series([r.get("operational_structure_lookup_elapsed_sec")]), errors="coerce").iloc[0]
        lookup_sec = float(lookup_sec) if pd.notna(lookup_sec) else np.nan
        total_sec = float(op_compare_sec + lookup_sec) if np.isfinite(lookup_sec) else np.nan
        v43_total_sec = float(v43_compare_sec + lookup_sec) if np.isfinite(lookup_sec) else np.nan
        rdkit_rows.append({
            "case_id": clean(r.get("case_id")),
            "rdkit_operational_decision": op_dec,
            "rdkit_operational_reason": op_reason,
            "rdkit_operational_compare_elapsed_sec": op_compare_sec,
            "rdkit_operational_total_elapsed_sec": total_sec,
            "rdkit_v43_sensitivity_decision": v43_dec,
            "rdkit_v43_sensitivity_reason": v43_reason,
            "rdkit_v43_sensitivity_compare_elapsed_sec": v43_compare_sec,
            "rdkit_v43_sensitivity_total_elapsed_sec": v43_total_sec,
        })
    bench = bench.merge(pd.DataFrame(rdkit_rows), on="case_id", how="left")
    deterministic_stage_wall_sec = float(time.perf_counter() - deterministic_stage_t0)

    pubchem_audit.to_csv(INTERMEDIATE / "05_pubchem_operational_identity_audit.csv", index=False, encoding="utf-8-sig")
    bench.to_csv(INTERMEDIATE / "05_identity_benchmark_enriched.csv", index=False, encoding="utf-8-sig")

    # RDKit-only preflight can be inspected before paying for Claude.
    preflight_cols = _benchmark_metadata_cols(bench) + [
        "operational_structure_status", "operational_resolved_smiles",
        "rdkit_operational_decision", "rdkit_operational_reason",
        "rdkit_v43_sensitivity_decision", "rdkit_v43_sensitivity_reason",
    ]
    bench[[c for c in preflight_cols if c in bench.columns]].to_csv(
        RDKIT_PREFLIGHT_FILE, index=False, encoding="utf-8-sig"
    )

    # ------------------------------------------------------------------
    # B. Freeze the exact Claude input manifest BEFORE any paid call
    # ------------------------------------------------------------------
    manifest_rows = []
    for _, r in bench[bench["operational_cas_eligible"]].iterrows():
        ptxt = prompt_for(r, "DB_INFORMED")
        manifest_rows.append({
            **{c: r.get(c, "") for c in _benchmark_metadata_cols(bench)},
            "input_condition": "DB_INFORMED",
            "candidate_structure_status": clean(r.get("operational_structure_status")),
            "candidate_pubchem_cid": clean(r.get("operational_pubchem_cid")),
            "prompt_sha256": hashlib.sha256(ptxt.encode("utf-8")).hexdigest(),
            "prompt_text": ptxt,
        })
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(PROMPT_MANIFEST_FILE, index=False, encoding="utf-8-sig")

    readiness = _load_readiness()
    n_cases = int(len(manifest))
    planned_calls = int(n_cases * N_REPEATS)
    api_plan = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "execute_claude": EXECUTE_CLAUDE,
        "benchmark_ready": bool(readiness.get("ready_for_claude_api", False)),
        "failed_readiness_checks": readiness.get("failed_checks", []),
        "n_cases": n_cases,
        "n_repeats": N_REPEATS,
        "planned_paid_claude_calls": planned_calls,
        "additional_hybrid_calls": 0,
        "unique_identity_pairs_evaluated_once": True,
        "v43_sensitivity_additional_claude_calls": 0,
        "reason_additional_hybrid_calls_zero": "Hybrid reuses the already-computed LLM+DB result for the same case/repeat only when RDKit returns REVIEW.",
        "conditions_run": ["DB_INFORMED"],
        "closed_llm_removed": True,
        "controlled_structure_claude_removed": True,
        "prompt_manifest": str(PROMPT_MANIFEST_FILE),
    }
    API_PLAN_FILE.write_text(json.dumps(api_plan, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nShared-DB cases prepared: {n_cases}")
    print(f"Planned Claude calls if enabled: {planned_calls} (= {n_cases} cases × {N_REPEATS} repeats)")
    print("Hybrid additional paid calls: 0")
    print(f"Prompt manifest: {PROMPT_MANIFEST_FILE}")
    print(f"RDKit preflight: {RDKIT_PREFLIGHT_FILE}")

    if not EXECUTE_CLAUDE:
        print("\n[STOP BEFORE PAID API]")
        print("No Claude API call was made.")
        print("Inspect the benchmark, PubChem audit, RDKit preflight, and exact prompt manifest first.")
        print("When everything is frozen and readiness=true, run with IDENTITY_EXECUTE_CLAUDE=1.")
        return

    if not bool(readiness.get("ready_for_claude_api", False)):
        failed = readiness.get("failed_checks", [])
        raise RuntimeError(
            "Paid Claude run blocked by Step 04 readiness gate. Failed checks: " + ", ".join(map(str, failed))
        )
    if not api_key():
        raise RuntimeError("ANTHROPIC_API_KEY is missing. No API call was made.")

    # ------------------------------------------------------------------
    # C. Claude + DB: ONE paid arm only
    # ------------------------------------------------------------------
    print(f"\nClaude model: {ANTHROPIC_MODEL} | repeats={N_REPEATS} | workers={WORKERS}")
    print("Condition: DB_INFORMED only (same PubChem structure information used by RDKit)")
    llm_rows, batch_runtime_rows = [], []
    records = bench[bench["operational_cas_eligible"]].to_dict("records")
    for rep in range(1, N_REPEATS + 1):
        print(f"\n[DB_INFORMED] repeat {rep}/{N_REPEATS} | n={len(records)}")
        batch_t0 = time.perf_counter(); done = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(run_one_llm, row, "DB_INFORMED", rep): row for row in records}
            for fut in as_completed(futs):
                llm_rows.append(fut.result()); done += 1
                if done % 10 == 0 or done == len(records):