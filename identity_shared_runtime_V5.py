# -*- coding: utf-8 -*-
"""Shared PubChem + Anthropic runtime for Study 2 FAIR V5.

This module contains transport/runtime utilities only. It does NOT contain a
chemical-identity comparator and it never reads benchmark labels. Keeping these
utilities separate makes FAIR V5 independent of the retired V4 evaluator while
still freezing the exact external-data/API behavior used by the experiment.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
INTERMEDIATE = ROOT / "intermediate"
INTERMEDIATE.mkdir(parents=True, exist_ok=True)


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


load_dotenv()

ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = os.getenv("ANTHROPIC_VERSION", "2023-06-01").strip()
ANTHROPIC_MODEL = os.getenv("IDENTITY_ANTHROPIC_MODEL", "claude-sonnet-5").strip()
N_REPEATS = max(1, int(os.getenv("IDENTITY_LLM_REPEATS", "3")))
TIMEOUT = max(30, int(os.getenv("IDENTITY_ANTHROPIC_TIMEOUT", "180")))
MAX_TOKENS = max(256, int(os.getenv("IDENTITY_ANTHROPIC_MAX_TOKENS", "700")))
EFFORT = os.getenv("IDENTITY_ANTHROPIC_EFFORT", "medium").strip().lower() or "medium"
FORCE = os.getenv("IDENTITY_FORCE_LLM", "0").strip().lower() in {"1", "true", "yes", "on"}
PUBCHEM_COLD_START = os.getenv("IDENTITY_PUBCHEM_COLD_START", "0").strip().lower() in {"1", "true", "yes", "on"}
PUBCHEM_RETRIES = max(3, int(os.getenv("IDENTITY_PUBCHEM_RETRIES", "8")))
PUBCHEM_MIN_INTERVAL_SEC = max(0.0, float(os.getenv("IDENTITY_PUBCHEM_MIN_INTERVAL_SEC", "0.35")))

CACHE_DIR = INTERMEDIATE / "identity_llm_cache_v5"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DB_PROMPT_VERSION = "study2-fair-v5-runtime"
CACHE_VERSION = "study2-fair-v5-claude-cache-v1"

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
You cannot independently browse or call tools. Use only the supplied fields.
Return only the requested JSON object."""

_PROMPT_BUILDER: Callable[[pd.Series, str], str] | None = None


def set_prompt_builder(fn: Callable[[pd.Series, str], str], prompt_version: str) -> None:
    global _PROMPT_BUILDER, DB_PROMPT_VERSION
    _PROMPT_BUILDER = fn
    DB_PROMPT_VERSION = str(prompt_version)


def prompt_for(row: pd.Series, condition: str) -> str:
    if _PROMPT_BUILDER is None:
        raise RuntimeError("FAIR V5 prompt builder has not been installed")
    return _PROMPT_BUILDER(row, condition)


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
PUBCHEM_CACHE_COLUMNS = [
    "cas", "structure_status", "pubchem_cid", "pubchem_title",
    "pubchem_isomeric_smiles", "pubchem_connectivity_smiles", "pubchem_inchikey",
    "pubchem_cas_verified", "pubchem_retrieval_date", "pubchem_record_url",
    "pubchem_query_url", "pubchem_error",
]
_LAST_PUBCHEM_REQUEST_AT = 0.0


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
    global _LAST_PUBCHEM_REQUEST_AT
    last = None
    for attempt in range(max(1, retries)):
        try:
            wait = PUBCHEM_MIN_INTERVAL_SEC - (time.monotonic() - _LAST_PUBCHEM_REQUEST_AT)
            if wait > 0:
                time.sleep(wait)
            _LAST_PUBCHEM_REQUEST_AT = time.monotonic()
            r = requests.get(url, timeout=timeout, headers={
                "User-Agent": "chemical-regulatory-study2-presentation/5.3",
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
                    delay = max(retry_after, min(30.0, 2 ** attempt))
                    print(f"[PUBCHEM RETRY] HTTP {r.status_code} attempt {attempt + 1}/{retries}; sleep {delay:.1f}s", flush=True)
                    time.sleep(delay)
                    continue
            r.raise_for_status()
            return r.json()
        except LookupError:
            raise
        except Exception as exc:
            last = exc
            if attempt + 1 < retries:
                delay = min(30.0, 2 ** attempt)
                print(f"[PUBCHEM RETRY] {type(exc).__name__} attempt {attempt + 1}/{retries}; sleep {delay:.1f}s", flush=True)
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
    url = f"{PUBCHEM_API_ROOT}/compound/name/{encoded}/property/Title,CanonicalSMILES,IsomericSMILES,InChIKey/JSON"
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
    return {
        normalize_cas(r.get("cas")): {c: clean(r.get(c)) for c in PUBCHEM_CACHE_COLUMNS}
        for _, r in df.iterrows() if normalize_cas(r.get("cas"))
    }


def _save_pubchem_cache(path: Path, cache: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{c: clean(v.get(c)) for c in PUBCHEM_CACHE_COLUMNS} for v in cache.values()]
    pd.DataFrame(rows, columns=PUBCHEM_CACHE_COLUMNS).sort_values("cas").to_csv(path, index=False, encoding="utf-8-sig")


def enrich_inventory_with_pubchem(inventory: pd.DataFrame, cache_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = inventory.copy()
    if "smiles" not in out.columns:
        out["smiles"] = ""
    cache = {} if PUBCHEM_COLD_START else _load_pubchem_cache(cache_file)
    total = len(out)
    print(f"[PUBCHEM] Starting shared exact-CAS structure resolution: {total} rows; cached CAS={len(cache)}", flush=True)
    audit_rows, chosen_smiles, statuses, sources, cids, urls, lookup_seconds, lookup_modes = [], [], [], [], [], [], [], []
    for pos, (idx, row) in enumerate(out.iterrows(), start=1):
        t0 = time.perf_counter()
        cas = normalize_cas(row.get("cas"))
        supplied = clean(row.get("smiles"))
        if supplied:
            rec = {c: "" for c in PUBCHEM_CACHE_COLUMNS}
            rec.update({"cas": cas, "structure_status": "USER_SMILES_PRESENT"})
            chosen, source, mode = supplied, "USER_SUPPLIED", "USER_SUPPLIED"
        else:
            rec = cache.get(cas)
            usable = rec is not None and clean(rec.get("structure_status")) != "PUBCHEM_REQUEST_ERROR"
            mode = "CACHE_HIT" if usable else ("NETWORK_LOOKUP" if rec is None else "NETWORK_RETRY")
            if not usable:
                print(f"[PUBCHEM {pos}/{total}] CAS {cas}: network lookup...", flush=True)
                rec = fetch_pubchem_structure_by_cas(cas)
                if clean(rec.get("structure_status")) == "PUBCHEM_REQUEST_ERROR":
                    if cache:
                        _save_pubchem_cache(cache_file, cache)
                    raise RuntimeError(f"Transient PubChem request failure for CAS {cas}: {clean(rec.get('pubchem_error'))}")
                if cas:
                    cache[cas] = rec.copy()
                    _save_pubchem_cache(cache_file, cache)
            if clean(rec.get("structure_status")) == "PUBCHEM_EXACT_CAS_VERIFIED":
                chosen = clean(rec.get("pubchem_isomeric_smiles")) or clean(rec.get("pubchem_connectivity_smiles"))
                source = "PUBCHEM_PUG_REST"
            else:
                chosen, source = "", "UNRESOLVED"
        elapsed = float(time.perf_counter() - t0)
        chosen_smiles.append(chosen); statuses.append(clean(rec.get("structure_status"))); sources.append(source)
        cids.append(clean(rec.get("pubchem_cid"))); urls.append(clean(rec.get("pubchem_record_url")))
        lookup_seconds.append(elapsed); lookup_modes.append(mode)
        audit_rows.append({"inventory_row": idx, **{c: clean(rec.get(c)) for c in PUBCHEM_CACHE_COLUMNS}, "lookup_mode": mode, "lookup_elapsed_sec": elapsed})
        print(f"[PUBCHEM {pos}/{total}] CAS {cas}: {clean(rec.get('structure_status'))} ({mode}, {elapsed:.2f}s)", flush=True)
    out["smiles"] = chosen_smiles; out["structure_status"] = statuses; out["structure_smiles_source"] = sources
    out["pubchem_cid"] = cids; out["pubchem_record_url"] = urls; out["structure_lookup_elapsed_sec"] = lookup_seconds; out["structure_lookup_mode"] = lookup_modes
    if cache:
        _save_pubchem_cache(cache_file, cache)
    print(f"[PUBCHEM] Completed shared structure resolution: {total}/{total}", flush=True)
    return out, pd.DataFrame(audit_rows)


def api_key() -> str:
    return clean(os.getenv("ANTHROPIC_API_KEY", ""))


def validate_api_key_for_http_header() -> str:
    key = api_key()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is missing. No API call was made.")
    try:
        key.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError("ANTHROPIC_API_KEY contains non-ASCII characters; replace placeholder text with the actual API key.") from exc
    return key


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
    raise ValueError("No JSON object in model response")


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
        "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
    }
    headers = {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json"}
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
            text = "".join(str(p.get("text", "")) for p in (payload.get("content") or []) if isinstance(p, dict) and p.get("type") == "text").strip()
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
        CACHE_VERSION, DB_PROMPT_VERSION, ANTHROPIC_MODEL, condition, str(repeat), clean(row.get("case_id")),
        hashlib.sha256(prompt_for(row, condition).encode("utf-8")).hexdigest(), EFFORT, str(MAX_TOKENS),
    ])
    key = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", ANTHROPIC_MODEL)
    d = CACHE_DIR / safe_model / condition.lower(); d.mkdir(parents=True, exist_ok=True)
    return d / f"r{repeat}_{clean(row.get('case_id'))}_{key}.json"


def normalize_llm(raw: dict, row: pd.Series, condition: str, repeat: int, status: str = "OK", elapsed_sec: float = np.nan, runtime_source: str = "") -> dict:
    decision = clean(raw.get("decision")).upper(); valid = decision in DECISIONS
    if not valid:
        decision = "REVIEW"
    return {
        "case_id": clean(row.get("case_id")), "identity_benchmark_id": clean(row.get("identity_benchmark_id")),
        "input_condition": condition, "llm_model": ANTHROPIC_MODEL, "repeat": repeat, "decision": decision,
        "classification_valid": bool(valid and status == "OK"), "reason": clean(raw.get("reason")),
        "confidence": clean(raw.get("confidence")).upper(), "call_status": status,
        "model_returned": clean(raw.get("_model_returned")), "input_tokens": raw.get("_input_tokens", ""),
        "output_tokens": raw.get("_output_tokens", ""), "elapsed_sec": float(elapsed_sec) if np.isfinite(elapsed_sec) else np.nan,
        "runtime_source": runtime_source,
    }


def run_one_llm(row_dict: dict, condition: str, repeat: int) -> dict:
    row = pd.Series(row_dict); p = cache_file(row, condition, repeat); t_cache = time.perf_counter()
    if p.exists() and not FORCE:
        try:
            saved = json.loads(p.read_text(encoding="utf-8"))
            if saved.get("call_status") == "OK":
                saved.setdefault("runtime_source", "CACHE_HIT_WITH_STORED_API_TIME" if saved.get("elapsed_sec") not in {None, ""} else "CACHE_HIT_NO_STORED_API_TIME")
                saved["cache_read_elapsed_sec"] = float(time.perf_counter() - t_cache)
                return saved
        except Exception:
            pass
    try:
        t0 = time.perf_counter(); raw = call_claude(row, condition); elapsed = time.perf_counter() - t0
        out = normalize_llm(raw, row, condition, repeat, "OK", elapsed, "FRESH_API_CALL")
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return out
    except Exception as exc:
        elapsed = time.perf_counter() - t0 if "t0" in locals() else np.nan
        return normalize_llm({}, row, condition, repeat, f"LLM_ERROR:{type(exc).__name__}:{str(exc)[:300]}", elapsed, "FRESH_API_CALL_ERROR")
