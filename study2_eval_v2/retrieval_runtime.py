# -*- coding: utf-8 -*-
"""Anthropic runtime for CAS-only regulatory retrieval evaluation.

This module never reads GOLD files.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
CACHE_DIR = ROOT / "intermediate" / "llm_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_dotenv() -> None:
    for p in [REPO_ROOT / ".env", Path.cwd() / ".env"]:
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


_load_dotenv()

ENDPOINT = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = os.getenv("ANTHROPIC_VERSION", "2023-06-01").strip()
MODEL = os.getenv("IDENTITY_ANTHROPIC_MODEL", "claude-sonnet-5").strip()
N_REPEATS = max(1, int(os.getenv("RETRIEVAL_LLM_REPEATS", os.getenv("IDENTITY_LLM_REPEATS", "3"))))
TIMEOUT = max(30, int(os.getenv("RETRIEVAL_ANTHROPIC_TIMEOUT", "180")))
MAX_TOKENS = max(256, int(os.getenv("RETRIEVAL_ANTHROPIC_MAX_TOKENS", "700")))
EFFORT = os.getenv("RETRIEVAL_ANTHROPIC_EFFORT", "medium").strip().lower() or "medium"
# Claude Sonnet 5 rejects non-default sampling parameters. Do not send
# temperature/top_p/top_k; use the model default sampling configuration.
SAMPLING_MODE = "MODEL_DEFAULT_NO_TEMPERATURE_PARAMETER"
FORCE = os.getenv("RETRIEVAL_FORCE_LLM", "0").strip().lower() in {"1", "true", "yes", "on"}

PROMPT_VERSION = "study2-retrieval-cas-smiles-v2-isomer-scope-20261001"
SYSTEM_PROMPT = """You are performing regulatory identity retrieval for a controlled chemical benchmark.
You receive only candidate CAS identifier(s), PubChem-derived SMILES for those CAS identifiers,
and a fixed regulatory catalog that is the search space.
Find which single catalog target the candidate belongs to.
A catalog entry's isomer_scope states which stereoisomers of its reference substance are covered.
If the reference SMILES defines no stereochemistry, every stereoisomer is covered. Otherwise:
ALL_STEREOISOMERS covers every stereoisomer; EXACT_STEREO_ONLY covers only the stereochemistry
drawn in the reference SMILES; UNSPECIFIED_REVIEW means stereo-only differences cannot be decided.
Return NOT_FOUND when none of the catalog targets applies.
Return REVIEW only when the supplied CAS/SMILES/catalog information is insufficient to decide reliably.
Do not use or infer hidden benchmark labels. Do not browse or use external knowledge beyond the supplied fields.
The catalog itself is the allowed regulatory reference database.
Return only the requested JSON object."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "query_id": {"type": "string"},
        "status": {"type": "string", "enum": ["FOUND", "NOT_FOUND", "REVIEW"]},
        "target_id": {"type": "string"},
        "reason": {"type": "string"},
        "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
    },
    "required": ["query_id", "status", "target_id", "reason", "confidence"],
    "additionalProperties": False,
}


def clean(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() in {"nan", "none", "null", "<na>"} else s


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
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[m.start():])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    raise ValueError("No JSON object in model response")


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def cache_path(query_id: str, condition: str, repeat: int, prompt: str) -> Path:
    payload = "|".join([
        PROMPT_VERSION, MODEL, condition, str(repeat), query_id,
        prompt_hash(prompt), EFFORT, str(MAX_TOKENS), SAMPLING_MODE,
    ])
    key = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    d = CACHE_DIR / re.sub(r"[^A-Za-z0-9._-]+", "_", MODEL) / condition.lower()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"r{repeat}_{query_id}_{key}.json"


def call_anthropic(query_id: str, prompt: str) -> dict:
    key = api_key()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is missing")
    try:
        key.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError("ANTHROPIC_API_KEY contains non-ASCII characters") from exc

    body = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
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
    # Content-level failures (unparseable/truncated JSON) are resampled by the
    # retry loop; count them so selection effects can be reported per condition.
    unparseable = 0
    for attempt in range(1, 5):
        try:
            r = requests.post(ENDPOINT, headers=headers, json=body, timeout=TIMEOUT)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                if attempt < 4:
                    wait = min(30.0, 2 ** attempt)
                    time.sleep(wait)
                    continue
            if r.status_code >= 400:
                raise requests.HTTPError(
                    f"Anthropic HTTP {r.status_code}: {r.text[:800]}", response=r
                )
            payload = r.json()
            text = "".join(
                str(p.get("text", ""))
                for p in (payload.get("content") or [])
                if isinstance(p, dict) and p.get("type") == "text"
            ).strip()
            try:
                raw = extract_json(text)
            except ValueError:
                unparseable += 1
                raise ValueError(
                    f"No JSON object in model response (stop_reason={payload.get('stop_reason')})"
                )
            usage = payload.get("usage", {}) or {}
            raw["_model_returned"] = clean(payload.get("model")) or MODEL
            raw["_input_tokens"] = usage.get("input_tokens", "")
            raw["_output_tokens"] = usage.get("output_tokens", "")
            raw["_api_attempts"] = attempt
            raw["_unparseable_responses"] = unparseable
            raw["_stop_reason"] = clean(payload.get("stop_reason"))
            return raw
        except Exception as exc:
            last = exc
            if attempt < 4:
                time.sleep(min(16.0, 2 ** attempt))
    raise last if last is not None else RuntimeError("Anthropic request failed")


def normalize(raw: dict, query_id: str, allowed_target_ids: set[str]) -> dict:
    status = clean(raw.get("status")).upper()
    target = clean(raw.get("target_id"))
    if status not in {"FOUND", "NOT_FOUND", "REVIEW"}:
        status, target = "REVIEW", ""
    if status == "FOUND":
        if target not in allowed_target_ids:
            status, target = "REVIEW", ""
    else:
        target = ""
    return {
        "query_id": query_id,
        "status": status,
        "target_id": target,
        "reason": clean(raw.get("reason")),
        "confidence": clean(raw.get("confidence")).upper(),
        "model_returned": clean(raw.get("_model_returned")),
        "input_tokens": raw.get("_input_tokens", ""),
        "output_tokens": raw.get("_output_tokens", ""),
        "api_attempts": raw.get("_api_attempts", ""),
        "unparseable_responses": raw.get("_unparseable_responses", ""),
        "stop_reason": clean(raw.get("_stop_reason")),
    }


def run_one(
    query_id: str,
    prompt: str,
    condition: str,
    repeat: int,
    allowed_target_ids: set[str],
) -> dict:
    p = cache_path(query_id, condition, repeat, prompt)
    if p.exists() and not FORCE:
        try:
            saved = json.loads(p.read_text(encoding="utf-8"))
            if saved.get("call_status") == "OK":
                saved["runtime_source"] = "CACHE_HIT"
                return saved
        except Exception:
            pass

    t0 = time.perf_counter()
    try:
        raw = call_anthropic(query_id, prompt)
        out = normalize(raw, query_id, allowed_target_ids)
        out.update({
            "condition": condition,
            "repeat": repeat,
            "call_status": "OK",
            "elapsed_sec": float(time.perf_counter() - t0),
            "runtime_source": "FRESH_API_CALL",
        })
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return out
    except Exception as exc:
        return {
            "query_id": query_id,
            "status": "REVIEW",
            "target_id": "",
            "reason": "",
            "confidence": "",
            "model_returned": "",
            "input_tokens": "",
            "output_tokens": "",
            "condition": condition,
            "repeat": repeat,
            "call_status": f"LLM_ERROR:{type(exc).__name__}:{str(exc)[:300]}",
            "elapsed_sec": float(time.perf_counter() - t0),
            "runtime_source": "FRESH_API_CALL_ERROR",
        }
