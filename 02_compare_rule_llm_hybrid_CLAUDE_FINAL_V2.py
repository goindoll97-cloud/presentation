# -*- coding: utf-8 -*-
"""
02_compare_rule_llm_hybrid.py
=============================
Research-only pipeline 2/3. No web UI.

Compares, on the exact same fixed benchmark:
  1) RULE_ENGINE_ONLY  : conventional direct-CAS deterministic screening baseline
  2) LLM_ONLY          : semantic interpretation from regulatory text
  3) LLM_PLUS_RULE     : direct-CAS deterministic fact + LLM semantic interpretation

Primary endpoint
----------------
Detection of regulatory identity scope that requires information beyond direct CAS matching.
This is intentionally narrower than "final legal compliance accuracy".

Multi-capability LLM design
---------------------------
Supports low/mid local Ollama models and an optional strong cloud Anthropic Claude model.
The paper-facing question is not whether one fixed LLM is permanently inferior to rules,
but how the marginal contribution of deterministic rules changes as LLM capability increases.

Examples
--------
ollama pull qwen3.5:0.8b
python 02_compare_rule_llm_hybrid.py

Final capability design (default):
  LOW  = qwen3.5:0.8b (Ollama)
  MID  = qwen3.5:4b   (Ollama)
  HIGH = claude-sonnet-5 (Anthropic API, when ANTHROPIC_API_KEY is set)

Override local models if needed:
Windows PowerShell:
  $env:LOCAL_LLM_MODELS="qwen3.5:0.8b,qwen3.5:4b"
  $env:ANTHROPIC_MODELS="claude-sonnet-5"
  python 02_compare_rule_llm_hybrid_CLAUDE_FINAL.py

Optional:
  $env:LLM_REPEATS="3"
  $env:LLM_MAX_CASES="0"     # 0 = all primary benchmark rows
  $env:FORCE_LLM="1"         # ignore cache and rerun

Outputs are CSV/JSON only; script 03 turns them into publication tables and figures.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
INTERMEDIATE = ROOT / "intermediate"
INTERMEDIATE.mkdir(parents=True, exist_ok=True)

BENCHMARK_FILE = INTERMEDIATE / "01_cas_gap_benchmark_primary.csv"
FULL_BENCHMARK_FILE = INTERMEDIATE / "01_cas_gap_benchmark_full.csv"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip().rstrip("/")
# FINAL study default: low + mid local models.  Override with LOCAL_LLM_MODELS.
LOCAL_LLM_MODELS = [
    m.strip() for m in os.getenv("LOCAL_LLM_MODELS", "qwen3.5:0.8b,qwen3.5:4b").split(",") if m.strip()
]
# Strong cloud model. It is run only when ANTHROPIC_API_KEY is available and RUN_ANTHROPIC is not disabled.
ANTHROPIC_MODELS = [
    m.strip() for m in os.getenv("ANTHROPIC_MODELS", "claude-sonnet-5").split(",") if m.strip()
]
RUN_ANTHROPIC = os.getenv("RUN_ANTHROPIC", "auto").strip().lower()
N_REPEATS = max(1, int(os.getenv("LLM_REPEATS", "1")))
MAX_CASES = max(0, int(os.getenv("LLM_MAX_CASES", "0")))
FORCE_LLM = os.getenv("FORCE_LLM", "0").strip().lower() in {"1", "true", "yes", "on"}
# Rerun only the cloud arm when needed, while preserving expensive local-model caches.
FORCE_ANTHROPIC = os.getenv("FORCE_ANTHROPIC", "0").strip().lower() in {"1", "true", "yes", "on"}
TIMEOUT = max(30, int(os.getenv("OLLAMA_TIMEOUT", "180")))
ANTHROPIC_TIMEOUT = max(30, int(os.getenv("ANTHROPIC_TIMEOUT", "180")))
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
SEED = int(os.getenv("LLM_SEED", "42"))
MAX_SOURCE_CHARS = max(800, int(os.getenv("LLM_MAX_SOURCE_CHARS", "2400")))
NUM_CTX = max(1024, int(os.getenv("LLM_NUM_CTX", "2048")))
NUM_PREDICT = max(96, int(os.getenv("LLM_NUM_PREDICT", "180")))
LLM_WORKERS = max(1, int(os.getenv("LLM_WORKERS", "2")))
ANTHROPIC_WORKERS = max(1, int(os.getenv("ANTHROPIC_WORKERS", "2")))
ANTHROPIC_MAX_OUTPUT_TOKENS = max(256, int(os.getenv("ANTHROPIC_MAX_OUTPUT_TOKENS", "1024")))
ANTHROPIC_EFFORT = os.getenv("ANTHROPIC_EFFORT", "medium").strip().lower() or "medium"
PROMPT_VERSION = "cas-only-scope-refined-fast-v3-20260910"
VALIDATION_VERSION = "classification-evidence-separated-v2-20260910"
ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = os.getenv("ANTHROPIC_VERSION", "2023-06-01").strip()
# Cache discriminator for the cloud transport. Changing this forces only Claude to rerun,
# while preserving expensive Ollama caches.
ANTHROPIC_API_CACHE_VERSION = "messages-sonnet5-structured-v2-additionalpropsfalse-20260911"

ALLOWED_SCOPES = {
    "DIRECT_CAS_ONLY",
    "NO_DIRECT_CAS_OTHER",
    "BROAD_SALT",
    "COMPOUND_GROUP",
    "DERIVATIVE_FAMILY",
    "STRUCTURAL_RANGE",
    "MIXTURE_OR_UVCB",
    "REACTION_PRODUCT",
    "EXPLICIT_EXCEPTION",
    "MANUAL_REVIEW",
}

# Strongly constrain Claude's classification fields. Evidence is still audited
# separately as a literal-source grounding endpoint; it does NOT gate the binary
# classification endpoint.
ANTHROPIC_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "benchmark_id": {"type": "string"},
        "scope_type": {
            "type": "string",
            "enum": sorted(ALLOWED_SCOPES),
        },
        "requires_extended_identity": {"type": "boolean"},
        "evidence_quote": {
            "type": "string",
            "description": "A short literal substring copied exactly from source_text; empty only if no reliable span can be copied.",
        },
    },
    "required": ["benchmark_id", "scope_type", "requires_extended_identity", "evidence_quote"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """한국 화학물질 규제문서의 CAS-only 스크리닝 한계를 분류한다.
최종 법적 해당/비해당 판정을 하지 않는다. 입력된 물질명, direct CAS, source_text만 사용한다.
질문은 하나다: '이 규제범위를 direct CAS 완전일치만으로 안전하게 표현할 수 있는가?'

판정 원칙:
1) direct CAS가 없으면 requires_extended_identity=true.
2) direct CAS가 있더라도 원문이 'A와/과/및 그 염류', 'A and its/their salts',
   'A와 그 화합물/유도체', 총칭 범위 + 구체적 목록, 또는 명시적 제외조건을 제시하면 true.
3) 매우 중요: 물질명 자체에 염/ salt, 화합물/compound/compd., 유도체/derivative,
   중합체/polymer, 반응생성물/reaction product, 혼합물/mixture, C12~18 같은 구조범위가
   들어 있더라도 그 특정 물질에 고유 direct CAS가 있고 원문이 그 CAS를 넘어서는 포괄범위를
   명시하지 않으면 DIRECT_CAS_ONLY, requires_extended_identity=false이다.
4) 즉 '화학적으로 복잡한 이름'과 'CAS-only 규제 식별 공백'을 혼동하지 않는다.
5) 근거가 애매하면 MANUAL_REVIEW로 답한다. 단, 근거 없는 추측으로 true를 만들지 않는다.
evidence_quote는 source_text에 실제로 연속해서 존재하는 짧은 문구만 쓴다.
반드시 JSON 객체 하나만 출력한다."""


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def is_loopback(url: str) -> bool:
    p = urlparse(url)
    return p.scheme in {"http", "https"} and p.hostname in {"127.0.0.1", "localhost", "::1"}


def ollama_status() -> dict:
    if not is_loopback(OLLAMA_BASE_URL):
        return {"ready": False, "status": "REMOTE_URL_BLOCKED", "installed_models": []}
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
        payload = r.json()
        names = [clean(x.get("name")) for x in payload.get("models", []) if isinstance(x, dict)]
        return {"ready": True, "status": "OLLAMA_READY", "installed_models": names}
    except Exception as exc:
        return {"ready": False, "status": "OLLAMA_OFFLINE", "installed_models": [], "message": f"{type(exc).__name__}: {exc}"}


def _load_dotenv_key(name: str) -> str:
    if os.getenv(name):
        return os.getenv(name, "").strip()
    for env_path in (ROOT / ".env", Path.cwd() / ".env"):
        if not env_path.exists():
            continue
        try:
            for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == name:
                    value = v.strip().strip('"').strip("'")
                    if value:
                        os.environ.setdefault(name, value)
                        return value
        except OSError:
            pass
    return ""


def anthropic_api_key() -> str:
    return _load_dotenv_key("ANTHROPIC_API_KEY")

def infer_capability(model: str, backend: str) -> tuple[str, int, float]:
    """Ordinal capability tier for the study; not claimed as a universal model ranking."""
    s = model.lower()
    if backend == "anthropic":
        return "HIGH", 3, np.nan
    m = re.search(r"(?<!\d)(\d+(?:\.\d+)?)b\b", s)
    size_b = float(m.group(1)) if m else np.nan
    if np.isfinite(size_b) and size_b <= 1.5:
        return "LOW", 1, size_b
    if np.isfinite(size_b) and size_b <= 6:
        return "MID", 2, size_b
    if np.isfinite(size_b):
        return "MID_HIGH", 2, size_b
    return "UNSPECIFIED", 2, size_b

def build_model_specs() -> list[dict]:
    specs = []
    for model in LOCAL_LLM_MODELS:
        tier, order, size_b = infer_capability(model, "ollama")
        specs.append({
            "llm_model": model, "llm_backend": "ollama", "capability_tier": tier,
            "capability_order": order, "parameter_size_b": size_b, "model_family": model.split(":")[0],
        })
    cloud_requested = RUN_ANTHROPIC not in {"0", "false", "no", "off"}
    if cloud_requested:
        for model in ANTHROPIC_MODELS:
            tier, order, size_b = infer_capability(model, "anthropic")
            specs.append({
                "llm_model": model, "llm_backend": "anthropic", "capability_tier": tier,
                "capability_order": order, "parameter_size_b": size_b, "model_family": "Claude",
            })
    return specs

def anthropic_status() -> dict:
    key = anthropic_api_key()
    if RUN_ANTHROPIC in {"0", "false", "no", "off"}:
        return {"ready": False, "status": "ANTHROPIC_DISABLED"}
    if not key:
        return {"ready": False, "status": "ANTHROPIC_API_KEY_MISSING"}
    return {"ready": True, "status": "ANTHROPIC_CONFIGURED"}

def model_installed(model: str, installed: list[str]) -> bool:
    # If a tag is explicitly requested (e.g., 0.8b vs 4b), require that exact tag.
    if model in installed:
        return True
    if ":" in model:
        return False
    wanted_base = model.split(":")[0]
    return any(x.split(":")[0] == wanted_base for x in installed)


def extract_json(content: Any) -> dict:
    if isinstance(content, dict):
        return content
    text = clean(content)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        val = json.loads(text)
        return val if isinstance(val, dict) else {}
    except Exception:
        decoder = json.JSONDecoder()
        for m in re.finditer(r"\{", text):
            try:
                val, _ = decoder.raw_decode(text[m.start():])
                if isinstance(val, dict):
                    return val
            except Exception:
                continue
    raise ValueError("No valid JSON object in LLM response")


def compact_source(value: Any) -> str:
    """Keep both beginning and end of long regulatory text.

    Exclusion clauses often appear near the end, so simple head truncation can
    create false-safe classifications.
    """
    text = clean(value)
    if len(text) <= MAX_SOURCE_CHARS:
        return text
    tail = max(400, MAX_SOURCE_CHARS // 3)
    head = MAX_SOURCE_CHARS - tail
    return text[:head].rstrip() + " …[중간 생략]… " + text[-tail:].lstrip()


def case_prompt(row: pd.Series) -> str:
    payload = {
        "benchmark_id": clean(row.get("benchmark_id")),
        "substance_name_ko": clean(row.get("substance_name_ko")),
        "direct_cas": clean(row.get("direct_cas")),
        "source_text": compact_source(row.get("source_text")),
    }
    schema_hint = {
        "benchmark_id": payload["benchmark_id"],
        "scope_type": "DIRECT_CAS_ONLY|NO_DIRECT_CAS_OTHER|BROAD_SALT|COMPOUND_GROUP|DERIVATIVE_FAMILY|STRUCTURAL_RANGE|MIXTURE_OR_UVCB|REACTION_PRODUCT|EXPLICIT_EXCEPTION|MANUAL_REVIEW",
        "requires_extended_identity": True,
        "evidence_quote": "source_text의 짧은 원문 근거",
    }
    return (
        "다음 사례를 분석하라. 아래 출력 형식의 JSON 객체 하나만 반환하라.\n"
        f"입력={json.dumps(payload, ensure_ascii=False)}\n"
        f"출력형식={json.dumps(schema_hint, ensure_ascii=False)}"
    )


def call_ollama(row: pd.Series, model: str) -> dict:
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": case_prompt(row)},
            ],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {
                "temperature": TEMPERATURE,
                "seed": SEED,
                "num_predict": NUM_PREDICT,
                "num_ctx": NUM_CTX,
            },
            "keep_alive": "10m",
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    payload = r.json()
    raw = extract_json(payload.get("message", {}).get("content", ""))
    raw["_ollama_model_returned"] = clean(payload.get("model")) or model
    raw["_duration_ns"] = payload.get("total_duration", "")
    return raw



def call_anthropic(row: pd.Series, model: str) -> dict:
    """Call Anthropic Messages API with structured JSON output.

    Claude Sonnet 5 has adaptive thinking on by default. For this short, fixed
    classification benchmark we explicitly disable thinking and use medium effort
    to align more closely with the local models' think=False condition and to
    reduce cost/latency. Classification validity and literal quote validity remain
    separate endpoints.
    """
    key = anthropic_api_key()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured")

    body = {
        "model": model,
        "max_tokens": ANTHROPIC_MAX_OUTPUT_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": case_prompt(row)}],
        "thinking": {"type": "disabled"},
        "output_config": {
            "effort": ANTHROPIC_EFFORT,
            "format": {
                "type": "json_schema",
                "schema": ANTHROPIC_RESPONSE_SCHEMA,
            },
        },
    }

    headers = {
        "x-api-key": key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    last_exc = None
    for attempt in range(1, 5):
        try:
            r = requests.post(ANTHROPIC_ENDPOINT, headers=headers, json=body, timeout=ANTHROPIC_TIMEOUT)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                retry_after = r.headers.get("retry-after", "")
                try:
                    wait_s = max(float(retry_after), min(30.0, 2.0 ** attempt)) if retry_after else min(30.0, 2.0 ** attempt)
                except Exception:
                    wait_s = min(30.0, 2.0 ** attempt)
                if attempt < 4:
                    time.sleep(wait_s)
                    continue
                raise requests.HTTPError(f"Anthropic HTTP {r.status_code}: {r.text[:500]}", response=r)
            if r.status_code >= 400:
                raise requests.HTTPError(f"Anthropic HTTP {r.status_code}: {r.text[:800]}", response=r)

            payload = r.json()
            if clean(payload.get("stop_reason")).lower() == "refusal":
                raise ValueError(f"Anthropic refusal: {str(payload)[:600]}")

            texts = []
            for part in payload.get("content", []) or []:
                if isinstance(part, dict) and part.get("type") == "text" and part.get("text") is not None:
                    texts.append(str(part.get("text")))
            content = "".join(texts).strip()
            if not content:
                raise ValueError(f"Anthropic response contains no model text: {str(payload)[:700]}")

            raw = extract_json(content)
            raw["_ollama_model_returned"] = clean(payload.get("model")) or model
            raw["_duration_ns"] = ""
            usage = payload.get("usage", {}) or {}
            raw["_input_tokens"] = usage.get("input_tokens", "")
            raw["_output_tokens"] = usage.get("output_tokens", "")
            raw["_total_tokens"] = (
                (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
                if isinstance(usage.get("input_tokens"), (int, float)) and isinstance(usage.get("output_tokens"), (int, float))
                else ""
            )
            return raw
        except Exception as exc:
            last_exc = exc
            if attempt < 4:
                time.sleep(min(16, 2 ** attempt))

    raise last_exc if last_exc is not None else RuntimeError("Anthropic call failed")

def validate_llm_output(raw: dict, row: pd.Series, model: str, repeat: int) -> dict:
    """Validate the semantic classification and evidence grounding separately.

    Methodological rationale:
    - Main Rule-vs-LLM-vs-Hybrid performance asks whether the model identified the
      CAS-only scope correctly. A model should not lose its classification merely
      because it normalized punctuation/spacing in a supporting quote.
    - Literal evidence grounding is still important for regulatory trust, so exact
      source-span agreement is retained as a separate audit endpoint.
    """
    source = clean(row.get("source_text"))
    raw_scope = clean(raw.get("scope_type")).upper()
    classification_valid = raw_scope in ALLOWED_SCOPES and raw_scope != "MANUAL_REVIEW"
    scope = raw_scope if raw_scope in ALLOWED_SCOPES else "MANUAL_REVIEW"

    requires = raw.get("requires_extended_identity")
    if isinstance(requires, str):
        requires = requires.strip().lower() in {"1", "true", "yes", "y"}
    elif not isinstance(requires, (bool, np.bool_)):
        requires = scope != "DIRECT_CAS_ONLY"
    requires = bool(requires)

    # Conservative internal consistency gate.
    if scope == "DIRECT_CAS_ONLY":
        requires = False
    elif scope in ALLOWED_SCOPES - {"DIRECT_CAS_ONLY", "MANUAL_REVIEW"}:
        requires = True

    quote = clean(raw.get("evidence_quote"))
    quote_valid = bool(quote and quote in source)
    evidence_status = "EVIDENCE_EXACT_MATCH" if quote_valid else "EVIDENCE_QUOTE_INVALID"
    status = "OK" if classification_valid else "MANUAL_REVIEW"

    conf = pd.to_numeric(pd.Series([raw.get("confidence")]), errors="coerce").iloc[0]
    conf = float(conf) if pd.notna(conf) else np.nan
    if np.isfinite(conf):
        conf = min(1.0, max(0.0, conf))

    return {
        "benchmark_id": clean(row.get("benchmark_id")),
        "llm_model": model,
        "repeat": int(repeat),
        "llm_status": status,
        "llm_classification_valid": bool(classification_valid),
        "llm_scope_type": scope,
        "llm_requires_extended_identity": requires,
        "llm_evidence_quote": quote if quote_valid else quote,
        "llm_evidence_quote_valid": bool(quote_valid),
        "llm_evidence_status": evidence_status,
        "llm_reason_ko": clean(raw.get("reason_ko")),
        "llm_confidence": conf,
        "ollama_model_returned": clean(raw.get("_ollama_model_returned")),
        "duration_ns": raw.get("_duration_ns", ""),
    }


def upgrade_cached_output(cached: dict) -> dict:
    """Upgrade v1 caches without rerunning models.

    Legacy EVIDENCE_QUOTE_INVALID rows retain a usable semantic classification.
    LLM_ERROR and MANUAL_REVIEW remain unavailable. This preserves old local-model
    work while removing the inappropriate literal-quote gate from the main endpoint.
    """
    status = clean(cached.get("llm_status")).upper()
    scope = clean(cached.get("llm_scope_type")).upper()
    classification_valid = (
        status != "LLM_ERROR"
        and scope in ALLOWED_SCOPES
        and scope != "MANUAL_REVIEW"
    )
    quote_valid = bool(cached.get("llm_evidence_quote_valid", False))
    cached["llm_classification_valid"] = bool(classification_valid)
    cached["llm_evidence_status"] = "EVIDENCE_EXACT_MATCH" if quote_valid else "EVIDENCE_QUOTE_INVALID"
    if classification_valid:
        cached["llm_status"] = "OK"
    elif status != "LLM_ERROR":
        cached["llm_status"] = "MANUAL_REVIEW"
    cached["validation_version"] = VALIDATION_VERSION
    return cached


def cache_path(spec: dict, repeat: int, row: pd.Series) -> Path:
    model = spec["llm_model"]
    backend = spec["llm_backend"]
    if backend == "ollama":
        # EXACT legacy key/path from REFINED_FAST so already completed 0.8B cases are reused.
        token = "|".join([
            model, str(repeat), clean(row.get("benchmark_id")), clean(row.get("source_text_sha256_16")),
            PROMPT_VERSION, str(MAX_SOURCE_CHARS), str(NUM_CTX), str(NUM_PREDICT),
            hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12],
        ])
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    else:
        token = "|".join([
            backend, model, str(repeat), clean(row.get("benchmark_id")), clean(row.get("source_text_sha256_16")),
            clean(row.get("reference_rule_version")), PROMPT_VERSION, str(MAX_SOURCE_CHARS),
            str(ANTHROPIC_MAX_OUTPUT_TOKENS), ANTHROPIC_EFFORT, ANTHROPIC_API_CACHE_VERSION,
            hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12],
        ])
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{backend}_{model}")
    key = hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]
    d = INTERMEDIATE / "llm_cache" / safe_model
    d.mkdir(parents=True, exist_ok=True)
    return d / f"r{repeat}_{key}.json"

def run_llm_case(row: pd.Series, spec: dict, repeat: int) -> dict:
    model = spec["llm_model"]
    backend = spec["llm_backend"]
    cp = cache_path(spec, repeat, row)
    force_this = FORCE_LLM or (backend == "anthropic" and FORCE_ANTHROPIC)
    if cp.exists() and not force_this:
        try:
            cached = json.loads(cp.read_text(encoding="utf-8"))
            cached = upgrade_cached_output(cached)
            cached.update({
                "llm_backend": backend,
                "capability_tier": spec["capability_tier"],
                "capability_order": spec["capability_order"],
                "parameter_size_b": spec["parameter_size_b"],
                "model_family": spec["model_family"],
                "prompt_version": PROMPT_VERSION,
                "reference_rule_version": clean(row.get("reference_rule_version")),
            })
            return cached
        except Exception:
            pass
    try:
        raw = call_ollama(row, model) if backend == "ollama" else call_anthropic(row, model)
        out = validate_llm_output(raw, row, model, repeat)
        out["input_tokens"] = raw.get("_input_tokens", "")
        out["output_tokens"] = raw.get("_output_tokens", "")
        out["total_tokens"] = raw.get("_total_tokens", "")
    except Exception as exc:
        out = {
            "benchmark_id": clean(row.get("benchmark_id")),
            "llm_model": model,
            "repeat": int(repeat),
            "llm_status": "LLM_ERROR",
            "llm_classification_valid": False,
            "llm_scope_type": "MANUAL_REVIEW",
            "llm_requires_extended_identity": True,
            "llm_evidence_quote": "",
            "llm_evidence_quote_valid": False,
            "llm_evidence_status": "EVIDENCE_UNAVAILABLE",
            "llm_reason_ko": f"{type(exc).__name__}: {backend} LLM call failed",
            "llm_confidence": np.nan,
            "ollama_model_returned": model,
            "duration_ns": "",
            "input_tokens": "", "output_tokens": "", "total_tokens": "",
        }
    out.update({
        "llm_backend": backend,
        "capability_tier": spec["capability_tier"],
        "capability_order": spec["capability_order"],
        "parameter_size_b": spec["parameter_size_b"],
        "model_family": spec["model_family"],
        "prompt_version": PROMPT_VERSION,
        "reference_rule_version": clean(row.get("reference_rule_version")),
        "validation_version": VALIDATION_VERSION,
    })
    # Do not persist transient backend/API failures. Otherwise a temporary 4xx/5xx
    # can poison all later reruns by being treated as a completed cache entry.
    if clean(out.get("llm_status")).upper() != "LLM_ERROR":
        try:
            cp.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass
    return out

def binary_metrics(pred: pd.Series, gold: pd.Series) -> dict:
    p = pred.astype(bool)
    g = gold.astype(bool)
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    tn = int((~p & ~g).sum())
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and precision + recall else np.nan
    return {
        "n": n, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy": (tp + tn) / n if n else np.nan,
        "precision": precision, "recall": recall, "specificity": specificity, "f1": f1,
        "false_safe_rate": fn / (tp + fn) if tp + fn else np.nan,
        "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
    }


def exact_mcnemar_p(a_correct: pd.Series, b_correct: pd.Series) -> tuple[int, int, float]:
    # Exact two-sided McNemar via Binomial(n, 0.5), no scipy dependency.
    a = a_correct.astype(bool).to_numpy()
    b = b_correct.astype(bool).to_numpy()
    b_only = int((~a & b).sum())
    a_only = int((a & ~b).sum())
    n = b_only + a_only
    if n == 0:
        return a_only, b_only, 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return a_only, b_only, min(1.0, 2 * tail)



def build_system_predictions(bench: pd.DataFrame, llm_results: pd.DataFrame) -> pd.DataFrame:
    """
    Build three systems on matched model/repeat strata.

    Main comparison uses `common_case_available=True`, i.e. the same benchmark rows
    for Rule-only, LLM-only and Hybrid. This prevents Rule-only from being evaluated
    on 100% of rows while LLM systems are evaluated only on valid-output rows.
    """
    parts = []
    gold_col = "reference_cas_only_insufficient" if "reference_cas_only_insufficient" in bench.columns else "reference_requires_extended_identity"

    if llm_results is None or llm_results.empty:
        r = bench.copy()
        r["system"] = "RULE_ENGINE_ONLY"
        r["llm_model"] = "NONE"
        r["repeat"] = 1
        r["prediction_available"] = True
        r["common_case_available"] = True
        r["pred_requires_extended_identity"] = ~r["has_direct_cas"].astype(bool)
        r["pred_scope_type"] = np.where(r["has_direct_cas"].astype(bool), "DIRECT_CAS_ONLY", "NO_DIRECT_CAS_OTHER")
        r["prediction_basis"] = "DIRECT_CAS_PRESENCE_ONLY"
        parts.append(r)
    else:
        for (model, repeat), lr in llm_results.groupby(["llm_model", "repeat"], dropna=False):
            merged = bench.merge(lr, on="benchmark_id", how="left")
            common = merged.get("llm_classification_valid", pd.Series(False, index=merged.index)).fillna(False).astype(bool)

            # 1) Conventional CAS-only deterministic baseline, replicated on the
            # same model/repeat stratum solely to enforce identical comparison rows.
            r = merged.copy()
            r["system"] = "RULE_ENGINE_ONLY"
            r["llm_model"] = model
            r["repeat"] = int(repeat)
            r["prediction_available"] = True
            r["common_case_available"] = common
            r["pred_requires_extended_identity"] = ~r["has_direct_cas"].astype(bool)
            r["pred_scope_type"] = np.where(r["has_direct_cas"].astype(bool), "DIRECT_CAS_ONLY", "NO_DIRECT_CAS_OTHER")
            r["prediction_basis"] = "DIRECT_CAS_PRESENCE_ONLY"
            parts.append(r)

            # 2) LLM-only
            z = merged.copy()
            z["system"] = "LLM_ONLY"
            z["llm_model"] = model
            z["repeat"] = int(repeat)
            z["prediction_available"] = common
            z["common_case_available"] = common
            z["pred_requires_extended_identity"] = z["llm_requires_extended_identity"].fillna(True).astype(bool)
            z["pred_scope_type"] = z["llm_scope_type"].fillna("MANUAL_REVIEW")
            z["prediction_basis"] = "LLM_SEMANTIC_SCOPE"
            parts.append(z)

            # 3) Hybrid: deterministic direct-CAS absence OR LLM semantic extension.
            h = merged.copy()
            h["system"] = "LLM_PLUS_RULE"
            h["llm_model"] = model
            h["repeat"] = int(repeat)
            h["prediction_available"] = common
            h["common_case_available"] = common
            h["pred_requires_extended_identity"] = (
                (~h["has_direct_cas"].astype(bool))
                | h["llm_requires_extended_identity"].fillna(True).astype(bool)
            )
            h["pred_scope_type"] = h["llm_scope_type"].fillna("MANUAL_REVIEW")
            h["prediction_basis"] = "DIRECT_CAS_FACT_PLUS_LLM_SEMANTIC_SCOPE"
            parts.append(h)

    out = pd.concat(parts, ignore_index=True, sort=False)
    out["gold_requires_extended_identity"] = out[gold_col].astype(bool)
    out["correct"] = (
        out["prediction_available"].astype(bool)
        & out["pred_requires_extended_identity"].astype(bool).eq(out["gold_requires_extended_identity"])
    )
    out["false_safe"] = (
        out["prediction_available"].astype(bool)
        & out["gold_requires_extended_identity"]
        & ~out["pred_requires_extended_identity"].astype(bool)
    )
    out["false_positive"] = (
        out["prediction_available"].astype(bool)
        & ~out["gold_requires_extended_identity"]
        & out["pred_requires_extended_identity"].astype(bool)
    )
    return out


def summarize_performance(pred: pd.DataFrame) -> pd.DataFrame:
    """
    Paper-facing performance is calculated only on the COMMON analysis set:
    rows with a validated LLM output, identically for all three systems.
    Native system coverage is reported separately.
    """
    rows = []
    for (system, model, repeat), g in pred.groupby(["system", "llm_model", "repeat"], dropna=False):
        common = g[g["common_case_available"].astype(bool)].copy()
        m = (
            binary_metrics(common["pred_requires_extended_identity"], common["gold_requires_extended_identity"])
            if len(common)
            else {k: np.nan for k in ["n","TP","FP","FN","TN","accuracy","precision","recall","specificity","f1","false_safe_rate","false_positive_rate"]}
        )
        native_avail = g["prediction_available"].astype(bool)
        positives = g["gold_requires_extended_identity"].astype(bool)
        positive_native_coverage = float(g.loc[positives, "prediction_available"].mean()) if positives.any() else np.nan
        rows.append({
            "system": system,
            "llm_model": model,
            "llm_backend": clean(g.get("llm_backend", pd.Series([""])).iloc[0]) if len(g) else "",
            "capability_tier": clean(g.get("capability_tier", pd.Series([""])).iloc[0]) if len(g) else "",
            "capability_order": pd.to_numeric(pd.Series([g.get("capability_order", pd.Series([np.nan])).iloc[0] if len(g) else np.nan]), errors="coerce").iloc[0],
            "parameter_size_b": pd.to_numeric(pd.Series([g.get("parameter_size_b", pd.Series([np.nan])).iloc[0] if len(g) else np.nan]), errors="coerce").iloc[0],
            "repeat": int(repeat),
            "n_total": int(len(g)),
            "n_common": int(len(common)),
            "common_analysis_coverage": float(len(common) / len(g)) if len(g) else np.nan,
            "system_output_coverage": float(native_avail.mean()) if len(g) else np.nan,
            "positive_case_output_coverage": positive_native_coverage,
            **m,
            "analysis_set": "COMMON_VALID_LLM_OUTPUT_ROWS",
            "reference_type": "SOURCE_DERIVED_PREDEFINED_OPERATIONAL_REFERENCE_NOT_EXPERT_GOLD",
        })
    return pd.DataFrame(rows)


def summarize_by_scope(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in pred.groupby(["system", "llm_model", "repeat", "reference_scope_type"], dropna=False):
        system, model, repeat, scope = keys
        common = g[g["common_case_available"].astype(bool)].copy()
        m = binary_metrics(common["pred_requires_extended_identity"], common["gold_requires_extended_identity"]) if len(common) else {}
        rows.append({
            "system": system,
            "llm_model": model,
            "repeat": int(repeat),
            "reference_scope_type": scope,
            "n_total": int(len(g)),
            "n_common": int(len(common)),
            "common_coverage": float(len(common)/len(g)) if len(g) else np.nan,
            **m,
        })
    return pd.DataFrame(rows)


def rule_full_benchmark_performance(bench: pd.DataFrame) -> pd.DataFrame:
    """Supplementary descriptive performance of the CAS-only rule on all benchmark rows."""
    gold_col = "reference_cas_only_insufficient" if "reference_cas_only_insufficient" in bench.columns else "reference_requires_extended_identity"
    pred = ~bench["has_direct_cas"].astype(bool)
    m = binary_metrics(pred, bench[gold_col].astype(bool))
    return pd.DataFrame([{
        "system": "RULE_ENGINE_ONLY",
        "analysis_set": "FULL_PRIMARY_BENCHMARK_SUPPLEMENTARY",
        "n_total": int(len(bench)),
        **m,
    }])

def marginal_benefit(perf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    llm = perf[perf["system"].eq("LLM_ONLY")]
    hyb = perf[perf["system"].eq("LLM_PLUS_RULE")]
    for _, a in llm.iterrows():
        b = hyb[(hyb["llm_model"] == a["llm_model"]) & (hyb["repeat"] == a["repeat"])]
        if b.empty:
            continue
        b = b.iloc[0]
        rows.append({
            "llm_model": a["llm_model"],
            "llm_backend": a.get("llm_backend", ""),
            "capability_tier": a.get("capability_tier", ""),
            "capability_order": a.get("capability_order", np.nan),
            "parameter_size_b": a.get("parameter_size_b", np.nan),
            "repeat": int(a["repeat"]),
            "delta_accuracy_hybrid_minus_llm": b.get("accuracy", np.nan) - a.get("accuracy", np.nan),
            "delta_recall_hybrid_minus_llm": b.get("recall", np.nan) - a.get("recall", np.nan),
            "delta_precision_hybrid_minus_llm": b.get("precision", np.nan) - a.get("precision", np.nan),
            "delta_false_safe_hybrid_minus_llm": b.get("false_safe_rate", np.nan) - a.get("false_safe_rate", np.nan),
            "delta_coverage_hybrid_minus_llm": b.get("system_output_coverage", np.nan) - a.get("system_output_coverage", np.nan),
        })
    return pd.DataFrame(rows)




def cross_model_common_ids(llm_results: pd.DataFrame) -> set[str]:
    """Cases with valid output from every evaluated model/repeat stratum.

    This is the strict analysis set for comparing capability levels.  It prevents a
    stronger model from looking better merely because it returned valid output on an
    easier subset than another model.
    """
    if llm_results is None or llm_results.empty:
        return set()
    strata = llm_results[["llm_model", "repeat"]].drop_duplicates()
    if strata.empty:
        return set()
    valid_sets = []
    for _, s in strata.iterrows():
        g = llm_results[(llm_results["llm_model"] == s["llm_model"]) & (llm_results["repeat"] == s["repeat"])]
        valid_sets.append(set(g.loc[g.get("llm_classification_valid", pd.Series(False, index=g.index)).fillna(False).astype(bool), "benchmark_id"].astype(str)))
    return set.intersection(*valid_sets) if valid_sets else set()


def capability_common_performance(pred: pd.DataFrame, common_ids: set[str]) -> pd.DataFrame:
    if pred is None or pred.empty or not common_ids:
        return pd.DataFrame()
    x = pred[pred["benchmark_id"].astype(str).isin(common_ids)].copy()
    rows = []
    for (system, model, repeat), g in x.groupby(["system", "llm_model", "repeat"], dropna=False):
        if system != "RULE_ENGINE_ONLY" and not g["prediction_available"].astype(bool).all():
            continue
        m = binary_metrics(g["pred_requires_extended_identity"], g["gold_requires_extended_identity"])
        rows.append({
            "system": system, "llm_model": model, "repeat": int(repeat),
            "llm_backend": clean(g.get("llm_backend", pd.Series([""])).iloc[0]) if len(g) else "",
            "capability_tier": clean(g.get("capability_tier", pd.Series([""])).iloc[0]) if len(g) else "",
            "capability_order": pd.to_numeric(pd.Series([g.get("capability_order", pd.Series([np.nan])).iloc[0] if len(g) else np.nan]), errors="coerce").iloc[0],
            "parameter_size_b": pd.to_numeric(pd.Series([g.get("parameter_size_b", pd.Series([np.nan])).iloc[0] if len(g) else np.nan]), errors="coerce").iloc[0],
            "n_cross_model_common": int(len(g)), **m,
            "analysis_set": "CROSS_MODEL_COMMON_VALID_OUTPUT_ROWS",
        })
    return pd.DataFrame(rows)


def capability_rule_benefit(cap_perf: pd.DataFrame) -> pd.DataFrame:
    if cap_perf is None or cap_perf.empty:
        return pd.DataFrame()
    rows = []
    for (model, repeat), g in cap_perf.groupby(["llm_model", "repeat"], dropna=False):
        llm = g[g["system"].eq("LLM_ONLY")]
        hyb = g[g["system"].eq("LLM_PLUS_RULE")]
        rule = g[g["system"].eq("RULE_ENGINE_ONLY")]
        if llm.empty or hyb.empty or rule.empty:
            continue
        a, b, r = llm.iloc[0], hyb.iloc[0], rule.iloc[0]
        rows.append({
            "llm_model": model, "repeat": int(repeat),
            "llm_backend": a.get("llm_backend", ""),
            "capability_tier": a.get("capability_tier", ""),
            "capability_order": a.get("capability_order", np.nan),
            "parameter_size_b": a.get("parameter_size_b", np.nan),
            "n_cross_model_common": a.get("n_cross_model_common", np.nan),
            "llm_recall": a.get("recall", np.nan),
            "rule_recall": r.get("recall", np.nan),
            "hybrid_recall": b.get("recall", np.nan),
            "delta_rule_benefit_recall": b.get("recall", np.nan) - a.get("recall", np.nan),
            "delta_rule_benefit_accuracy": b.get("accuracy", np.nan) - a.get("accuracy", np.nan),
            "delta_rule_benefit_false_safe": b.get("false_safe_rate", np.nan) - a.get("false_safe_rate", np.nan),
            "delta_llm_minus_rule_recall": a.get("recall", np.nan) - r.get("recall", np.nan),
            "delta_hybrid_minus_rule_recall": b.get("recall", np.nan) - r.get("recall", np.nan),
        })
    return pd.DataFrame(rows).sort_values(["capability_order", "llm_model", "repeat"], kind="stable")


def paired_tests(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    systems = ["RULE_ENGINE_ONLY", "LLM_ONLY", "LLM_PLUS_RULE"]
    comparisons = [
        ("LLM_PLUS_RULE", "LLM_ONLY"),
        ("LLM_ONLY", "RULE_ENGINE_ONLY"),
        ("LLM_PLUS_RULE", "RULE_ENGINE_ONLY"),
    ]
    models = sorted(set(pred.loc[pred["llm_model"].astype(str).ne("NONE"), "llm_model"].dropna().astype(str)))
    for model in models:
        repeats = sorted(pred.loc[pred["llm_model"].eq(model), "repeat"].dropna().astype(int).unique())
        for repeat in repeats:
            for a_sys, b_sys in comparisons:
                a = pred[(pred["system"] == a_sys) & (pred["llm_model"] == model) & (pred["repeat"] == repeat)]
                b = pred[(pred["system"] == b_sys) & (pred["llm_model"] == model) & (pred["repeat"] == repeat)]
                pair = a[["benchmark_id", "common_case_available", "correct"]].merge(
                    b[["benchmark_id", "common_case_available", "correct"]],
                    on="benchmark_id", suffixes=("_a", "_b")
                )
                pair = pair[pair["common_case_available_a"] & pair["common_case_available_b"]]
                if not len(pair):
                    continue
                b_only, a_only, p = exact_mcnemar_p(pair["correct_b"], pair["correct_a"])
                rows.append({
                    "comparison": f"{a_sys}_vs_{b_sys}",
                    "llm_model": model,
                    "repeat": repeat,
                    "n_pairs": int(len(pair)),
                    f"{b_sys}_only_correct": int(b_only),
                    f"{a_sys}_only_correct": int(a_only),
                    "mcnemar_exact_two_sided_p": p,
                })
    return pd.DataFrame(rows)

def repeat_consistency(llm_results: pd.DataFrame) -> pd.DataFrame:
    if llm_results is None or llm_results.empty or N_REPEATS < 2:
        return pd.DataFrame()
    rows = []
    for model, g in llm_results.groupby("llm_model"):
        piv = g[g.get("llm_classification_valid", pd.Series(False, index=g.index)).fillna(False).astype(bool)].pivot_table(
            index="benchmark_id", columns="repeat", values="llm_requires_extended_identity", aggfunc="first"
        )
        if piv.empty:
            continue
        valid = piv.dropna()
        agreement = valid.nunique(axis=1).eq(1)
        rows.append({
            "llm_model": model, "n_cases_with_all_repeats": int(len(valid)),
            "repeat_decision_consistency": float(agreement.mean()) if len(valid) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 80)
    print("02 / RULE ENGINE vs LLM vs HYBRID")
    print("=" * 80)
    if not BENCHMARK_FILE.exists():
        raise FileNotFoundError(f"Run script 01 first: {BENCHMARK_FILE}")

    bench = pd.read_csv(BENCHMARK_FILE, encoding="utf-8-sig", low_memory=False)
    if MAX_CASES > 0:
        bench = bench.head(MAX_CASES).copy()
    bench["has_direct_cas"] = bench["has_direct_cas"].astype(str).str.lower().isin(["true", "1", "yes"])
    bench["reference_requires_extended_identity"] = bench["reference_requires_extended_identity"].astype(str).str.lower().isin(["true", "1", "yes"])
    if "reference_cas_only_insufficient" in bench.columns:
        bench["reference_cas_only_insufficient"] = bench["reference_cas_only_insufficient"].astype(str).str.lower().isin(["true", "1", "yes"])
    print(f"Primary benchmark rows: {len(bench):,}")

    ollama = ollama_status()
    anthropic = anthropic_status()
    installed = ollama.get("installed_models", [])
    specs = build_model_specs()
    print(f"Ollama: {ollama.get('status')} | installed={installed}")
    print(f"Anthropic: {anthropic.get('status')}")
    print("Model plan:")
    for spec in specs:
        print(f"  - {spec['capability_tier']:>8} | {spec['llm_backend']:<6} | {spec['llm_model']}")

    llm_rows = []
    model_status_rows = []
    for spec in specs:
        model = spec["llm_model"]
        backend = spec["llm_backend"]
        if backend == "ollama":
            ready = ollama.get("ready", False) and model_installed(model, installed)
            backend_status = ollama.get("status") if ready else "MODEL_NOT_INSTALLED_OR_OLLAMA_OFFLINE"
            workers = LLM_WORKERS
        else:
            ready = anthropic.get("ready", False)
            backend_status = anthropic.get("status")
            workers = ANTHROPIC_WORKERS
        model_status_rows.append({**spec, "backend_status": backend_status, "model_available": bool(ready), "requested_repeats": N_REPEATS})
        if not ready:
            print(f"[SKIP] {backend}:{model} -> {backend_status}")
            continue

        if backend == "anthropic" and len(bench):
            print(f"[PREFLIGHT] {backend}:{model} API/schema check ...")
            try:
                probe = call_anthropic(bench.iloc[0], model)
                probe_scope = clean(probe.get("scope_type")).upper()
                if probe_scope not in ALLOWED_SCOPES:
                    raise ValueError(f"structured output scope_type invalid: {probe_scope!r}")
                if "requires_extended_identity" not in probe:
                    raise ValueError("structured output missing requires_extended_identity")
                print(f"[PREFLIGHT] {backend}:{model} OK")
            except Exception as exc:
                msg = str(exc).replace(anthropic_api_key(), "***") if anthropic_api_key() else str(exc)
                print(f"[SKIP] {backend}:{model} preflight failed -> {type(exc).__name__}: {msg[:600]}")
                model_status_rows[-1]["backend_status"] = f"ANTHROPIC_PREFLIGHT_FAILED:{type(exc).__name__}"
                model_status_rows[-1]["model_available"] = False
                continue

        for repeat in range(1, N_REPEATS + 1):
            setting = f"workers={workers}"
            if backend == "ollama":
                setting += f", ctx={NUM_CTX}, max_out={NUM_PREDICT}"
            else:
                setting += f", effort={ANTHROPIC_EFFORT}, thinking=disabled, max_out={ANTHROPIC_MAX_OUTPUT_TOKENS}"
            print(f"\n[{spec['capability_tier']}] {backend}:{model} repeat {repeat}/{N_REPEATS} | {setting}")
            started = time.perf_counter()
            row_list = [row for _, row in bench.iterrows()]
            completed = 0

            def report():
                elapsed = time.perf_counter() - started
                sec_per_case = elapsed / max(1, completed)
                remain = sec_per_case * (len(row_list) - completed)
                print(f"  {completed:,}/{len(row_list):,} | {sec_per_case:.1f} s/case | remaining~{remain/60:.1f} min")

            if workers == 1:
                for row in row_list:
                    llm_rows.append(run_llm_case(row, spec, repeat))
                    completed += 1
                    if completed % 5 == 0 or completed == len(row_list):
                        report()
            else:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    future_map = {ex.submit(run_llm_case, row, spec, repeat): clean(row.get("benchmark_id")) for row in row_list}
                    for fut in as_completed(future_map):
                        try:
                            llm_rows.append(fut.result())
                        except Exception as exc:
                            llm_rows.append({
                                "benchmark_id": future_map[fut], "llm_model": model, "llm_backend": backend,
                                "capability_tier": spec["capability_tier"], "capability_order": spec["capability_order"],
                                "parameter_size_b": spec["parameter_size_b"], "model_family": spec["model_family"],
                                "repeat": repeat, "llm_status": "LLM_ERROR", "llm_scope_type": "MANUAL_REVIEW",
                                "llm_requires_extended_identity": True, "llm_reason_ko": f"{type(exc).__name__}: worker failure",
                                "prompt_version": PROMPT_VERSION,
                            })
                        completed += 1
                        if completed % 5 == 0 or completed == len(row_list):
                            report()

    llm_results = pd.DataFrame(llm_rows)
    if not llm_results.empty:
        sort_cols = [c for c in ["llm_model", "repeat", "benchmark_id"] if c in llm_results.columns]
        llm_results = llm_results.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    pred = build_system_predictions(bench, llm_results)
    perf = summarize_performance(pred)
    by_scope = summarize_by_scope(pred)
    marginal = marginal_benefit(perf)
    global_common_ids = cross_model_common_ids(llm_results)
    cap_perf = capability_common_performance(pred, global_common_ids)
    cap_benefit = capability_rule_benefit(cap_perf)
    tests = paired_tests(pred)
    consistency = repeat_consistency(llm_results)
    model_status = pd.DataFrame(model_status_rows)
    rule_full = rule_full_benchmark_performance(bench)

    # Detailed misses are especially important for legal-screening interpretation.
    misses = pred[pred["prediction_available"].astype(bool) & pred["false_safe"].astype(bool)].copy()
    false_pos = pred[pred["prediction_available"].astype(bool) & pred["false_positive"].astype(bool)].copy()

    outputs = {
        "02_llm_semantic_results.csv": llm_results,
        "02_threeway_predictions.csv": pred,
        "02_threeway_performance.csv": perf,
        "02_threeway_by_scope.csv": by_scope,
        "02_hybrid_marginal_benefit.csv": marginal,
        "02_paired_mcnemar_tests.csv": tests,
        "02_llm_repeat_consistency.csv": consistency,
        "02_model_runtime_status.csv": model_status,
        "02_false_safe_cases.csv": misses,
        "02_false_positive_cases.csv": false_pos,
        "02_rule_full_benchmark_performance.csv": rule_full,
        "02_capability_common_performance.csv": cap_perf,
        "02_capability_rule_benefit.csv": cap_benefit,
        "02_cross_model_common_ids.csv": pd.DataFrame({"benchmark_id": sorted(global_common_ids)}),
    }
    for name, df in outputs.items():
        df.to_csv(INTERMEDIATE / name, index=False, encoding="utf-8-sig")

    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ollama_base_url": OLLAMA_BASE_URL,
        "model_specs": specs,
        "anthropic_configured": bool(anthropic.get("ready")),
        "anthropic_models": ANTHROPIC_MODELS,
        "repeats": N_REPEATS,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "max_cases": MAX_CASES,
        "prompt_version": PROMPT_VERSION,
        "max_source_chars": MAX_SOURCE_CHARS,
        "num_ctx": NUM_CTX,
        "num_predict": NUM_PREDICT,
        "llm_workers": LLM_WORKERS,
        "anthropic_workers": ANTHROPIC_WORKERS,
        "anthropic_effort": ANTHROPIC_EFFORT,
        "anthropic_thinking": "disabled",
        "anthropic_api_cache_version": ANTHROPIC_API_CACHE_VERSION,
        "cross_model_common_n": int(len(global_common_ids)),
        "capability_analysis": "ordinal LOW/MID/HIGH tiers; cross-model common valid-output rows",
        "primary_endpoint": "regulatory scope not safely representable by direct-CAS equality alone",
        "systems": ["RULE_ENGINE_ONLY", "LLM_ONLY", "LLM_PLUS_RULE"],
        "llm_failure_handling": "classification validity and literal evidence grounding are separate; main metrics use common classification-valid rows; evidence exact-match is reported separately",
        "validation_version": VALIDATION_VERSION,
    }
    (INTERMEDIATE / "02_run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[Overall performance]")
    if len(perf):
        show = ["system", "llm_model", "repeat", "n_common", "common_analysis_coverage", "system_output_coverage", "accuracy", "precision", "recall", "false_safe_rate", "false_positive_rate"]
        print(perf[show].round(4).to_string(index=False))
    if not llm_results.empty:
        class_ok = llm_results.get("llm_classification_valid", pd.Series(False, index=llm_results.index)).fillna(False).astype(bool)
        evidence_ok = llm_results.get("llm_evidence_quote_valid", pd.Series(False, index=llm_results.index)).fillna(False).astype(bool)
        ok_rate = float(class_ok.mean())
        ev_rate = float(evidence_ok.mean())
        print(f"\nLLM classification-valid coverage: {ok_rate:.3f}")
        print(f"Literal evidence exact-match rate: {ev_rate:.3f}")
        print(f"Cross-model common valid cases: {len(global_common_ids):,}")
        if len(cap_benefit):
            print("\n[Capability vs deterministic-rule marginal benefit]")
            cols = ["capability_tier", "llm_model", "llm_recall", "rule_recall", "hybrid_recall", "delta_rule_benefit_recall"]
            print(cap_benefit[cols].round(4).to_string(index=False))
    else:
        print("\nNo LLM rows were produced. Install/start Ollama and the requested model, then rerun script 02.")
    print("\nSaved to: intermediate/")
    print("Next: python 03_make_paper_results_CAPABILITY_FINAL.py")


if __name__ == "__main__":
    main()
