# -*- coding: utf-8 -*-
"""Shared utilities for Validation V2 benchmark construction.

Candidate collection and GOLD-label curation are deliberately separated.
External-DB retrieval never creates a GOLD label by itself. Only rows explicitly
marked curation_status=APPROVED are allowed into final validation files.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DATA = ROOT / "data"
SEEDS = ROOT / "seeds"
SOURCES = ROOT / "sources"
CACHE = ROOT / "cache"
for p in (DATA, SEEDS, SOURCES, CACHE):
    p.mkdir(parents=True, exist_ok=True)

DECISIONS = {"MATCH", "NO_MATCH", "REVIEW"}
PUBCHEM_ROOT = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_RETRIES = 4
PUBCHEM_MIN_INTERVAL_SEC = 0.25
_LAST_PUBCHEM_REQUEST_AT = 0.0


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def valid_cas(cas: str) -> bool:
    cas = re.sub(r"\s+", "", clean(cas))
    if not re.fullmatch(r"\d{2,7}-\d{2}-\d", cas):
        return False
    body, chk = cas.rsplit("-", 1)
    digits = body.replace("-", "")
    return sum((i + 1) * int(d) for i, d in enumerate(reversed(digits))) % 10 == int(chk)


def _pubchem_get_json(url: str, timeout: int = 30, allow_404: bool = False):
    """Rate-limited PubChem GET with retry for transient failures."""
    global _LAST_PUBCHEM_REQUEST_AT
    last_exc = None
    for attempt in range(PUBCHEM_RETRIES):
        try:
            wait = PUBCHEM_MIN_INTERVAL_SEC - (time.monotonic() - _LAST_PUBCHEM_REQUEST_AT)
            if wait > 0:
                time.sleep(wait)
            _LAST_PUBCHEM_REQUEST_AT = time.monotonic()
            r = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "validation-v2/1.1", "Accept": "application/json"},
            )
            if r.status_code == 404 and allow_404:
                return None
            if r.status_code == 429 or 500 <= r.status_code < 600:
                if attempt + 1 < PUBCHEM_RETRIES:
                    retry_after = r.headers.get("Retry-After", "")
                    try:
                        delay = float(retry_after) if retry_after else min(8.0, 2 ** attempt)
                    except Exception:
                        delay = min(8.0, 2 ** attempt)
                    time.sleep(max(0.5, delay))
                    continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < PUBCHEM_RETRIES:
                time.sleep(min(8.0, 2 ** attempt))
                continue
    if last_exc:
        raise last_exc
    return None


def _verify_props_for_cas(props: list[dict], cas: str, timeout: int = 30) -> list[dict]:
    verified = []
    for prop in props:
        cid = clean(prop.get("CID"))
        if not cid:
            continue
        syn_url = f"{PUBCHEM_ROOT}/compound/cid/{quote(cid, safe='')}/synonyms/JSON"
        syn_payload = _pubchem_get_json(syn_url, timeout=timeout, allow_404=True)
        if not syn_payload:
            continue
        info = syn_payload.get("InformationList", {}).get("Information", []) or []
        synonyms = info[0].get("Synonym", []) if info else []
        normalized = {re.sub(r"\s+", "", str(x)) for x in synonyms}
        if cas in normalized:
            verified.append(prop)
    return verified


def _query_pubchem_properties(term: str, timeout: int = 30):
    term = clean(term)
    if not term:
        return []
    q = quote(term, safe="")
    url = f"{PUBCHEM_ROOT}/compound/name/{q}/property/Title,CanonicalSMILES,IsomericSMILES,InChIKey/JSON"
    payload = _pubchem_get_json(url, timeout=timeout, allow_404=True)
    if not payload:
        return []
    return payload.get("PropertyTable", {}).get("Properties", []) or []


def pubchem_by_exact_cas(cas: str, candidate_name: str = "", timeout: int = 30) -> dict:
    """Resolve a candidate in PubChem and verify its exact CAS synonym.

    Lookup strategy
    ---------------
    1. Query PubChem directly by CAS.
    2. If the direct CAS query returns no compound, query by candidate name.
    3. In either path, accept a record only if the requested CAS is present in
       the returned CID's PubChem synonym list.

    This prevents a PubChem CAS-search miss from incorrectly rejecting a valid
    externally curated candidate while retaining strict exact-CAS verification.
    Identity metadata only: this function never infers a regulatory GOLD label.
    """
    cas = re.sub(r"\s+", "", clean(cas))
    candidate_name = clean(candidate_name)
    out = {
        "candidate_cas": cas,
        "external_db": "PubChem",
        "external_db_status": "",
        "pubchem_lookup_route": "",
        "pubchem_cid": "",
        "pubchem_title": "",
        "pubchem_isomeric_smiles": "",
        "pubchem_canonical_smiles": "",
        "pubchem_inchikey": "",
        "external_record_url": "",
        "retrieved_at_utc": utc_now(),
    }
    if not valid_cas(cas):
        out["external_db_status"] = "NO_VALID_CAS"
        return out

    try:
        # Route 1: direct CAS lookup.
        direct_props = _query_pubchem_properties(cas, timeout=timeout)
        verified = _verify_props_for_cas(direct_props, cas, timeout=timeout)
        route = "CAS_DIRECT"

        # Route 2: PubChem occasionally resolves the compound page and synonym
        # correctly but the PUG name endpoint does not return the CAS query.
        # In that case, query the curated candidate name and still require the
        # exact CAS in the resulting CID synonym list.
        if len(verified) == 0 and candidate_name:
            name_props = _query_pubchem_properties(candidate_name, timeout=timeout)
            verified = _verify_props_for_cas(name_props, cas, timeout=timeout)
            route = "NAME_FALLBACK_EXACT_CAS"

        if len(verified) == 0:
            out["external_db_status"] = "NOT_FOUND_OR_CAS_UNVERIFIED"
            out["pubchem_lookup_route"] = route
            return out
        if len(verified) > 1:
            # Multiple PubChem records can legitimately represent the same
            # formula/identity with the same CAS. Do not pick one silently.
            out["external_db_status"] = "MULTIPLE_EXACT_CAS_RECORDS_REVIEW"
            out["pubchem_lookup_route"] = route
            out["pubchem_cid"] = ";".join(clean(x.get("CID")) for x in verified)
            return out

        p = verified[0]
        cid = clean(p.get("CID"))
        out.update({
            "external_db_status": "EXACT_CAS_VERIFIED",
            "pubchem_lookup_route": route,
            "pubchem_cid": cid,
            "pubchem_title": clean(p.get("Title")),
            "pubchem_isomeric_smiles": clean(p.get("IsomericSMILES")) or clean(p.get("SMILES")),
            "pubchem_canonical_smiles": clean(p.get("CanonicalSMILES")) or clean(p.get("ConnectivitySMILES")),
            "pubchem_inchikey": clean(p.get("InChIKey")),
            "external_record_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
        })
        return out
    except Exception as exc:
        out["external_db_status"] = f"ERROR:{type(exc).__name__}"
        return out


def require_approved(df: pd.DataFrame, label_col: str = "gold_label") -> pd.DataFrame:
    if "curation_status" not in df.columns:
        raise ValueError("Missing curation_status column")
    approved = df[df["curation_status"].astype(str).str.upper().eq("APPROVED")].copy()
    if approved.empty:
        return approved
    if label_col not in approved.columns:
        raise ValueError(f"Missing {label_col}")
    bad = ~approved[label_col].astype(str).str.upper().isin(DECISIONS)
    if bad.any():
        raise ValueError(f"Approved rows contain invalid labels: {approved.loc[bad, label_col].tolist()}")
    approved[label_col] = approved[label_col].astype(str).str.upper()
    return approved


def split_input_gold(df: pd.DataFrame, input_cols: list[str], gold_cols: list[str], stem: str):
    miss_i = [c for c in input_cols if c not in df.columns]
    miss_g = [c for c in gold_cols if c not in df.columns]
    if miss_i or miss_g:
        raise ValueError(f"Missing columns input={miss_i}, gold={miss_g}")
    ip = DATA / f"{stem}_INPUT.csv"
    gp = DATA / f"{stem}_GOLD.csv"
    write_csv(df[input_cols].copy(), ip)
    write_csv(df[gold_cols].copy(), gp)
    return ip, gp


def dump_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
