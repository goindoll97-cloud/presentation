# -*- coding: utf-8 -*-
"""Shared utilities for Validation V2 benchmark construction.

Design principle
----------------
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


def pubchem_by_exact_cas(cas: str, timeout: int = 30) -> dict:
    """Resolve a CAS through PubChem and verify that exact CAS is a synonym.

    This function provides identity metadata only. It does not infer a regulatory
    MATCH/NO_MATCH label.
    """
    cas = re.sub(r"\s+", "", clean(cas))
    out = {
        "candidate_cas": cas,
        "external_db": "PubChem",
        "external_db_status": "",
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

    q = quote(cas, safe="")
    url = f"{PUBCHEM_ROOT}/compound/name/{q}/property/Title,CanonicalSMILES,IsomericSMILES,InChIKey/JSON"
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "validation-v2/1.0"})
        if r.status_code == 404:
            out["external_db_status"] = "NOT_FOUND"
            return out
        r.raise_for_status()
        props = r.json().get("PropertyTable", {}).get("Properties", []) or []
        verified = []
        for prop in props:
            cid = clean(prop.get("CID"))
            if not cid:
                continue
            time.sleep(0.12)
            syn_url = f"{PUBCHEM_ROOT}/compound/cid/{quote(cid, safe='')}/synonyms/JSON"
            sr = requests.get(syn_url, timeout=timeout, headers={"User-Agent": "validation-v2/1.0"})
            sr.raise_for_status()
            info = sr.json().get("InformationList", {}).get("Information", []) or []
            synonyms = info[0].get("Synonym", []) if info else []
            if cas in {re.sub(r"\s+", "", str(x)) for x in synonyms}:
                verified.append(prop)
        if len(verified) != 1:
            out["external_db_status"] = "MULTIPLE_OR_UNVERIFIED"
            return out
        p = verified[0]
        cid = clean(p.get("CID"))
        out.update({
            "external_db_status": "EXACT_CAS_VERIFIED",
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


def split_input_gold(df: pd.DataFrame, input_cols: list[str], gold_cols: list[str], stem: str) -> tuple[Path, Path]:
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
