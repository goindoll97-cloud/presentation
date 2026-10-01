# -*- coding: utf-8 -*-
"""Apply independent official mixture-CAS references to the retrieval catalog.

This step never reads retrieval_GOLD.csv. It enriches only mixture targets whose
CAS has been independently verified from an official regulatory source.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REFERENCE = ROOT / "reference" / "mixture_regulatory_reference.csv"
CATALOG = DATA / "regulatory_catalog.csv"
AUDIT = DATA / "mixture_regulatory_reference_applied.csv"

CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")


def clean(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def valid_cas_checksum(cas: str) -> bool:
    value = re.sub(r"\s+", "", clean(cas))
    if not CAS_RE.fullmatch(value):
        return False
    body, check = value.rsplit("-", 1)
    digits = body.replace("-", "")
    checksum = sum((i + 1) * int(d) for i, d in enumerate(reversed(digits))) % 10
    return checksum == int(check)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for p in [REFERENCE, CATALOG]:
        if not p.exists():
            raise FileNotFoundError(p)

    ref = pd.read_csv(REFERENCE, dtype=str).fillna("")
    catalog = pd.read_csv(CATALOG, dtype=str).fillna("")

    required = {
        "target_id", "target_scope_id", "official_mixture_cas",
        "source_name", "source_url", "evidence_note",
    }
    missing = sorted(required - set(ref.columns))
    if missing:
        raise RuntimeError(f"Mixture reference missing columns: {missing}")
    if ref["target_id"].duplicated().any():
        raise RuntimeError("Duplicate target_id in mixture regulatory reference")

    if "official_mixture_cas" not in catalog.columns:
        catalog["official_mixture_cas"] = ""
    if "official_mixture_cas_basis" not in catalog.columns:
        catalog["official_mixture_cas_basis"] = ""

    audit_rows = []
    for r in ref.itertuples(index=False):
        tid = clean(r.target_id)
        cas = clean(r.official_mixture_cas)
        if not tid or not cas:
            raise RuntimeError("Blank target_id/CAS in mixture regulatory reference")
        if not valid_cas_checksum(cas):
            raise RuntimeError(f"Invalid official mixture CAS checksum: {tid} -> {cas}")

        hit = catalog.index[catalog["target_id"].eq(tid)].tolist()
        if len(hit) != 1:
            raise RuntimeError(f"Reference target must map to exactly one catalog row: {tid}")
        idx = hit[0]
        if clean(catalog.at[idx, "target_type"]) != "MIXTURE":
            raise RuntimeError(f"Official mixture CAS attached to non-mixture target: {tid}")

        existing = clean(catalog.at[idx, "official_mixture_cas"])
        if existing and existing != cas:
            raise RuntimeError(f"Conflicting official mixture CAS for {tid}: {existing} vs {cas}")

        catalog.at[idx, "official_mixture_cas"] = cas
        catalog.at[idx, "official_mixture_cas_basis"] = "INDEPENDENT_OFFICIAL_REGULATORY_REFERENCE"
        basis = clean(catalog.at[idx, "catalog_identity_basis"])
        marker = "INDEPENDENT_OFFICIAL_MIXTURE_CAS_REFERENCE"
        if marker not in basis.split("+"):
            catalog.at[idx, "catalog_identity_basis"] = basis + ("+" if basis else "") + marker

        audit_rows.append({
            "target_id": tid,
            "target_scope_id": clean(r.target_scope_id),
            "official_mixture_cas": cas,
            "source_name": clean(r.source_name),
            "source_url": clean(r.source_url),
            "evidence_note": clean(r.evidence_note),
            "reference_sha256": sha256(REFERENCE),
        })

    catalog.to_csv(CATALOG, index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(AUDIT, index=False, encoding="utf-8-sig")

    print(f"[MIXTURE REFERENCE] applied={len(audit_rows)}")
    if audit_rows:
        print(pd.DataFrame(audit_rows)[["target_id", "official_mixture_cas", "source_name"]].to_string(index=False))
    print(f"[OUT] {AUDIT}")


if __name__ == "__main__":
    main()
