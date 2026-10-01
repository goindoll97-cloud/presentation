# -*- coding: utf-8 -*-
"""Apply documented CAS corrections to the derived retrieval benchmark only.

The frozen Validation V2 source files are NEVER modified here. Corrections are
applied only when the source identifier is independently verified and the
replacement CAS passes checksum validation. The correction audit is intentionally
deterministic: no execution timestamp is written into the CSV hash surface.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CATALOG = DATA / "regulatory_catalog.csv"
QINPUT = DATA / "retrieval_INPUT.csv"
QGOLD = DATA / "retrieval_GOLD.csv"
EXCLUDED = DATA / "retrieval_EXCLUDED_NO_CAS.csv"
MANIFEST = DATA / "retrieval_build_manifest.json"
AUDIT = DATA / "retrieval_CAS_CORRECTIONS.csv"

CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")
CAS_CORRECTIONS = {
    "3084-48-0": {
        "corrected_cas": "3084-48-8",
        "chemical_identity": "trihexylphosphine oxide",
        "reason": (
            "Source CAS fails CAS checksum; corrected CAS 3084-48-8 has valid checksum "
            "and is independently associated with trihexylphosphine oxide."
        ),
        "evidence_url_1": "https://www.chemicalbook.com/ChemicalProductProperty_KR_CB2918549.htm",
        "evidence_url_2": "https://www.alfa-chemistry.com/cas_3084-48-8.htm",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def normalize_cas(x) -> str:
    return re.sub(r"\s+", "", clean(x))


def valid_cas_checksum(cas: str) -> bool:
    value = normalize_cas(cas)
    if not CAS_RE.fullmatch(value):
        return False
    body, check = value.rsplit("-", 1)
    digits = body.replace("-", "")
    checksum = sum((i + 1) * int(d) for i, d in enumerate(reversed(digits))) % 10
    return checksum == int(check)


def correct_one(cas: str) -> tuple[str, str]:
    value = normalize_cas(cas)
    meta = CAS_CORRECTIONS.get(value)
    if not meta:
        return value, ""
    corrected = normalize_cas(meta["corrected_cas"])
    if not valid_cas_checksum(corrected):
        raise RuntimeError(f"Configured corrected CAS is invalid: {value} -> {corrected}")
    return corrected, value


def correct_set(value: str) -> tuple[str, list[tuple[str, str]]]:
    parts = [normalize_cas(v) for v in clean(value).split("|") if normalize_cas(v)]
    changes, corrected = [], []
    for p in parts:
        new, old = correct_one(p)
        corrected.append(new)
        if old:
            changes.append((old, new))
    return "|".join(sorted(set(corrected))), changes


def audit_row(location: str, record_id: str, old: str, new: str) -> dict:
    meta = CAS_CORRECTIONS[old]
    return {
        "location": location,
        "record_id": record_id,
        "original_cas": old,
        "corrected_cas": new,
        "chemical_identity": meta["chemical_identity"],
        "reason": meta["reason"],
        "evidence_url_1": meta["evidence_url_1"],
        "evidence_url_2": meta["evidence_url_2"],
    }


def main() -> None:
    for p in [CATALOG, QINPUT, QGOLD, EXCLUDED, MANIFEST]:
        if not p.exists():
            raise FileNotFoundError(p)

    qin = pd.read_csv(QINPUT, dtype=str).fillna("")
    qgold = pd.read_csv(QGOLD, dtype=str).fillna("")
    catalog = pd.read_csv(CATALOG, dtype=str).fillna("")
    build = json.loads(MANIFEST.read_text(encoding="utf-8"))
    audit_rows = []

    for idx, row in qin.iterrows():
        new_value, changes = correct_set(row["cas_inputs"])
        if changes:
            qin.at[idx, "cas_inputs"] = new_value
            for old, new in changes:
                audit_rows.append(audit_row(
                    "retrieval_INPUT.cas_inputs", clean(row["query_id"]), old, new
                ))

    cas_fields = ["reference_cas_set", "official_member_cas_set", "mixture_component_cas_set"]
    target_id_map = {}
    for idx, row in catalog.iterrows():
        original_tid = clean(row["target_id"])
        for field in cas_fields:
            old_value = clean(row.get(field, ""))
            if not old_value:
                continue
            new_value, changes = correct_set(old_value)
            if changes:
                catalog.at[idx, field] = new_value
                for old, new in changes:
                    audit_rows.append(audit_row(
                        f"regulatory_catalog.{field}", original_tid, old, new
                    ))

        if original_tid.startswith("PS::"):
            old_cas = original_tid.split("::", 1)[1]
            new_cas, changed_from = correct_one(old_cas)
            if changed_from:
                new_tid = f"PS::{new_cas}"
                catalog.at[idx, "target_id"] = new_tid
                target_id_map[original_tid] = new_tid
                audit_rows.append(audit_row(
                    "regulatory_catalog.target_id", original_tid, changed_from, new_cas
                ))

    if target_id_map:
        qgold["gold_target_id"] = qgold["gold_target_id"].replace(target_id_map)

    if qin["cas_inputs"].duplicated().any():
        dup = qin.loc[qin["cas_inputs"].duplicated(keep=False), ["query_id", "cas_inputs"]]
        raise RuntimeError(
            "CAS correction created duplicate retrieval signatures: "
            + str(dup.to_dict("records"))
        )

    bad = []
    for r in qin.itertuples(index=False):
        parts = [p for p in str(r.cas_inputs).split("|") if p]
        if not parts or any(not valid_cas_checksum(p) for p in parts):
            bad.append({"query_id": r.query_id, "cas_inputs": r.cas_inputs})
    if bad:
        raise RuntimeError(f"Undocumented invalid CAS remains after correction: {bad[:10]}")

    audit = pd.DataFrame(audit_rows, columns=[
        "location", "record_id", "original_cas", "corrected_cas",
        "chemical_identity", "reason", "evidence_url_1", "evidence_url_2",
    ])
    if audit.empty:
        raise RuntimeError(
            "Expected documented source CAS correction 3084-48-0 -> 3084-48-8 was not encountered"
        )

    qin.to_csv(QINPUT, index=False, encoding="utf-8-sig")
    qgold.to_csv(QGOLD, index=False, encoding="utf-8-sig")
    catalog.to_csv(CATALOG, index=False, encoding="utf-8-sig")
    audit.to_csv(AUDIT, index=False, encoding="utf-8-sig")

    build["cas_correction_policy"] = (
        "Frozen Validation V2 source files are unchanged. Documented source CAS inconsistencies "
        "are normalized only in the derived retrieval representation when independently verified. "
        "The correction audit contains no run timestamp so repeated builds are hash-stable."
    )
    build["cas_corrections"] = [{"original_cas": old, **meta} for old, meta in CAS_CORRECTIONS.items()]
    build["cas_correction_audit_rows"] = int(len(audit))
    build["sha256"].update({
        "regulatory_catalog.csv": sha256(CATALOG),
        "retrieval_INPUT.csv": sha256(QINPUT),
        "retrieval_GOLD.csv": sha256(QGOLD),
        "retrieval_EXCLUDED_NO_CAS.csv": sha256(EXCLUDED),
        "retrieval_CAS_CORRECTIONS.csv": sha256(AUDIT),
    })
    MANIFEST.write_text(json.dumps(build, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[CAS CORRECTION] audit rows={len(audit)}")
    print(audit[["location", "record_id", "original_cas", "corrected_cas"]].to_string(index=False))
    print(f"[OUT] {AUDIT}")


if __name__ == "__main__":
    main()
