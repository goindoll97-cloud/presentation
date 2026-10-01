# -*- coding: utf-8 -*-
"""Prepare one shared PubChem SMILES table for both evaluated systems.

PubChem is used only as a structure source. Exact-CAS synonym checking is an
internal retrieval/QC safeguard; evaluated systems receive only CAS + SMILES
(and an unresolved status when a SMILES cannot be obtained).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DATA = ROOT / "data"
INTER = ROOT / "intermediate"
INTER.mkdir(parents=True, exist_ok=True)

RETRIEVAL_FREEZE = ROOT / "RETRIEVAL_DATASET_FREEZE.json"
QINPUT = DATA / "retrieval_INPUT.csv"
CATALOG = DATA / "regulatory_catalog.csv"
RUNTIME_FILE = REPO_ROOT / "identity_shared_runtime_V5.py"

CACHE = INTER / "pubchem_exact_cas_cache.csv"
MINIMAL = INTER / "shared_pubchem_smiles.csv"
QC = INTER / "shared_pubchem_qc_internal.csv"
FREEZE_OUT = ROOT / "SHARED_PUBCHEM_FREEZE.json"


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


def split_cas(x) -> list[str]:
    return [clean(v) for v in clean(x).split("|") if clean(v)]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify_retrieval_freeze() -> dict:
    if not RETRIEVAL_FREEZE.exists():
        raise RuntimeError("Run 01_freeze_retrieval_dataset.py first")
    f = json.loads(RETRIEVAL_FREEZE.read_text(encoding="utf-8"))
    expected = f.get("sha256", {})
    checks = {
        "retrieval_INPUT.csv": expected.get("retrieval_INPUT.csv") == sha256(QINPUT),
        "regulatory_catalog.csv": expected.get("regulatory_catalog.csv") == sha256(CATALOG),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Retrieval data changed after freeze: {checks}")
    return checks


def main() -> None:
    checks = verify_retrieval_freeze()
    runtime = load_module(RUNTIME_FILE, "study2_retrieval_pubchem_runtime")

    qin = pd.read_csv(QINPUT, dtype=str).fillna("")
    catalog = pd.read_csv(CATALOG, dtype=str).fillna("")

    cas_set = set()
    for value in qin["cas_inputs"]:
        cas_set.update(split_cas(value))

    for col in [
        "reference_cas_set",
        "official_member_cas_set",
        "mixture_component_cas_set",
        "official_mixture_cas",
    ]:
        if col in catalog.columns:
            for value in catalog[col]:
                cas_set.update(split_cas(value))

    cas_list = sorted(c for c in cas_set if c)
    inventory = pd.DataFrame({"cas": cas_list, "chemical_name": [""] * len(cas_list)})
    inventory["smiles"] = ""

    # The shared runtime performs exact-CAS verification internally. Names/titles
    # from PubChem are not retained in the minimal structure table used by either
    # evaluated system.
    resolved, audit = runtime.enrich_inventory_with_pubchem(inventory, CACHE)

    minimal = resolved[["cas", "smiles", "structure_status"]].copy()
    if minimal["cas"].duplicated().any():
        raise RuntimeError("Duplicate CAS in shared PubChem structure table")
    minimal.to_csv(MINIMAL, index=False, encoding="utf-8-sig")
    audit.to_csv(QC, index=False, encoding="utf-8-sig")

    counts = minimal["structure_status"].value_counts().to_dict()
    freeze = {
        "protocol": "STUDY2_SHARED_PUBCHEM_SMILES_V1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_unique_cas": int(len(minimal)),
        "structure_status_counts": {str(k): int(v) for k, v in counts.items()},
        "evaluated_system_visible_columns": ["cas", "smiles", "structure_status"],
        "external_data_rule": (
            "LLM-only and Hybrid must read the same shared_pubchem_smiles.csv. "
            "PubChem names, titles, synonyms, InChIKeys, CIDs, descriptions, "
            "classifications, and regulatory annotations are not model inputs."
        ),
        "exact_cas_qc_note": (
            "PubChem synonym data may be used internally only to verify that a "
            "CAS lookup resolved to the exact record. This QC information is not "
            "supplied to either evaluated system."
        ),
        "retrieval_freeze_checks": checks,
        "sha256": {
            "RETRIEVAL_DATASET_FREEZE.json": sha256(RETRIEVAL_FREEZE),
            "../identity_shared_runtime_V5.py": sha256(RUNTIME_FILE),
            "intermediate/shared_pubchem_smiles.csv": sha256(MINIMAL),
        },
    }
    FREEZE_OUT.write_text(json.dumps(freeze, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(freeze, ensure_ascii=False, indent=2))
    print(f"[OUT] {MINIMAL}")
    print(f"[INTERNAL QC] {QC}")
    print(f"[FROZEN] {FREEZE_OUT}")


if __name__ == "__main__":
    main()
