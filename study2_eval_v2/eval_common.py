# -*- coding: utf-8 -*-
"""Shared utilities for Study 2 CAS-only retrieval evaluation.

This module never reads retrieval_GOLD.csv.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
DATA = ROOT / "data"
INTER = ROOT / "intermediate"
INTER.mkdir(parents=True, exist_ok=True)

QINPUT = DATA / "retrieval_INPUT.csv"
CATALOG_FILE = DATA / "regulatory_catalog.csv"
RETRIEVAL_FREEZE = ROOT / "RETRIEVAL_DATASET_FREEZE.json"
PUBCHEM_FREEZE = ROOT / "SHARED_PUBCHEM_FREEZE.json"
SHARED_SMILES = INTER / "shared_pubchem_smiles.csv"
PARENT_ENGINE = REPO_ROOT / "cheminformatics_identity_V5_FAIR.py"


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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_cas(x) -> list[str]:
    return [clean(v) for v in clean(x).split("|") if clean(v)]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def verify_frozen_inputs() -> dict:
    for p in [QINPUT, CATALOG_FILE, RETRIEVAL_FREEZE, PUBCHEM_FREEZE, SHARED_SMILES]:
        if not p.exists():
            raise FileNotFoundError(p)

    rf = json.loads(RETRIEVAL_FREEZE.read_text(encoding="utf-8"))
    pf = json.loads(PUBCHEM_FREEZE.read_text(encoding="utf-8"))

    rsha = rf.get("sha256", {})
    psha = pf.get("sha256", {})
    checks = {
        "retrieval_INPUT": rsha.get("retrieval_INPUT.csv") == sha256(QINPUT),
        "regulatory_catalog": rsha.get("regulatory_catalog.csv") == sha256(CATALOG_FILE),
        "shared_pubchem_smiles": (
            psha.get("intermediate/shared_pubchem_smiles.csv") == sha256(SHARED_SMILES)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Frozen evaluation inputs changed: {checks}")
    return checks


def load_inputs():
    checks = verify_frozen_inputs()
    qin = pd.read_csv(QINPUT, dtype=str).fillna("")
    catalog = pd.read_csv(CATALOG_FILE, dtype=str).fillna("")
    structures = pd.read_csv(SHARED_SMILES, dtype=str).fillna("")
    smap = {
        clean(r.cas): {
            "smiles": clean(r.smiles),
            "structure_status": clean(r.structure_status),
        }
        for r in structures.itertuples(index=False)
    }
    return qin, catalog, smap, checks


def cas_record(cas: str, smap: dict) -> dict:
    rec = smap.get(clean(cas), {})
    return {
        "cas": clean(cas),
        "smiles": clean(rec.get("smiles")),
    }


def query_payload(row: pd.Series, smap: dict) -> dict:
    cas_list = split_cas(row.get("cas_inputs"))
    return {
        "query_id": clean(row.get("query_id")),
        "candidate": [cas_record(c, smap) for c in cas_list],
    }


def catalog_payload(catalog: pd.DataFrame, smap: dict) -> list[dict]:
    entries = []
    for r in catalog.itertuples(index=False):
        target_id = clean(r.target_id)
        entry = {
            "target_id": target_id,
            "regulatory_scope": clean(r.regulatory_scope_text),
        }

        refs = split_cas(getattr(r, "reference_cas_set", ""))
        if refs:
            entry["reference_substances"] = [cas_record(c, smap) for c in refs]

        members = split_cas(getattr(r, "official_member_cas_set", ""))
        if members:
            entry["official_enumerated_members"] = [cas_record(c, smap) for c in members]

        mix_cas = split_cas(getattr(r, "official_mixture_cas", ""))
        comps = split_cas(getattr(r, "mixture_component_cas_set", ""))
        if mix_cas or comps:
            mix = {}
            if mix_cas:
                mix["mixture_cas"] = [cas_record(c, smap) for c in mix_cas]
            if comps:
                mix["component_cas"] = [cas_record(c, smap) for c in comps]
            entry["mixture_identity"] = mix

        entries.append(entry)
    return entries


def build_prompt(row: pd.Series, catalog: pd.DataFrame, smap: dict) -> str:
    payload = {
        "candidate_query": query_payload(row, smap),
        "regulatory_catalog": catalog_payload(catalog, smap),
        "task": (
            "Search the entire regulatory catalog. Return the one target_id that the "
            "candidate belongs to. If none applies, return NOT_FOUND. If the supplied "
            "CAS/SMILES information is insufficient to decide reliably, return REVIEW."
        ),
    }
    return "INPUT=" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def gate_query(row: pd.Series, catalog: pd.DataFrame, smap: dict) -> dict:
    """Conservative Hybrid gate. Only high-confidence unique decisions are final."""
    qid = clean(row.get("query_id"))
    cas_list = split_cas(row.get("cas_inputs"))

    if len(cas_list) > 1:
        sig = set(cas_list)
        hits = []
        for r in catalog.itertuples(index=False):
            comps = set(split_cas(getattr(r, "mixture_component_cas_set", "")))
            if comps and comps == sig:
                hits.append(clean(r.target_id))
        hits = sorted(set(hits))
        if len(hits) == 1:
            return {
                "query_id": qid,
                "gate_status": "FOUND",
                "gate_target_id": hits[0],
                "gate_reason": "EXACT_MIXTURE_COMPONENT_CAS_SET",
            }
        return {
            "query_id": qid,
            "gate_status": "REVIEW",
            "gate_target_id": "",
            "gate_reason": "NO_UNIQUE_EXACT_MIXTURE_COMPONENT_SET",
        }

    if len(cas_list) != 1:
        return {
            "query_id": qid,
            "gate_status": "REVIEW",
            "gate_target_id": "",
            "gate_reason": "CAS_INPUT_UNRESOLVED",
        }

    cas = cas_list[0]
    exact_hits = []
    for r in catalog.itertuples(index=False):
        tid = clean(r.target_id)
        if cas in split_cas(getattr(r, "reference_cas_set", "")):
            exact_hits.append(tid)
        if cas in split_cas(getattr(r, "official_member_cas_set", "")):
            exact_hits.append(tid)
        if cas in split_cas(getattr(r, "official_mixture_cas", "")):
            exact_hits.append(tid)

    exact_hits = sorted(set(exact_hits))
    if len(exact_hits) == 1:
        return {
            "query_id": qid,
            "gate_status": "FOUND",
            "gate_target_id": exact_hits[0],
            "gate_reason": "EXACT_CAS_IN_REGULATORY_CATALOG",
        }
    if len(exact_hits) > 1:
        return {
            "query_id": qid,
            "gate_status": "REVIEW",
            "gate_target_id": "",
            "gate_reason": "CAS_MAPS_TO_MULTIPLE_CATALOG_TARGETS",
        }

    query_smiles = clean(smap.get(cas, {}).get("smiles"))
    if query_smiles and PARENT_ENGINE.exists():
        engine = load_module(PARENT_ENGINE, "study2_retrieval_parent_engine")
        match_hits = []
        for r in catalog[catalog["target_type"].eq("PARENT_SALT")].itertuples(index=False):
            refs = split_cas(getattr(r, "reference_cas_set", ""))
            if len(refs) != 1:
                continue
            ref_smiles = clean(smap.get(refs[0], {}).get("smiles"))
            if not ref_smiles:
                continue
            try:
                decision, _reason = engine.compare_salt_parent_v5(
                    query_smiles, ref_smiles, "ALL_STEREOISOMERS"
                )
            except Exception:
                continue
            if clean(decision).upper() == "MATCH":
                match_hits.append(clean(r.target_id))

        match_hits = sorted(set(match_hits))
        if len(match_hits) == 1:
            return {
                "query_id": qid,
                "gate_status": "FOUND",
                "gate_target_id": match_hits[0],
                "gate_reason": "UNIQUE_RDKIT_PARENT_SALT_MATCH",
            }
        if len(match_hits) > 1:
            return {
                "query_id": qid,
                "gate_status": "REVIEW",
                "gate_target_id": "",
                "gate_reason": "MULTIPLE_PARENT_SALT_STRUCTURE_MATCHES",
            }

    return {
        "query_id": qid,
        "gate_status": "REVIEW",
        "gate_target_id": "",
        "gate_reason": "NO_HIGH_CONFIDENCE_DETERMINISTIC_TARGET",
    }


def consensus_outcome(group: pd.DataFrame) -> tuple[str, str]:
    outcomes = []
    for r in group.itertuples(index=False):
        status = clean(getattr(r, "status", "")).upper()
        target = clean(getattr(r, "target_id", ""))
        if status == "FOUND" and target:
            outcomes.append(target)
        elif status == "NOT_FOUND":
            outcomes.append("NOT_FOUND")
        else:
            outcomes.append("REVIEW")

    if not outcomes:
        return "REVIEW", ""
    counts = {v: outcomes.count(v) for v in set(outcomes)}
    winner, n = max(counts.items(), key=lambda z: (z[1], z[0]))
    if n <= len(outcomes) / 2:
        return "REVIEW", ""
    if winner == "NOT_FOUND":
        return "NOT_FOUND", ""
    if winner == "REVIEW":
        return "REVIEW", ""
    return "FOUND", winner
