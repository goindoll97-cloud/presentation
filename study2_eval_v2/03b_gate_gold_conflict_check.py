# -*- coding: utf-8 -*-
"""Human QC: compare frozen Hybrid gate decisions with retrieval GOLD before any LLM call.

01a can only pre-check exact CAS hits because SMILES do not exist yet. RDKit
parent/salt relations (e.g. a NOT_FOUND candidate that is a salt of another
catalog parent) therefore first become visible here, after 03 has frozen the gate.

This script reads GOLD, so it is a researcher QC step like 01a, not part of either
evaluated system; 04 does not read its output. Run it before setting
RETRIEVAL_EXECUTE_LLM=1.

Bias note: only Hybrid-gate disagreements are surfaced here. Correcting GOLD toward
the gate favours Hybrid, so every resulting override must be adjudicated on
regulatory grounds and reported (count before/after) in the paper. If GOLD is
changed, rerun 01 -> 02 -> 03 so every freeze is rebuilt.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
GOLD = ROOT / "data" / "retrieval_GOLD.csv"
QINPUT = ROOT / "data" / "retrieval_INPUT.csv"
GATE = ROOT / "intermediate" / "hybrid_gate_preflight.csv"
OUT = ROOT / "review" / "gate_gold_conflicts_pre_execution.csv"


def main() -> None:
    for p in [GOLD, QINPUT, GATE]:
        if not p.exists():
            raise FileNotFoundError(p)
    gold = pd.read_csv(GOLD, dtype=str).fillna("")
    qin = pd.read_csv(QINPUT, dtype=str).fillna("")
    gate = pd.read_csv(GATE, dtype=str).fillna("")

    df = gate.merge(gold, on="query_id", how="inner", validate="one_to_one").merge(
        qin, on="query_id", how="left", validate="one_to_one"
    )
    conflicts = df[
        df["gate_status"].eq("FOUND") & df["gate_target_id"].ne(df["gold_target_id"])
    ][[
        "query_id", "cas_inputs", "gold_target_id", "gold_status", "gold_derivation",
        "gate_target_id", "gate_reason",
    ]].copy()
    conflicts["conflict_type"] = conflicts["gold_status"].map(
        lambda s: "GOLD_NOT_FOUND_GATE_FOUND" if s == "NOT_FOUND" else "GOLD_AND_GATE_DIFFERENT_TARGET"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    conflicts.to_csv(OUT, index=False, encoding="utf-8-sig")

    print(f"[GATE vs GOLD] Hybrid deterministic FOUND decisions: {int(df['gate_status'].eq('FOUND').sum())}")
    print(f"[GATE vs GOLD] conflicts: {len(conflicts)}")
    if len(conflicts):
        print(conflicts.to_string(index=False))
        print(
            "\nAdjudicate each row on regulatory grounds. If GOLD is wrong, add it to "
            "review/retrieval_gold_overrides.csv and rerun 01 -> 02 -> 03. If the gate is wrong, "
            "leave GOLD unchanged (it will be scored as a Hybrid error)."
        )
    print(f"[OUT] {OUT}")


if __name__ == "__main__":
    main()
