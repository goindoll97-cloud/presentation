# -*- coding: utf-8 -*-
"""Study 2 / Step 06: publication tables and figures for chemical identity study."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
INTERMEDIATE = ROOT / "intermediate"
OUT = ROOT / "paper_output" / "study2_identity"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def read(name: str) -> pd.DataFrame:
    p = INTERMEDIATE / name
    return pd.read_csv(p).fillna("") if p.exists() else pd.DataFrame()


def numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def summary_text(mean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lookup = {r["system"]: r for _, r in mean.iterrows()} if not mean.empty else {}
    for setting, systems in [
        ("OPERATIONAL_CAS_ONLY", ["CLAUDE_CAS_ONLY", "RDKIT_CAS_LOOKUP", "HYBRID_OPERATIONAL"]),
        ("CONTROLLED_STRUCTURE", ["CLAUDE_STRUCTURE", "RDKIT_STRUCTURE", "HYBRID_STRUCTURE"]),
    ]:
        for s in systems:
            r = lookup.get(s)
            if r is None:
                continue
            rows.append({
                "setting": setting,
                "system": s,
                "interpretation": (
                    "LLM-only identity decision" if s.startswith("CLAUDE") else
                    "deterministic cheminformatics identity resolution" if s.startswith("RDKIT") else
                    "deterministic-first hybrid with LLM fallback"
                ),
                "coverage": r.get("coverage_mean"),
                "accuracy": r.get("accuracy_mean"),
                "recall": r.get("recall_mean"),
                "specificity": r.get("specificity_mean"),
                "false_safe_rate": r.get("false_safe_rate_mean"),
            })
    return pd.DataFrame(rows)


def bar_metric(mean: pd.DataFrame, setting: str, systems: list[str], metric: str, title: str, filename: str) -> None:
    g = mean[(mean["setting"] == setting) & mean["system"].isin(systems)].copy()
    if g.empty:
        return
    g["system"] = pd.Categorical(g["system"], systems, ordered=True)
    g = g.sort_values("system")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(g["system"].astype(str), g[metric].astype(float))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.replace("_mean", "").replace("_", " ").title())
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(g[metric].astype(float)):
        ax.text(i, min(1.02, v + 0.025), f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    bench = read("04_identity_challenge_benchmark.csv")
    qc = read("04_identity_challenge_qc.csv")
    enriched = read("05_identity_benchmark_enriched.csv")
    llm = read("05_identity_claude_outputs.csv")
    pred = read("05_identity_system_predictions.csv")
    perf = read("05_identity_performance_by_repeat.csv")
    mean = read("05_identity_performance_mean.csv")
    by_rule = read("05_identity_performance_by_rule.csv")
    conflicts = read("05_identity_claude_rdkit_conflicts.csv")
    tests = read("05_identity_paired_mcnemar.csv")
    consistency = read("05_identity_repeat_consistency.csv")
    failures = read("05_identity_failure_cases.csv")
    pubchem = read("05_pubchem_operational_identity_audit.csv")

    if mean.empty:
        raise FileNotFoundError("Run 05_compare_claude_rdkit_identity.py first")

    mean = numeric(mean, [
        "coverage_mean", "coverage_sd", "accuracy_mean", "accuracy_sd", "recall_mean", "recall_sd",
        "specificity_mean", "specificity_sd", "false_safe_rate_mean", "false_safe_rate_sd",
        "false_positive_rate_mean", "false_positive_rate_sd",
    ])
    compact = summary_text(mean)

    notes = pd.DataFrame([
        {"item": "Study 2 question", "value": "After semantic scope recognition, can a high-capability LLM resolve CAS-to-broad-salt chemical identity as reliably as deterministic cheminformatics?"},
        {"item": "Primary operational comparison", "value": "Claude sees regulatory scope + candidate CAS only; deterministic arm resolves exact CAS through PubChem and applies RDKit parent/salt normalization."},
        {"item": "Controlled comparison", "value": "Claude and RDKit both receive the same candidate SMILES and frozen reference-parent SMILES."},
        {"item": "Hybrid policy", "value": "RDKit-primary. Claude is used only when deterministic structure resolution abstains/requires review."},
        {"item": "Reference status", "value": "Source-supported chemical-identity operational reference; not expert legal gold and not a final compliance determination."},
        {"item": "Scope", "value": "Broad-salt proof of concept only; compound groups, mixtures/UVCB, reaction products, and generic derivative families are not claimed as structurally solved here."},
    ])

    # Optional Study 1 link for the two-stage narrative.
    study1 = read("02_capability_rule_benefit.csv")

    out_xlsx = OUT / "PAPER_STUDY2_CHEMICAL_IDENTITY_RESULTS.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        notes.to_excel(writer, sheet_name="0_Study_notes", index=False)
        compact.to_excel(writer, sheet_name="Table1_Main_identity", index=False)
        mean.to_excel(writer, sheet_name="Table2_Performance_mean", index=False)
        perf.to_excel(writer, sheet_name="Table3_By_repeat", index=False)
        by_rule.to_excel(writer, sheet_name="Table4_By_rule", index=False)
        tests.to_excel(writer, sheet_name="Table5_Paired_McNemar", index=False)
        consistency.to_excel(writer, sheet_name="Table6_Repeat_consistency", index=False)
        conflicts.to_excel(writer, sheet_name="Table7_Claude_RDKit_conflict", index=False)
        qc.to_excel(writer, sheet_name="Benchmark_QC", index=False)
        bench.to_excel(writer, sheet_name="Supp_Benchmark", index=False)
        enriched.to_excel(writer, sheet_name="Supp_Resolved_structures", index=False)
        pubchem.to_excel(writer, sheet_name="Supp_PubChem_audit", index=False)
        failures.to_excel(writer, sheet_name="Supp_Failure_cases", index=False)
        llm.to_excel(writer, sheet_name="Supp_Claude_outputs", index=False)
        pred.to_excel(writer, sheet_name="Supp_All_predictions", index=False)
        if not study1.empty:
            study1.to_excel(writer, sheet_name="Study1_rule_benefit_link", index=False)

    bar_metric(mean, "OPERATIONAL_CAS_ONLY",
               ["CLAUDE_CAS_ONLY", "RDKIT_CAS_LOOKUP", "HYBRID_OPERATIONAL"],
               "recall_mean", "Study 2A. CAS-only operational identity recall", "Fig1_operational_recall.png")
    bar_metric(mean, "OPERATIONAL_CAS_ONLY",
               ["CLAUDE_CAS_ONLY", "RDKIT_CAS_LOOKUP", "HYBRID_OPERATIONAL"],
               "false_safe_rate_mean", "Study 2A. CAS-only false-safe rate", "Fig2_operational_false_safe.png")
    bar_metric(mean, "CONTROLLED_STRUCTURE",
               ["CLAUDE_STRUCTURE", "RDKIT_STRUCTURE", "HYBRID_STRUCTURE"],
               "recall_mean", "Study 2B. Same-structure-input identity recall", "Fig3_structure_recall.png")
    bar_metric(mean, "CONTROLLED_STRUCTURE",
               ["CLAUDE_STRUCTURE", "RDKIT_STRUCTURE", "HYBRID_STRUCTURE"],
               "accuracy_mean", "Study 2B. Same-structure-input identity accuracy", "Fig4_structure_accuracy.png")

    # Coverage figure: important because deterministic systems may abstain when CAS lookup fails.
    systems = ["CLAUDE_CAS_ONLY", "RDKIT_CAS_LOOKUP", "HYBRID_OPERATIONAL", "CLAUDE_STRUCTURE", "RDKIT_STRUCTURE", "HYBRID_STRUCTURE"]
    g = mean[mean["system"].isin(systems)].copy()
    if not g.empty:
        g["system"] = pd.Categorical(g["system"], systems, ordered=True)
        g = g.sort_values("system")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(g["system"].astype(str), g["coverage_mean"].astype(float))
        ax.set_ylim(0, 1.05); ax.set_ylabel("Decision coverage"); ax.set_title("Decision coverage by identity-resolution architecture")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout(); fig.savefig(FIG / "Fig5_identity_coverage.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    # Console summary
    print("=" * 84)
    print("06 / STUDY 2 - PAPER RESULTS")
    print("=" * 84)
    show_cols = ["system", "setting", "coverage_mean", "accuracy_mean", "recall_mean", "specificity_mean", "false_safe_rate_mean"]
    print(mean[show_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved workbook: {out_xlsx}")
    print(f"Figures: {FIG}")


if __name__ == "__main__":
    main()
