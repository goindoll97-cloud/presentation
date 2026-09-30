# -*- coding: utf-8 -*-
"""Study 2 / Step 07 FAIR V5 publication figures.

Each figure answers one interpretable question instead of combining all metrics
into a single crowded plot.

Main figures
------------
Fig01  Overall correct resolution with 95% CI.
Fig02  Coverage vs selective-accuracy trade-off.
Fig03  Case-outcome composition (correct / REVIEW / incorrect).
Fig04  Overall correct resolution by difficulty.

Supplementary figures
---------------------
FigS01 Parent-level overall correct resolution heatmap.
FigS02 Challenge-class heatmap for classes with n >= MIN_CHALLENGE_N.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
INTER = ROOT / "intermediate"
OUT = ROOT / "figures_study2_v5"
OUT.mkdir(parents=True, exist_ok=True)

PERF = INTER / "06_fair_v5_performance_mean.csv"
PARENT = INTER / "06_fair_v5_performance_by_parent.csv"
DIFF = INTER / "06_fair_v5_performance_by_difficulty.csv"
CHALLENGE = INTER / "06_fair_v5_performance_by_challenge_class.csv"
PRED = INTER / "05_v5_identity_system_predictions_caselevel.csv"
META = INTER / "05_v5_run_metadata.json"

SYSTEMS = ["CLAUDE_DB", "RDKIT_SALT_AWARE_V5", "HYBRID_SALT_AWARE_V5"]
LABELS = {
    "CLAUDE_DB": "LLM + DB",
    "RDKIT_SALT_AWARE_V5": "DB + salt-aware RDKit",
    "HYBRID_SALT_AWARE_V5": "Hybrid",
}
MARKERS = ["o", "s", "D"]
MIN_CHALLENGE_N = 3


def save(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {stem}.png/.pdf")


def title_prefix(status: str) -> str:
    return "Independent final holdout" if str(status).startswith("FINAL_") else "Development / exploratory reanalysis"


def as_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def overall_resolution_fig(df: pd.DataFrame, status: str) -> None:
    """Primary endpoint: overall correct resolution with 95% Wilson CI."""
    for bench, g0 in df.groupby("benchmark_set", sort=False):
        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        y = np.arange(len(SYSTEMS))
        vals, lo, hi, counts = [], [], [], []

        for s in SYSTEMS:
            row = g0[g0.system.eq(s)]
            if row.empty:
                vals.append(np.nan); lo.append(np.nan); hi.append(np.nan); counts.append("")
                continue
            r = row.iloc[0]
            v = float(r["overall_correct_resolution_rate"]) * 100
            l = float(r["overall_correct_resolution_ci95_low"]) * 100
            h = float(r["overall_correct_resolution_ci95_high"]) * 100
            vals.append(v); lo.append(l); hi.append(h)
            n_correct = int(round(float(r["overall_correct_resolution_rate"]) * int(r["n_total"])))
            counts.append(f"{n_correct}/{int(r['n_total'])}")

        vals = np.asarray(vals, dtype=float)
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)

        for i, s in enumerate(SYSTEMS):
            if not np.isfinite(vals[i]):
                continue
            ax.errorbar(
                vals[i], y[i],
                xerr=np.array([[vals[i] - lo[i]], [hi[i] - vals[i]]]),
                fmt=MARKERS[i], capsize=4, markersize=8, linewidth=1.4,
            )
            ax.annotate(
                f"{vals[i]:.1f}%  ({counts[i]})",
                (vals[i], y[i]), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=10,
            )

        ax.set_yticks(y)
        ax.set_yticklabels([LABELS[s] for s in SYSTEMS])
        ax.invert_yaxis()
        ax.set_xlim(85, 103 if str(status).startswith("FINAL_") else 105)
        ax.set_xlabel("Overall correct resolution (%)")
        ax.set_title(f"{title_prefix(status)}: overall correct resolution")
        ax.grid(axis="x", alpha=.25)
        ax.text(
            0.0, -0.16,
            "Horizontal bars show 95% Wilson confidence intervals; REVIEW remains in the denominator.",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
        )
        fig.subplots_adjust(left=.28, right=.96, top=.88, bottom=.24)
        save(fig, f"Fig01_{bench.lower()}_overall_resolution")


def coverage_accuracy_tradeoff_fig(df: pd.DataFrame, status: str) -> None:
    """Separates abstention/coverage from accuracy among decided cases."""
    for bench, g0 in df.groupby("benchmark_set", sort=False):
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        for i, s in enumerate(SYSTEMS):
            row = g0[g0.system.eq(s)]
            if row.empty:
                continue
            r = row.iloc[0]
            x = float(r["coverage"]) * 100
            y = float(r["accuracy"]) * 100
            overall = float(r["overall_correct_resolution_rate"]) * 100
            ax.scatter(x, y, s=90, marker=MARKERS[i], label=LABELS[s], zorder=3)
            offset_y = 7 if i != 1 else -25
            ax.annotate(
                f"{LABELS[s]}\nOCR={overall:.1f}%",
                (x, y), xytext=(7, offset_y), textcoords="offset points", fontsize=9,
            )

        ax.set_xlim(93, 101)
        ax.set_ylim(96, 101)
        ax.set_xlabel("Coverage (%)")
        ax.set_ylabel("Selective accuracy (%)")
        ax.set_title(f"{title_prefix(status)}: coverage–accuracy trade-off")
        ax.grid(alpha=.25)
        ax.legend(frameon=False, loc="lower left")
        fig.tight_layout()
        save(fig, f"Fig02_{bench.lower()}_coverage_accuracy_tradeoff")


def case_outcome_fig(pred: pd.DataFrame, status: str) -> None:
    """Directly separates correct resolution, REVIEW, and incorrect resolution."""
    if pred.empty:
        return

    outcomes = ["Correct", "REVIEW", "Incorrect"]
    x = np.arange(len(SYSTEMS))
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    bottom = np.zeros(len(SYSTEMS), dtype=float)

    for outcome in outcomes:
        vals = []
        for s in SYSTEMS:
            g = pred[pred["system"].eq(s)].copy()
            decided = as_bool_series(g["decided"])
            correct = as_bool_series(g["correct"])
            if outcome == "Correct":
                n = int((decided & correct).sum())
            elif outcome == "REVIEW":
                n = int((~decided).sum())
            else:
                n = int((decided & ~correct).sum())
            vals.append(n)

        bars = ax.bar(x, vals, bottom=bottom, label=outcome)
        for b, v, btm in zip(bars, vals, bottom):
            if v > 0:
                ax.text(
                    b.get_x() + b.get_width()/2, btm + v/2,
                    str(v), ha="center", va="center", fontsize=10,
                )
        bottom += np.asarray(vals, dtype=float)

    n_total = int(pred[pred.system.eq(SYSTEMS[0])].shape[0])
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[s] for s in SYSTEMS])
    ax.set_ylabel("Number of cases")
    ax.set_ylim(0, max(n_total, int(bottom.max())) * 1.08)
    ax.set_title(f"{title_prefix(status)}: case-level outcome composition")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, -0.12))
    ax.grid(axis="y", alpha=.2)
    fig.subplots_adjust(bottom=.22)
    stem = "Fig03_final_independent_holdout_case_outcomes" if str(status).startswith("FINAL_") else "Fig03_development_case_outcomes"
    save(fig, stem)


def difficulty_fig(df: pd.DataFrame, status: str) -> None:
    """Overall correct resolution by independently assigned case difficulty."""
    if df.empty:
        return

    order = ["easy", "medium", "hard"]
    for bench, g in df.groupby("benchmark_set", sort=False):
        present = {str(v).strip().lower() for v in g["difficulty"].dropna()}
        levels = [d for d in order if d in present]
        if not levels:
            continue

        n_map = {}
        for d in levels:
            r = g[g["difficulty"].astype(str).str.strip().str.lower().eq(d)]
            n_map[d] = int(r["n_total"].max()) if len(r) else 0

        x = np.arange(len(levels))
        width = .23
        fig, ax = plt.subplots(figsize=(8.0, 5.4))
        for j, s in enumerate(SYSTEMS):
            vals = []
            for d in levels:
                r = g[g.system.eq(s) & g["difficulty"].astype(str).str.strip().str.lower().eq(d)]
                vals.append(float(r["overall_correct_resolution_rate"].iloc[0]) * 100 if len(r) else np.nan)
            bars = ax.bar(x + (j-1)*width, vals, width, label=LABELS[s])
            for b, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(b.get_x()+b.get_width()/2, v+0.8, f"{v:.1f}", ha="center", fontsize=8.5)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{d.title()}\n(n={n_map[d]})" for d in levels])
        ax.set_ylim(85, 104)
        ax.set_ylabel("Overall correct resolution (%)")
        ax.set_title(f"{title_prefix(status)}: performance by difficulty")
        ax.grid(axis="y", alpha=.25)
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, -0.14))
        fig.subplots_adjust(bottom=.24)
        save(fig, f"Fig04_{bench.lower()}_difficulty")


def parent_heatmap(df: pd.DataFrame, status: str) -> None:
    if df.empty or "reference_parent_name" not in df.columns:
        return

    for bench, g in df.groupby("benchmark_set", sort=False):
        p = g.pivot(
            index="reference_parent_name", columns="system",
            values="overall_correct_resolution_rate",
        ).reindex(columns=SYSTEMS)
        if p.empty:
            continue

        p = p.assign(_m=p.mean(axis=1)).sort_values("_m").drop(columns="_m")
        arr = p.to_numpy(float) * 100
        fig, ax = plt.subplots(figsize=(8.0, max(5.5, .42*len(p)+1.8)))
        im = ax.imshow(arr, aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(np.arange(len(SYSTEMS)))
        ax.set_xticklabels([LABELS[s] for s in SYSTEMS])
        ax.set_yticks(np.arange(len(p.index)))
        ax.set_yticklabels(p.index)

        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isfinite(arr[i, j]):
                    ax.text(j, i, f"{arr[i,j]:.1f}", ha="center", va="center", fontsize=8.5)

        ax.set_title(f"{title_prefix(status)}: parent-level overall correct resolution")
        cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.03)
        cb.set_label("Overall correct resolution (%)")
        fig.tight_layout()
        save(fig, f"FigS01_{bench.lower()}_parent_heatmap")


def challenge_heatmap(df: pd.DataFrame, status: str) -> None:
    """Supplementary class-level view, excluding very small classes."""
    if df.empty or "challenge_class" not in df.columns:
        return

    for bench, g in df.groupby("benchmark_set", sort=False):
        counts = g.groupby("challenge_class", as_index=False)["n_total"].max()
        keep = counts[counts["n_total"] >= MIN_CHALLENGE_N].copy()
        if keep.empty:
            continue

        n_map = dict(zip(keep["challenge_class"].astype(str), keep["n_total"].astype(int)))
        gg = g[g["challenge_class"].astype(str).isin(n_map)].copy()
        p = gg.pivot(
            index="challenge_class", columns="system",
            values="overall_correct_resolution_rate",
        ).reindex(columns=SYSTEMS)

        order_df = p.copy()
        order_df["_m"] = order_df.mean(axis=1)
        order_df["_n"] = [n_map.get(str(i), 0) for i in order_df.index]
        p = order_df.sort_values(["_m", "_n"], ascending=[True, False]).drop(columns=["_m", "_n"])
        if p.empty:
            continue

        arr = p.to_numpy(float) * 100
        row_labels = [f"{str(i).replace('_', ' ')} (n={n_map.get(str(i), 0)})" for i in p.index]
        fig, ax = plt.subplots(figsize=(8.5, max(5.3, .46*len(p)+1.7)))
        im = ax.imshow(arr, aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(np.arange(len(SYSTEMS)))
        ax.set_xticklabels([LABELS[s] for s in SYSTEMS])
        ax.set_yticks(np.arange(len(p.index)))
        ax.set_yticklabels(row_labels)

        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isfinite(arr[i, j]):
                    ax.text(j, i, f"{arr[i,j]:.1f}", ha="center", va="center", fontsize=8.5)

        ax.set_title(
            f"{title_prefix(status)}: challenge-class overall correct resolution\n"
            f"(classes with n ≥ {MIN_CHALLENGE_N})"
        )
        cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.03)
        cb.set_label("Overall correct resolution (%)")
        fig.tight_layout()
        save(fig, f"FigS02_{bench.lower()}_challenge_class_heatmap")


def main() -> None:
    if not PERF.exists():
        raise FileNotFoundError("Run 06_make_identity_results_FAIR_V5.py first")

    meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}
    status = meta.get("analysis_status", "UNKNOWN")
    perf = pd.read_csv(PERF)
    parent = pd.read_csv(PARENT) if PARENT.exists() and PARENT.stat().st_size > 0 else pd.DataFrame()
    diff = pd.read_csv(DIFF) if DIFF.exists() and DIFF.stat().st_size > 0 else pd.DataFrame()
    challenge = pd.read_csv(CHALLENGE) if CHALLENGE.exists() and CHALLENGE.stat().st_size > 0 else pd.DataFrame()
    pred = pd.read_csv(PRED) if PRED.exists() and PRED.stat().st_size > 0 else pd.DataFrame()

    overall_resolution_fig(perf, status)
    coverage_accuracy_tradeoff_fig(perf, status)
    case_outcome_fig(pred, status)
    difficulty_fig(diff, status)
    parent_heatmap(parent, status)
    challenge_heatmap(challenge, status)

    print(f"Analysis status shown on figures: {status}")
    print(f"Figures saved to: {OUT}")


if __name__ == "__main__":
    main()
