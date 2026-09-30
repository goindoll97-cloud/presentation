# -*- coding: utf-8 -*-
"""Study 2 / Step 07 FAIR V5 publication figures."""
from __future__ import annotations

import json
import math
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
META = INTER / "05_v5_run_metadata.json"
SYSTEMS = ["CLAUDE_DB", "RDKIT_SALT_AWARE_V5", "HYBRID_SALT_AWARE_V5"]
LABELS = {
    "CLAUDE_DB": "LLM + DB",
    "RDKIT_SALT_AWARE_V5": "DB + salt-aware RDKit",
    "HYBRID_SALT_AWARE_V5": "Hybrid",
}
MARKERS = ["o", "s", "D"]


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {stem}.png/.pdf")


def is_final(status):
    return str(status).startswith("FINAL_")


def wilson_interval(k, n, z=1.959963984540054):
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def perf_fig(df, status):
    metrics = [
        ("coverage", "Coverage"),
        ("accuracy", "Selective accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("overall_correct_resolution_rate", "Overall correct resolution"),
    ]
    for bench, g0 in df.groupby("benchmark_set", sort=False):
        y = np.arange(len(metrics))
        offs = np.linspace(-0.18, 0.18, len(SYSTEMS))
        fig, ax = plt.subplots(figsize=(9.0, 5.4))

        for j, s in enumerate(SYSTEMS):
            row_df = g0[g0.system.eq(s)]
            if row_df.empty:
                continue
            row = row_df.iloc[0]
            vals, lows, highs = [], [], []
            for c, _ in metrics:
                v = float(row[c]) * 100
                vals.append(v)
                if c == "coverage":
                    lo, hi = wilson_interval(int(row["n_decided"]), int(row["n_total"]))
                    lows.append(lo * 100); highs.append(hi * 100)
                elif c == "accuracy":
                    lows.append(float(row["accuracy_ci95_low"]) * 100)
                    highs.append(float(row["accuracy_ci95_high"]) * 100)
                elif c == "overall_correct_resolution_rate":
                    lows.append(float(row["overall_correct_resolution_ci95_low"]) * 100)
                    highs.append(float(row["overall_correct_resolution_ci95_high"]) * 100)
                else:
                    lows.append(np.nan); highs.append(np.nan)

            yy = y + offs[j]
            ax.scatter(vals, yy, s=72, marker=MARKERS[j], label=LABELS[s], zorder=4)
            for v, lo, hi, yv in zip(vals, lows, highs, yy):
                if np.isfinite(lo) and np.isfinite(hi):
                    ax.errorbar(
                        v, yv,
                        xerr=np.array([[v - lo], [hi - v]]),
                        fmt="none", capsize=3, linewidth=1.2, zorder=3,
                    )
                if np.isfinite(v):
                    ax.annotate(
                        f"{v:.1f}", (v, yv), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=9,
                    )

        ax.set_yticks(y)
        ax.set_yticklabels([label for _, label in metrics], fontsize=11)
        ax.get_yticklabels()[-1].set_fontweight("bold")
        ax.invert_yaxis()
        if is_final(status):
            ax.set_xlim(88, 103)
            ax.set_xticks(np.arange(88, 103, 2))
            ax.set_title("Performance on the independent final holdout", fontsize=14, pad=12)
        else:
            ax.set_xlim(0, 105)
            ax.set_title("Performance in the development / exploratory reanalysis", fontsize=14, pad=12)
        ax.set_xlabel("Performance (%)", fontsize=11)
        ax.grid(axis="x", alpha=0.25)
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=10)
        ax.text(
            0.0, -0.22,
            "Horizontal bars show 95% Wilson confidence intervals for coverage, selective accuracy, "
            "and overall correct resolution.",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
        )
        fig.subplots_adjust(left=0.27, right=0.97, top=0.88, bottom=0.28)
        save(fig, f"Fig01_{bench.lower()}_performance")


def parent_heatmap(df, status):
    if df.empty or "reference_parent_name" not in df.columns:
        return
    for bench, g in df.groupby("benchmark_set", sort=False):
        p = (
            g.pivot(index="reference_parent_name", columns="system", values="overall_correct_resolution_rate")
             .reindex(columns=SYSTEMS)
        )
        if p.empty:
            continue
        p = p.assign(_m=p.mean(axis=1)).sort_values("_m").drop(columns="_m")
        arr = p.to_numpy(float) * 100
        fig, ax = plt.subplots(figsize=(8.2, max(5.8, 0.42 * len(p) + 1.8)))
        im = ax.imshow(arr, aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(np.arange(len(SYSTEMS)))
        ax.set_xticklabels([LABELS[s] for s in SYSTEMS], fontsize=10)
        ax.set_yticks(np.arange(len(p.index)))
        ax.set_yticklabels(p.index, fontsize=10)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isfinite(arr[i, j]):
                    ax.text(j, i, f"{arr[i, j]:.1f}", ha="center", va="center", fontsize=9)
        if is_final(status):
            ax.set_title(
                "Parent-level overall correct resolution on the independent holdout",
                fontsize=13, pad=12,
            )
        else:
            ax.set_title(
                "Parent-level overall correct resolution in the development reanalysis",
                fontsize=13, pad=12,
            )
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
        cb.set_label("Overall correct resolution (%)", fontsize=10)
        fig.tight_layout()
        save(fig, f"FigS01_{bench.lower()}_parent_heatmap")


def difficulty_fig(df, status):
    if df.empty:
        return
    for bench, g in df.groupby("benchmark_set", sort=False):
        raw_levels = set(g.difficulty.astype(str).str.upper())
        levels = [x for x in ["EASY", "MEDIUM", "MODERATE", "HARD"] if x in raw_levels]
        if not levels:
            continue
        x = np.arange(len(levels)); width = 0.24
        fig, ax = plt.subplots(figsize=(8.4, 5.0))
        for j, s in enumerate(SYSTEMS):
            vals = []
            for d in levels:
                r = g[g.system.eq(s) & g.difficulty.astype(str).str.upper().eq(d)]
                vals.append(float(r.overall_correct_resolution_rate.iloc[0]) * 100 if len(r) else np.nan)
            bars = ax.bar(x + (j - 1) * width, vals, width, label=LABELS[s])
            for b, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([d.title() for d in levels])
        ax.set_ylim(0, 110)
        ax.set_ylabel("Overall correct resolution (%)")
        ax.set_title(
            "Overall correct resolution by difficulty on the independent holdout"
            if is_final(status)
            else "Overall correct resolution by difficulty in the development reanalysis"
        )
        ax.grid(axis="y", alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        save(fig, f"Fig02_{bench.lower()}_difficulty")


def main():
    if not PERF.exists():
        raise FileNotFoundError("Run 06_make_identity_results_FAIR_V5.py first")
    meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}
    status = meta.get("analysis_status", "UNKNOWN")
    perf = pd.read_csv(PERF)
    parent = pd.read_csv(PARENT) if PARENT.exists() and PARENT.stat().st_size > 0 else pd.DataFrame()
    diff = pd.read_csv(DIFF) if DIFF.exists() and DIFF.stat().st_size > 0 else pd.DataFrame()
    perf_fig(perf, status)
    parent_heatmap(parent, status)
    difficulty_fig(diff, status)
    print(f"Analysis status shown on figures: {status}")


if __name__ == "__main__":
    main()
