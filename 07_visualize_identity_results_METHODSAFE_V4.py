# -*- coding: utf-8 -*-
"""Study 2 / Step 07 - publication-ready visualization.

Reads Step 05/06 METHOD-SAFE outputs from ./intermediate and writes figures to
./figures_study2. Each figure is saved as both 600-dpi PNG and vector PDF.

Main figures
------------
Fig01_primary_performance
Fig02_secondary_performance
Fig03_outcome_composition
Fig04_runtime
Fig05_secondary_difficulty

Supplementary figures
---------------------
FigS01_v43_sensitivity
FigS02_parent_level_heatmap

Notes
-----
- Main inference uses only the frozen systems:
    LLM + DB, DB + RDKit, Hybrid
- RDKIT_V43_SENSITIVITY and HYBRID_V43_SENSITIVITY are kept strictly separate
  as post-preflight sensitivity analyses.
- Selective accuracy is calculated only among MATCH/NO_MATCH decisions.
- Overall correct resolution includes REVIEW in the denominator as unresolved.
"""

from __future__ import annotations

from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
INTER = ROOT / "intermediate"
OUT = ROOT / "figures_study2"
OUT.mkdir(parents=True, exist_ok=True)

PERF_MEAN = INTER / "06_methodsafe_performance_mean.csv"
PRED_FILE = INTER / "05_identity_system_predictions.csv"
RUNTIME_FILE = INTER / "05_identity_runtime_ppt_operational.csv"
DIFFICULTY_FILE = INTER / "06_methodsafe_secondary_performance_by_difficulty.csv"
PARENT_FILE = INTER / "06_methodsafe_performance_by_parent.csv"

MAIN_SYSTEMS = ["CLAUDE_DB", "RDKIT_CAS_LOOKUP", "HYBRID_OPERATIONAL"]
MAIN_LABELS = {
    "CLAUDE_DB": "LLM + DB",
    "RDKIT_CAS_LOOKUP": "DB + RDKit",
    "HYBRID_OPERATIONAL": "Hybrid",
}
SENS_SYSTEMS = ["RDKIT_CAS_LOOKUP", "RDKIT_V43_SENSITIVITY", "HYBRID_OPERATIONAL", "HYBRID_V43_SENSITIVITY"]
SENS_LABELS = {
    "RDKIT_CAS_LOOKUP": "Frozen RDKit",
    "RDKIT_V43_SENSITIVITY": "RDKit V4.3",
    "HYBRID_OPERATIONAL": "Frozen Hybrid",
    "HYBRID_V43_SENSITIVITY": "Hybrid V4.3",
}
BENCH_LABELS = {
    "PRIMARY_SOURCE_CURATED": "Primary source-curated benchmark",
    "SECONDARY_STRUCTURE_ANCHORED": "Secondary structure-anchored benchmark",
}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def clean_bool(x) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def savefig(fig: plt.Figure, stem: str) -> None:
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {png.name}")
    print(f"[SAVED] {pdf.name}")


def add_bar_labels(ax, bars, fmt="{:.1f}", suffix="") -> None:
    for bar in bars:
        h = bar.get_height()
        if not np.isfinite(h):
            continue
        ax.annotate(
            fmt.format(h) + suffix,
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def figure_performance(perf: pd.DataFrame, benchmark_set: str, stem: str) -> None:
    g = perf[
        perf["benchmark_set"].eq(benchmark_set)
        & perf["system"].isin(MAIN_SYSTEMS)
    ].copy()

    metrics = [
        ("coverage_mean", "Coverage"),
        ("accuracy_mean", "Selective accuracy"),
        ("balanced_accuracy_mean", "Balanced accuracy"),
        ("overall_correct_resolution_rate_mean", "Overall correct resolution"),
    ]

    # A dot plot is used instead of truncated-axis bars so that small differences
    # near 100% remain visible without visually exaggerating bar lengths.
    y = np.arange(len(metrics))
    offsets = np.linspace(-0.18, 0.18, len(MAIN_SYSTEMS))
    fig, ax = plt.subplots(figsize=(8.6, 4.9))

    all_values = []
    markers = ["o", "s", "D"]
    for k, s in enumerate(MAIN_SYSTEMS):
        row = g[g["system"].eq(s)]
        vals = [float(row[col].iloc[0]) * 100 if len(row) else np.nan for col, _ in metrics]
        all_values.extend([v for v in vals if np.isfinite(v)])
        ax.scatter(vals, y + offsets[k], s=58, marker=markers[k], label=MAIN_LABELS[s], zorder=3)
        for val, yy in zip(vals, y + offsets[k]):
            if np.isfinite(val):
                ax.annotate(f"{val:.1f}", (val, yy), xytext=(5, 0), textcoords="offset points",
                            va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label in metrics])
    ax.invert_yaxis()
    ax.set_xlabel("Performance (%)")
    if all_values:
        lo = max(0.0, math.floor(min(all_values) - 2.0))
        ax.set_xlim(lo, 101.5)
    ax.set_title(BENCH_LABELS[benchmark_set])
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.grid(axis="x", linewidth=0.5, alpha=0.35)
    fig.tight_layout()
    savefig(fig, stem)


def figure_outcome_composition(pred: pd.DataFrame) -> None:
    # Repeats were perfectly consistent for the LLM arm; repeat 1 gives one row
    # per operational case and avoids visually triple-counting repeated runs.
    p = pred[
        pred["system"].isin(MAIN_SYSTEMS)
        & pred["repeat"].eq(1)
    ].copy()

    rows = []
    for benchmark_set, flag_col in [
        ("PRIMARY_SOURCE_CURATED", "in_primary_source_curated"),
        ("SECONDARY_STRUCTURE_ANCHORED", "in_secondary_structure_anchored"),
    ]:
        q = p[p[flag_col].map(clean_bool)].copy()
        for system in MAIN_SYSTEMS:
            s = q[q["system"].eq(system)].copy()
            correct = int((s["decided"].map(clean_bool) & s["correct"].map(clean_bool)).sum())
            incorrect = int((s["decided"].map(clean_bool) & ~s["correct"].map(clean_bool)).sum())
            review = int((~s["decided"].map(clean_bool)).sum())
            rows.append({
                "benchmark_set": benchmark_set,
                "system": system,
                "correct": correct,
                "incorrect": incorrect,
                "review": review,
                "n": len(s),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "Fig03_outcome_composition_data.csv", index=False, encoding="utf-8-sig")

    labels = []
    corrects, incorrects, reviews = [], [], []
    for b in ["PRIMARY_SOURCE_CURATED", "SECONDARY_STRUCTURE_ANCHORED"]:
        short = "Primary" if b.startswith("PRIMARY") else "Secondary"
        for s in MAIN_SYSTEMS:
            r = out[(out["benchmark_set"].eq(b)) & (out["system"].eq(s))].iloc[0]
            labels.append(f"{short}\n{MAIN_LABELS[s]}")
            corrects.append(int(r["correct"]))
            incorrects.append(int(r["incorrect"]))
            reviews.append(int(r["review"]))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.2, 5.5))
    b1 = ax.bar(x, corrects, label="Correct")
    b2 = ax.bar(x, incorrects, bottom=corrects, label="Incorrect")
    bottoms = np.array(corrects) + np.array(incorrects)
    b3 = ax.bar(x, reviews, bottom=bottoms, label="Review / unresolved")

    for bars, vals, bottoms_for_text in [
        (b1, corrects, np.zeros(len(x))),
        (b2, incorrects, np.array(corrects)),
        (b3, reviews, bottoms),
    ]:
        for bar, v, bot in zip(bars, vals, bottoms_for_text):
            if v <= 0:
                continue
            ax.text(bar.get_x() + bar.get_width()/2, bot + v/2, str(v), ha="center", va="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Number of cases")
    ax.set_title("Outcome composition by benchmark and system (repeat 1)")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.grid(axis="y", linewidth=0.5, alpha=0.3)
    fig.tight_layout()
    savefig(fig, "Fig03_outcome_composition")


def figure_runtime(runtime: pd.DataFrame) -> None:
    order = ["LLM + DB", "DB + RDKit", "Hybrid"]
    g = runtime[runtime["presentation_system"].isin(order)].copy()
    g["presentation_system"] = pd.Categorical(g["presentation_system"], order, ordered=True)
    g = g.sort_values("presentation_system")

    x = np.arange(len(order))
    vals = g["mean_sec_per_case"].astype(float).to_numpy()
    med = g["median_sec_per_case"].astype(float).to_numpy()
    p25 = g["p25_sec_per_case"].astype(float).to_numpy()
    p75 = g["p75_sec_per_case"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.bar(x, vals, width=0.58, label="Mean latency")
    # IQR is shown as asymmetric error bars around the median, which is more
    # informative than SD for the strongly skewed DB/network timing distribution.
    ax.errorbar(
        x,
        med,
        yerr=np.vstack([med - p25, p75 - med]),
        fmt="o",
        capsize=4,
        linewidth=1,
        label="Median and IQR",
    )
    add_bar_labels(ax, bars, fmt="{:.2f}", suffix=" s")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Seconds per operational case")
    ax.set_title("End-to-end per-case runtime")
    ax.legend(frameon=False)
    ax.grid(axis="y", linewidth=0.5, alpha=0.3)
    fig.tight_layout()
    savefig(fig, "Fig04_runtime")


def figure_secondary_difficulty(diff: pd.DataFrame) -> None:
    g = diff[diff["system"].isin(MAIN_SYSTEMS)].copy()
    # Mean over repeats. In the final run the three repeats are consistent, but
    # averaging keeps the plotting code correct if future repeats vary.
    agg = (
        g.groupby(["difficulty", "system"], as_index=False)
        .agg(
            overall_correct_resolution=("overall_correct_resolution_rate", "mean"),
            coverage=("coverage", "mean"),
        )
    )

    difficulty_order = [d for d in ["EASY", "MODERATE", "HARD"] if d in set(agg["difficulty"])]
    x = np.arange(len(difficulty_order))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for j, s in enumerate(MAIN_SYSTEMS):
        vals = []
        for d in difficulty_order:
            row = agg[(agg["difficulty"].eq(d)) & (agg["system"].eq(s))]
            vals.append(float(row["overall_correct_resolution"].iloc[0]) * 100 if len(row) else np.nan)
        bars = ax.bar(x + (j - 1) * width, vals, width, label=MAIN_LABELS[s])
        add_bar_labels(ax, bars, fmt="{:.1f}")

    ax.set_xticks(x)
    ax.set_xticklabels([d.title() for d in difficulty_order])
    ax.set_ylabel("Overall correct resolution (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Secondary benchmark performance by case difficulty")
    ax.legend(frameon=False)
    ax.grid(axis="y", linewidth=0.5, alpha=0.3)
    fig.tight_layout()
    savefig(fig, "Fig05_secondary_difficulty")


def figure_v43_sensitivity(perf: pd.DataFrame) -> None:
    g = perf[
        perf["benchmark_set"].eq("SECONDARY_STRUCTURE_ANCHORED")
        & perf["system"].isin(SENS_SYSTEMS)
    ].copy()

    metrics = [
        ("recall_mean", "Recall"),
        ("false_safe_rate_mean", "False-safe rate"),
        ("overall_correct_resolution_rate_mean", "Overall correct\nresolution"),
    ]
    x = np.arange(len(SENS_SYSTEMS))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    for j, (col, label) in enumerate(metrics):
        vals = []
        for s in SENS_SYSTEMS:
            row = g[g["system"].eq(s)]
            vals.append(float(row[col].iloc[0]) * 100 if len(row) else np.nan)
        bars = ax.bar(x + (j - 1) * width, vals, width, label=label)
        add_bar_labels(ax, bars, fmt="{:.1f}")

    ax.set_xticks(x)
    ax.set_xticklabels([SENS_LABELS[s] for s in SENS_SYSTEMS])
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Post-preflight V4.3 sensitivity analysis on the secondary benchmark")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(axis="y", linewidth=0.5, alpha=0.3)
    fig.tight_layout()
    savefig(fig, "FigS01_v43_sensitivity")


def figure_parent_heatmap(parent: pd.DataFrame) -> None:
    g = parent[
        parent["benchmark_set"].eq("SECONDARY_STRUCTURE_ANCHORED")
        & parent["system"].isin(MAIN_SYSTEMS)
    ].copy()

    agg = (
        g.groupby(["reference_parent_name", "system"], as_index=False)
        .agg(overall_correct_resolution=("overall_correct_resolution_rate", "mean"))
    )
    pivot = agg.pivot(index="reference_parent_name", columns="system", values="overall_correct_resolution")
    pivot = pivot.reindex(columns=MAIN_SYSTEMS)
    # Keep a stable, interpretable order: lower-performing parents first.
    pivot = pivot.assign(_mean=pivot.mean(axis=1)).sort_values("_mean").drop(columns="_mean")
    pivot.to_csv(OUT / "FigS02_parent_level_heatmap_data.csv", encoding="utf-8-sig")

    arr = pivot.to_numpy(dtype=float) * 100
    height = max(5.5, 0.34 * len(pivot.index) + 1.8)
    fig, ax = plt.subplots(figsize=(7.2, height))
    im = ax.imshow(arr, aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(MAIN_SYSTEMS)))
    ax.set_xticklabels([MAIN_LABELS[s] for s in MAIN_SYSTEMS])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_title("Secondary benchmark: parent-level overall correct resolution")

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            txt = "NA" if not np.isfinite(val) else f"{val:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Overall correct resolution (%)")
    fig.tight_layout()
    savefig(fig, "FigS02_parent_level_heatmap")


def main() -> None:
    for p in [PERF_MEAN, PRED_FILE, RUNTIME_FILE, DIFFICULTY_FILE, PARENT_FILE]:
        require_file(p)

    perf = pd.read_csv(PERF_MEAN)
    pred = pd.read_csv(PRED_FILE)
    runtime = pd.read_csv(RUNTIME_FILE)
    diff = pd.read_csv(DIFFICULTY_FILE)
    parent = pd.read_csv(PARENT_FILE)

    # Guard against accidentally plotting the earlier failed Claude run.
    claude_primary = perf[
        perf["benchmark_set"].eq("PRIMARY_SOURCE_CURATED")
        & perf["system"].eq("CLAUDE_DB")
    ]
    if len(claude_primary) and float(claude_primary["coverage_mean"].iloc[0]) == 0.0:
        raise RuntimeError(
            "CLAUDE_DB coverage is 0. This appears to be the earlier failed API run. "
            "Re-run Step 05 successfully and then Step 06 before plotting."
        )

    figure_performance(perf, "PRIMARY_SOURCE_CURATED", "Fig01_primary_performance")
    figure_performance(perf, "SECONDARY_STRUCTURE_ANCHORED", "Fig02_secondary_performance")
    figure_outcome_composition(pred)
    figure_runtime(runtime)
    figure_secondary_difficulty(diff)
    figure_v43_sensitivity(perf)
    figure_parent_heatmap(parent)

    print("\nDone.")
    print(f"Figures: {OUT}")
    print("Main figures: Fig01-Fig05")
    print("Supplementary: FigS01-FigS02")


if __name__ == "__main__":
    main()
