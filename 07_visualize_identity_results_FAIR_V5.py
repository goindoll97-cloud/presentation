# -*- coding: utf-8 -*-
"""Study 2 / Step 07 FAIR V5 publication figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
INTER=ROOT/"intermediate"
OUT=ROOT/"figures_study2_v5"
OUT.mkdir(parents=True,exist_ok=True)
PERF=INTER/"06_fair_v5_performance_mean.csv"
PARENT=INTER/"06_fair_v5_performance_by_parent.csv"
DIFF=INTER/"06_fair_v5_performance_by_difficulty.csv"
META=INTER/"05_v5_run_metadata.json"
SYSTEMS=["CLAUDE_DB","RDKIT_SALT_AWARE_V5","HYBRID_SALT_AWARE_V5"]
LABELS={"CLAUDE_DB":"LLM + DB","RDKIT_SALT_AWARE_V5":"DB + salt-aware RDKit","HYBRID_SALT_AWARE_V5":"Hybrid"}


def save(fig,stem):
    fig.savefig(OUT/f"{stem}.png",dpi=600,bbox_inches="tight")
    fig.savefig(OUT/f"{stem}.pdf",bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {stem}.png/.pdf")


def title_prefix(status):
    return "Independent final holdout" if str(status).startswith("FINAL_") else "Development / exploratory reanalysis"


def perf_fig(df,status):
    metrics=[("coverage","Coverage"),("accuracy","Selective accuracy"),("balanced_accuracy","Balanced accuracy"),("overall_correct_resolution_rate","Overall correct resolution")]
    for bench,g0 in df.groupby("benchmark_set",sort=False):
        y=np.arange(len(metrics)); offs=np.linspace(-.18,.18,len(SYSTEMS)); fig,ax=plt.subplots(figsize=(9.2,5.0)); markers=["o","s","D"]
        for j,s in enumerate(SYSTEMS):
            row=g0[g0.system.eq(s)]; vals=[float(row[c].iloc[0])*100 if len(row) and c in row else np.nan for c,_ in metrics]
            ax.scatter(vals,y+offs[j],s=60,marker=markers[j],label=LABELS[s],zorder=3)
            for v,yy in zip(vals,y+offs[j]):
                if np.isfinite(v): ax.annotate(f"{v:.1f}",(v,yy),xytext=(5,0),textcoords="offset points",va="center",fontsize=8)
        ax.set_yticks(y); ax.set_yticklabels([x[1] for x in metrics]); ax.invert_yaxis(); ax.set_xlim(0,105); ax.set_xlabel("Performance (%)"); ax.set_title(f"{title_prefix(status)}\n{bench}"); ax.grid(axis="x",alpha=.3); ax.legend(frameon=False,ncol=3,loc="upper center",bbox_to_anchor=(.5,-.15)); fig.tight_layout(); save(fig,f"Fig01_{bench.lower()}_performance")


def parent_heatmap(df,status):
    if df.empty or "reference_parent_name" not in df.columns: return
    for bench,g in df.groupby("benchmark_set",sort=False):
        p=g.pivot(index="reference_parent_name",columns="system",values="overall_correct_resolution_rate").reindex(columns=SYSTEMS)
        if p.empty: continue
        p=p.assign(_m=p.mean(axis=1)).sort_values("_m").drop(columns="_m"); arr=p.to_numpy(float)*100; fig,ax=plt.subplots(figsize=(8.0,max(5.5,.42*len(p)+1.8))); im=ax.imshow(arr,aspect="auto",vmin=0,vmax=100)
        ax.set_xticks(np.arange(len(SYSTEMS))); ax.set_xticklabels([LABELS[s] for s in SYSTEMS]); ax.set_yticks(np.arange(len(p.index))); ax.set_yticklabels(p.index)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isfinite(arr[i,j]): ax.text(j,i,f"{arr[i,j]:.1f}",ha="center",va="center",fontsize=8)
        ax.set_title(f"{title_prefix(status)}: parent-level overall correct resolution\n{bench}"); cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.03); cb.set_label("Overall correct resolution (%)"); fig.tight_layout(); save(fig,f"FigS01_{bench.lower()}_parent_heatmap")


def difficulty_fig(df,status):
    if df.empty: return
    for bench,g in df.groupby("benchmark_set",sort=False):
        levels=[x for x in ["EASY","MODERATE","HARD"] if x in set(g.difficulty.astype(str))]
        if not levels: continue
        x=np.arange(len(levels)); width=.24; fig,ax=plt.subplots(figsize=(8.4,5.0))
        for j,s in enumerate(SYSTEMS):
            vals=[]
            for d in levels:
                r=g[g.system.eq(s)&g.difficulty.astype(str).eq(d)]; vals.append(float(r.overall_correct_resolution_rate.iloc[0])*100 if len(r) else np.nan)
            bars=ax.bar(x+(j-1)*width,vals,width,label=LABELS[s])
            for b,v in zip(bars,vals):
                if np.isfinite(v): ax.text(b.get_x()+b.get_width()/2,v+1,f"{v:.1f}",ha="center",fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([d.title() for d in levels]); ax.set_ylim(0,110); ax.set_ylabel("Overall correct resolution (%)"); ax.set_title(f"{title_prefix(status)}: performance by difficulty\n{bench}"); ax.grid(axis="y",alpha=.3); ax.legend(frameon=False); fig.tight_layout(); save(fig,f"Fig02_{bench.lower()}_difficulty")


def main():
    if not PERF.exists(): raise FileNotFoundError("Run 06_make_identity_results_FAIR_V5.py first")
    meta=json.loads(META.read_text(encoding="utf-8")) if META.exists() else {}; status=meta.get("analysis_status","UNKNOWN"); perf=pd.read_csv(PERF); parent=pd.read_csv(PARENT) if PARENT.exists() and PARENT.stat().st_size>0 else pd.DataFrame(); diff=pd.read_csv(DIFF) if DIFF.exists() and DIFF.stat().st_size>0 else pd.DataFrame(); perf_fig(perf,status); parent_heatmap(parent,status); difficulty_fig(diff,status); print(f"Analysis status shown on figures: {status}")


if __name__=="__main__":
    main()
