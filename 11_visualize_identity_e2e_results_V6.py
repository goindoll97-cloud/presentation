# -*- coding: utf-8 -*-
"""Publication figures for Study 2 V6 LLM+PubChem vs Hybrid+PubChem."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
INTER = ROOT / "intermediate_v6"
OUT = ROOT / "figures_study2_v6"
OUT.mkdir(parents=True, exist_ok=True)
PERF = INTER / "10_v6_performance.csv"
EFF = INTER / "10_v6_efficiency_summary.csv"
PRED = INTER / "09_v6_system_predictions_caselevel.csv"
DIFF = INTER / "10_v6_performance_by_difficulty.csv"
PARENT = INTER / "10_v6_performance_by_parent.csv"
SYSTEMS = ["LLM_PUBCHEM", "HYBRID_PUBCHEM"]
LABELS = {"LLM_PUBCHEM": "LLM + PubChem", "HYBRID_PUBCHEM": "Hybrid + PubChem"}


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {stem}.png/.pdf")


def fig01_performance(perf):
    p = perf.set_index("system").reindex(SYSTEMS)
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    x = np.arange(len(SYSTEMS)); w = 0.32
    vals1 = p["coverage"].to_numpy(float) * 100
    vals2 = p["overall_correct_resolution_rate"].to_numpy(float) * 100
    b1 = ax.bar(x-w/2, vals1, w, label="Coverage")
    b2 = ax.bar(x+w/2, vals2, w, label="Overall correct resolution")
    for bars, vals in [(b1, vals1), (b2, vals2)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+0.8, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[s] for s in SYSTEMS])
    ax.set_ylim(0, 108); ax.set_ylabel("Performance (%)")
    ax.set_title("End-to-end regulatory identity performance")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); save(fig, "Fig01_e2e_performance")


def fig02_outcomes(pred):
    rows=[]
    for s in SYSTEMS:
        g=pred[pred.system.eq(s)]
        correct=int(g.overall_correct_resolution.astype(str).str.lower().isin(["true","1"]).sum())
        review=int(g.decision.astype(str).str.upper().eq("REVIEW").sum())
        incorrect=len(g)-correct-review
        rows.append((correct, review, incorrect))
    arr=np.array(rows)
    fig, ax=plt.subplots(figsize=(7.4,4.7)); x=np.arange(len(SYSTEMS)); bottom=np.zeros(len(SYSTEMS))
    names=["Correct","REVIEW","Incorrect"]
    for j,name in enumerate(names):
        bars=ax.bar(x,arr[:,j],bottom=bottom,label=name)
        for i,b in enumerate(bars):
            if arr[i,j]>0: ax.text(b.get_x()+b.get_width()/2,bottom[i]+arr[i,j]/2,str(arr[i,j]),ha="center",va="center",fontsize=9)
        bottom+=arr[:,j]
    ax.set_xticks(x); ax.set_xticklabels([LABELS[s] for s in SYSTEMS]); ax.set_ylabel("Number of cases")
    ax.set_title("Case-level outcomes (n=72)"); ax.legend(frameon=False); ax.grid(axis="y",alpha=.2)
    fig.tight_layout(); save(fig,"Fig02_case_outcomes")


def fig03_efficiency(eff):
    wanted=["LLM calls","Input tokens","Output tokens","API cost (USD)","LLM API time (s)"]
    e=eff[eff.metric.isin(wanted)].copy()
    e["LLM_normalized"]=100.0
    e["Hybrid_normalized"]=(pd.to_numeric(e["HYBRID_PUBCHEM"],errors="coerce")/pd.to_numeric(e["LLM_PUBCHEM"],errors="coerce"))*100
    x=np.arange(len(e)); w=.34
    fig,ax=plt.subplots(figsize=(9.0,4.9))
    b1=ax.bar(x-w/2,e["LLM_normalized"],w,label="LLM + PubChem")
    b2=ax.bar(x+w/2,e["Hybrid_normalized"],w,label="Hybrid + PubChem")
    for b,v in zip(b2,e["Hybrid_normalized"]):
        if np.isfinite(v): ax.text(b.get_x()+b.get_width()/2,v+2,f"{v:.1f}%",ha="center",fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(e.metric,rotation=20,ha="right")
    ax.set_ylabel("Relative resource use (LLM + PubChem = 100%)"); ax.set_ylim(0,115)
    ax.set_title("Operational efficiency of selective LLM use")
    ax.legend(frameon=False); ax.grid(axis="y",alpha=.25)
    fig.tight_layout(); save(fig,"Fig03_operational_efficiency")


def fig04_difficulty(diff):
    levels=[x for x in ["easy","medium","hard"] if x in set(diff.difficulty.astype(str).str.lower())]
    if not levels: return
    fig,ax=plt.subplots(figsize=(7.5,4.7)); x=np.arange(len(levels)); w=.34
    for j,s in enumerate(SYSTEMS):
        vals=[]; ns=[]
        for d in levels:
            r=diff[(diff.system.eq(s)) & (diff.difficulty.astype(str).str.lower().eq(d))]
            vals.append(float(r.overall_correct_resolution_rate.iloc[0])*100 if len(r) else np.nan)
            ns.append(int(r.n_total.iloc[0]) if len(r) else 0)
        bars=ax.bar(x+(j-.5)*w,vals,w,label=LABELS[s])
        for b,v in zip(bars,vals):
            if np.isfinite(v): ax.text(b.get_x()+b.get_width()/2,v+1,f"{v:.1f}",ha="center",fontsize=8)
    counts=[]
    for d in levels:
        r=diff[diff.difficulty.astype(str).str.lower().eq(d)]
        counts.append(int(r.n_total.iloc[0]) if len(r) else 0)
    ax.set_xticks(x); ax.set_xticklabels([f"{d.title()}\n(n={n})" for d,n in zip(levels,counts)])
    ax.set_ylim(0,110); ax.set_ylabel("Overall correct resolution (%)"); ax.set_title("Performance by case difficulty")
    ax.legend(frameon=False); ax.grid(axis="y",alpha=.25)
    fig.tight_layout(); save(fig,"Fig04_performance_by_difficulty")


def figs01_parent(parent):
    p=parent.pivot(index="reference_parent_name",columns="system",values="overall_correct_resolution_rate").reindex(columns=SYSTEMS)
    p=p.assign(_m=p.mean(axis=1)).sort_values("_m").drop(columns="_m"); arr=p.to_numpy(float)*100
    fig,ax=plt.subplots(figsize=(7.5,max(5.3,.45*len(p)+1.5))); im=ax.imshow(arr,aspect="auto",vmin=0,vmax=100)
    ax.set_xticks(np.arange(len(SYSTEMS))); ax.set_xticklabels([LABELS[s] for s in SYSTEMS])
    ax.set_yticks(np.arange(len(p))); ax.set_yticklabels(p.index)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i,j]): ax.text(j,i,f"{arr[i,j]:.1f}",ha="center",va="center",fontsize=8)
    ax.set_title("Parent-level overall correct resolution")
    cb=fig.colorbar(im,ax=ax,fraction=.04,pad=.03); cb.set_label("Overall correct resolution (%)")
    fig.tight_layout(); save(fig,"FigS01_parent_heatmap")


def main():
    for p in (PERF,EFF,PRED,DIFF,PARENT):
        if not p.exists(): raise FileNotFoundError(f"Run Step 10 first: missing {p}")
    perf=pd.read_csv(PERF); eff=pd.read_csv(EFF); pred=pd.read_csv(PRED); diff=pd.read_csv(DIFF); parent=pd.read_csv(PARENT)
    fig01_performance(perf); fig02_outcomes(pred); fig03_efficiency(eff); fig04_difficulty(diff); figs01_parent(parent)


if __name__=="__main__":
    main()
