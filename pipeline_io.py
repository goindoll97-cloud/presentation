# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import pickle
import pandas as pd

ROOT = Path(__file__).resolve().parent
INTERMEDIATE = ROOT / "intermediate"
PAPER_OUTPUT = ROOT / "paper_output"
FIGURE_DIR = PAPER_OUTPUT / "figures"
for p in (INTERMEDIATE, PAPER_OUTPUT, FIGURE_DIR):
    p.mkdir(parents=True, exist_ok=True)

def dump_pickle(name: str, obj):
    path = INTERMEDIATE / name
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path

def load_pickle(name: str, required: bool = True):
    path = INTERMEDIATE / name
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"필수 중간결과가 없습니다: {path}\n"
                "앞 단계 스크립트를 먼저 실행하세요."
            )
        return None
    with path.open("rb") as f:
        return pickle.load(f)

def save_csv(df: pd.DataFrame, name: str):
    path = INTERMEDIATE / name
    if df is None:
        df = pd.DataFrame()
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path

def dataframe_or_empty(x):
    return x if isinstance(x, pd.DataFrame) else pd.DataFrame()
