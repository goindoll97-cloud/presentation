# -*- coding: utf-8 -*-
"""
01_build_official_benchmark.py
==============================
Research-only pipeline 1/3. No Streamlit/web UI.

Purpose
-------
1) Create data/intermediate/paper_output folders automatically.
2) Load the user's source-verified approved regulatory master.
   - If not present locally, download the approved master from the user's GitHub repository.
3) Query the National Law Information Center (법제처/국가법령정보센터) API when LAW_OC is set.
4) For selected chemical notices, retrieve the official JSON body and attached appendix sources.
5) For 인체등유해성물질, prefer the official machine-readable '목록(전체)' Excel workbook
   when supplied locally; otherwise retain PDF -> HWP/HWPX fallback for source audit.
6) Retain the grouped designation/CAS and flat sentence-style PDF parsers for other notices.
7) Never auto-promote newly parsed official-source rows to approved rules. They are saved as candidates/audit evidence only.
8) Build a fixed benchmark for the paper's primary endpoint:
      "Does this regulatory row require identity information beyond direct CAS matching?"

Important scientific boundary
-----------------------------
- The main benchmark reference is a source-derived, predefined operational reference,
  NOT an independent expert gold standard.
- The benchmark evaluates CAS-representation insufficiency / identity-scope detection,
  not a final legal conclusion by an LLM.
- Official API/PDF extraction is separately audited and does not overwrite the approved master.

Blank-folder use
----------------
Put only the three study scripts in a folder. This script will download the approved
seed master from the user's public GitHub repository when no local copy is found.
Internet access is therefore required on the first run unless the master is already local.

Optional environment variable
-----------------------------
LAW_OC=<your National Law Information Center Open API credential>
HUMAN_HAZARD_XLSX=<path to official 2026-5 인체등유해성물질 목록(전체).xlsx>
BENCHMARK_NEGATIVE_RATIO=1.0      # negatives per positive in the balanced primary benchmark
BENCHMARK_SEED=20260910

Dependencies
------------
pip install pandas numpy requests pdfplumber openpyxl matplotlib
# For legacy binary .hwp appendix fallback only:
pip install olefile

Run
---
python 01_build_official_benchmark.py
"""
from __future__ import annotations

import io
import json
import os
import re
import hashlib
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INTERMEDIATE = ROOT / "intermediate"
PAPER_OUTPUT = ROOT / "paper_output"
FIGURE_DIR = PAPER_OUTPUT / "figures"
for d in (DATA_DIR, INTERMEDIATE, PAPER_OUTPUT, FIGURE_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED_MASTER_LOCAL = DATA_DIR / "regulatory_rules_master_APPROVED.csv"
SEED_MASTER_URL = (
    "https://raw.githubusercontent.com/goindoll97-cloud/goindoll97/main/"
    "engine/data/regulatory_rules_master_APPROVED.csv"
)
BROAD_SALT_RULES_LOCAL = DATA_DIR / "broad_salt_rules.csv"
BROAD_SALT_RULES_URL = (
    "https://raw.githubusercontent.com/goindoll97-cloud/goindoll97/main/"
    "engine/data/broad_salt_rules.csv"
)
BROAD_SALT_VALIDATION_LOCAL = DATA_DIR / "broad_salt_validation_cases.csv"
BROAD_SALT_VALIDATION_URL = (
    "https://raw.githubusercontent.com/goindoll97-cloud/goindoll97/main/"
    "engine/data/broad_salt_validation_cases.csv"
)

LAW_BASE = "https://www.law.go.kr"
CAS_RE = re.compile(r"(?<!\d)(\d{2,7}-\d{2}-\d)(?!\d)")
DESIG_RE = re.compile(r"^0?\d{1,3}-\d+-\d+$")
INLINE_CODE_RE = re.compile(r"^(0?\d{1,3}-\d+-\d+)\s+(.*)$")
ITEM_RE = re.compile(r"^(\d{1,3})\s+(.*?)\s*(\d{2,7}-\d{2}-\d)\s*$")
CATCHALL_RE = re.compile(r"^(\d{1,3})\s+(.+?)\s*-?\s*$")
NUM_DASH_ONLY_RE = re.compile(r"^(\d{1,3})\s*-\s*$")

BENCHMARK_SEED = int(os.getenv("BENCHMARK_SEED", "20260910"))
NEGATIVE_RATIO = float(os.getenv("BENCHMARK_NEGATIVE_RATIO", "1.0"))

NOTICE_SPECS = [
    {
        "regime": "HUMAN_HAZARD",
        "title": "인체급성유해성물질, 인체만성유해성물질 및 생태유해성물질의 지정고시",
        "terms": ["인체급성유해성물질", "생태유해성물질"],
    },
    {
        "regime": "ACCIDENT_PREPAREDNESS",
        "title": "사고대비물질의 지정",
        "terms": ["사고대비물질", "지정"],
    },
    {
        "regime": "RESTRICTED_PROHIBITED",
        "title": "제한물질·금지물질의 지정",
        "terms": ["제한물질", "금지물질", "지정"],
    },
]

# Conservative semantic taxonomy.  A positive class means direct-CAS representation
# alone is insufficient for the operational screening endpoint.
SCOPE_ORDER = [
    "EXPLICIT_EXCEPTION",
    "REACTION_PRODUCT",
    "MIXTURE_OR_UVCB",
    "BROAD_SALT",
    "DERIVATIVE_FAMILY",
    "COMPOUND_GROUP",
    "STRUCTURAL_RANGE",
    "NO_DIRECT_CAS_OTHER",
    "DIRECT_CAS_ONLY",
]


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    return "" if s.lower() in {"nan", "none", "null", "<na>"} else s


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", clean(value)).strip()


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return clean(value).lower() in {"1", "true", "yes", "y", "예", "o"}


def download_seed(url: str, path: Path, timeout: int = 120) -> tuple[bool, str]:
    if path.exists() and path.stat().st_size > 50:
        return True, "LOCAL_EXISTING"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
        return True, "DOWNLOADED_FROM_USER_GITHUB"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def read_csv_robust(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def find_human_hazard_official_xlsx() -> Optional[Path]:
    """Locate the manually downloaded official '인체등유해성물질 목록(전체)' workbook.

    Preferred use:
        HUMAN_HAZARD_XLSX=<absolute or relative path>

    If the environment variable is not set, search the study root and data/
    directory for an .xlsx file whose name contains both '인체등유해성물질' and
    '목록'/'전체'. Temporary Excel lock files (~$...) are ignored.

    The workbook is used only for source audit. It never overwrites the frozen
    approved master or changes benchmark labels.
    """
    configured = clean(os.getenv("HUMAN_HAZARD_XLSX"))
    if not configured:
        for env_path in (ROOT / ".env", Path.cwd() / ".env"):
            if not env_path.exists():
                continue
            try:
                for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if "=" not in raw or raw.strip().startswith("#"):
                        continue
                    k, v = raw.split("=", 1)
                    if k.strip() == "HUMAN_HAZARD_XLSX":
                        configured = v.strip().strip('"').strip("'")
                        break
            except Exception:
                pass
            if configured:
                break
    if configured:
        p = Path(configured)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists() and p.is_file():
            return p.resolve()

    candidates: list[Path] = []
    for base in (ROOT, DATA_DIR):
        if not base.exists():
            continue
        for p in base.glob("*.xlsx"):
            name = p.name
            if name.startswith("~$"):
                continue
            if "인체등유해성물질" in name and "목록" in name and "전체" in name:
                candidates.append(p.resolve())

    if not candidates:
        return None

    # Prefer the most recent notice number encoded in the filename, then mtime.
    def _rank(p: Path) -> tuple[tuple[int, int], float]:
        return notice_key(p.name), p.stat().st_mtime

    return max(candidates, key=_rank)


def parse_human_hazard_official_xlsx(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse the official machine-readable human-hazard appendix workbook.

    Expected current layout (2026-5):
      A 고유번호
      B 소번호
      C 국문명
      D 영문명
      E CAS No.
      F 인체급성 유해성 혼합물 함량(%)
      G 인체만성 유해성 혼합물 함량(%)
      H 생태 유해성 혼합물 함량(%)
      I 유해성물질
      J 종전 유독물질 지정고시 혼합물 함량(%)

    The parser finds the header by content rather than relying on a fixed Excel
    row number. Blank designation cells are intentionally forward-filled only
    for source-audit identity, because the official workbook uses merged/grouped
    presentation for subrows.
    """
    raw = pd.read_excel(path, sheet_name=0, header=None, dtype=object, engine="openpyxl")
    if raw.empty:
        raise ValueError("OFFICIAL_XLSX_EMPTY")

    header_row: Optional[int] = None
    for i in range(min(20, len(raw))):
        row_text = " | ".join(normalize_text(v) for v in raw.iloc[i].tolist())
        if "고유번호" in row_text and re.search(r"CAS\s*No", row_text, flags=re.I):
            header_row = i
            break
    if header_row is None:
        raise ValueError("OFFICIAL_XLSX_HEADER_NOT_FOUND")

    # First actual regulatory row after the two-level header.
    data_start: Optional[int] = None
    for i in range(header_row + 1, min(header_row + 20, len(raw))):
        first = _normalize_hwp_dashes(normalize_text(raw.iat[i, 0]))
        if HWP_DESIG_RE.fullmatch(first):
            data_start = i
            break
    if data_start is None:
        raise ValueError("OFFICIAL_XLSX_FIRST_DESIGNATION_NOT_FOUND")

    rows: list[dict[str, Any]] = []
    current_designation = ""
    inferred_rows = 0
    deleted_rows = 0

    for i in range(data_start, len(raw)):
        vals = [raw.iat[i, j] if j < raw.shape[1] else None for j in range(10)]
        if all(not clean(v) for v in vals):
            continue

        raw_designation = _normalize_hwp_dashes(normalize_text(vals[0]))
        if raw_designation and HWP_DESIG_RE.fullmatch(raw_designation):
            current_designation = raw_designation
            designation_inferred = False
        elif not raw_designation and current_designation:
            designation_inferred = True
            inferred_rows += 1
        else:
            # Footer/note rows are not regulatory rows.
            continue

        name_ko = normalize_text(vals[2])
        name_en = normalize_text(vals[3])
        cas_cell = _normalize_hwp_dashes(normalize_text(vals[4]))
        hazard_type = normalize_text(vals[8])
        is_deleted = hazard_type == "삭제"
        if is_deleted:
            deleted_rows += 1

        # Preserve '-' as no-CAS information in the source snapshot but expose
        # an empty audit CAS so split_cas() behaves consistently.
        audit_cas = "" if cas_cell in {"", "-"} else cas_cell

        rows.append({
            "designation_id": current_designation,
            "designation_id_inferred": designation_inferred,
            "sub_no": normalize_text(vals[1]),
            "item_no": normalize_text(vals[1]),
            "substance_name": name_ko,
            "substance_name_en": name_en,
            "cas": audit_cas,
            "cas_source_cell": cas_cell,
            "acute_threshold_pct": vals[5],
            "chronic_threshold_pct": vals[6],
            "eco_threshold_pct": vals[7],
            "hazard_type": hazard_type,
            "former_toxic_threshold_pct": vals[9],
            "is_deleted": bool(is_deleted),
            "source_excel_row": int(i + 1),
            "is_catch_all": False,
            "group_title": name_ko,
            "concentration_threshold_pct": None,
            "concentration_operator": "",
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("OFFICIAL_XLSX_NO_REGULATORY_ROWS")

    file_notice = ""
    m = re.search(r"(20\d{2})\s*[-–_]\s*(\d+)", path.name)
    if m:
        file_notice = f"{m.group(1)}-{int(m.group(2))}"

    meta = {
        "official_xlsx_path": str(path),
        "official_xlsx_name": path.name,
        "official_xlsx_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "official_xlsx_notice": file_notice,
        "official_xlsx_rows": int(len(out)),
        "official_xlsx_active_rows": int((~out["is_deleted"]).sum()),
        "official_xlsx_deleted_rows": int(out["is_deleted"].sum()),
        "official_xlsx_inferred_designation_rows": int(out["designation_id_inferred"].sum()),
        "official_xlsx_unique_designations": int(out["designation_id"].nunique()),
        "official_xlsx_no_cas_rows": int((out["cas"] == "").sum()),
    }
    return out, meta


def first_existing(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def split_cas(value: Any) -> list[str]:
    vals = CAS_RE.findall(clean(value))
    return list(dict.fromkeys(vals))


def notice_key(label: Any) -> tuple[int, int]:
    m = re.search(r"(20\d{2})\s*[-–]\s*(\d+)", clean(label))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def latest_notice_from_master(master: pd.DataFrame) -> str:
    col = first_existing(master, ["notice", "source_reference", "official_notice"])
    if col is None:
        return ""
    vals = [clean(v) for v in master[col].dropna().unique() if notice_key(v) != (0, 0)]
    return max(vals, key=notice_key) if vals else ""


def recursive_dicts(obj: Any) -> Iterable[dict]:
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from recursive_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from recursive_dicts(value)


def normalize_title(value: Any) -> str:
    return re.sub(r"[\s·ㆍ\-–—,]", "", clean(value)).lower()


def title_matches(candidate: str, spec: dict) -> bool:
    c = normalize_title(candidate)
    return c == normalize_title(spec["title"]) or all(
        normalize_title(t) in c for t in spec["terms"]
    )


def get_law_oc() -> str:
    # Keep credentials out of outputs.
    for env_path in (ROOT / ".env", Path.cwd() / ".env"):
        if env_path.exists():
            try:
                for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if "=" not in raw or raw.strip().startswith("#"):
                        continue
                    k, v = raw.split("=", 1)
                    if k.strip() == "LAW_OC" and not os.getenv("LAW_OC"):
                        os.environ["LAW_OC"] = v.strip().strip('"').strip("'")
            except Exception:
                pass
    return os.getenv("LAW_OC", "").strip()


def _notice_label(record: dict) -> str:
    number = clean(record.get("발령번호") or record.get("행정규칙발령번호"))
    date = clean(record.get("발령일자"))
    m = re.search(r"(20\d{2})\s*[-–]\s*(\d+)", number)
    if m:
        return f"{m.group(1)}-{int(m.group(2))}"
    nums = re.findall(r"\d+", number)
    return f"{date[:4]}-{int(nums[-1])}" if date[:4].isdigit() and nums else ""


def query_notice(spec: dict, timeout: int = 45) -> tuple[dict, dict]:
    oc = get_law_oc()
    checked = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not oc:
        return {}, {
            "regime": spec["regime"],
            "title_requested": spec["title"],
            "status": "API_KEY_MISSING",
            "checked_at_utc": checked,
            "message": "LAW_OC not set; approved seed master can still be used.",
        }
    try:
        r = requests.get(
            f"{LAW_BASE}/DRF/lawSearch.do",
            params={
                "OC": oc,
                "target": "admrul",
                "type": "JSON",
                "nw": 1,
                "search": 1,
                "query": spec["title"],
                "display": 100,
            },
            timeout=timeout,
        )
        r.raise_for_status()
        candidates = []
        for rec in recursive_dicts(r.json()):
            title = clean(rec.get("행정규칙명") or rec.get("행정규칙제목"))
            serial = clean(rec.get("행정규칙일련번호") or rec.get("행정규칙 일련번호"))
            if title and serial and title_matches(title, spec):
                candidates.append({
                    "regime": spec["regime"],
                    "official_notice": _notice_label(rec),
                    "official_serial": serial,
                    "official_title": title,
                    "issue_date": clean(rec.get("발령일자")),
                    "effective_date": clean(rec.get("시행일자")),
                    "ministry": clean(rec.get("소관부처명")),
                })
        if not candidates:
            return {}, {
                "regime": spec["regime"], "title_requested": spec["title"],
                "status": "NO_MATCH", "checked_at_utc": checked,
            }
        best = sorted(candidates, key=lambda x: notice_key(x["official_notice"]))[-1]
        return best, {**best, "status": "FOUND", "checked_at_utc": checked}
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}".replace(oc, "***")
        msg = re.sub(r"([?&]OC=)[^&\s]+", r"\1***", msg, flags=re.I)
        return {}, {
            "regime": spec["regime"], "title_requested": spec["title"],
            "status": "API_ERROR", "checked_at_utc": checked, "message": msg,
        }


def fetch_notice_body(info: dict, timeout: int = 90) -> tuple[Any, str]:
    oc = get_law_oc()
    serial = clean(info.get("official_serial"))
    if not oc or not serial:
        return {}, "NOT_AVAILABLE"
    try:
        r = requests.get(
            f"{LAW_BASE}/DRF/lawService.do",
            params={"OC": oc, "target": "admrul", "ID": serial, "type": "JSON"},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json(), "FETCHED"
    except Exception as exc:
        return {}, f"ERROR:{type(exc).__name__}"


def loose_word(word: str) -> str:
    return r"\s*".join(re.escape(ch) for ch in word)


THRESHOLD_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*%\s*(" +
    "|".join(loose_word(w) for w in ["이상", "초과", "이하", "미만"]) + r")"
)
HEADER_NOISE_RE = re.compile(r"고유번호\s*물\s*질\s*명\s*CAS\s*No\.?")
FLAT_LIST_CAS_RE = re.compile(r"\d{2,7}-\d{2}-\d")
FLAT_LIST_BRACKET_RE = re.compile(r"\[(.+?)\]")
FLAT_LIST_NOISE_RE = re.compile(
    r"\[별표\s*\d+\]|법제처\s*-?\s*\d+\s*/?\s*\d*\s*-?\s*국가법령정보센터|-\s*\d+\s*/\s*\d+\s*-"
)
FLAT_LIST_ENTRY_END_RE = re.compile(
    r"및\s*" + loose_word("이를") + r"\s*(\d+(?:\.\d+)?)\s*%\s*(이상|초과|이하|미만)\s*" +
    loose_word("함유한") + r"\s*" + loose_word("혼합물") +
    r"(?:" + loose_word("질은") + r"\s*" + loose_word("제외한다") + r"\.|\.\s*다만[^.]*" + loose_word("제외한다") + r"\.)?"
)


def find_appendix_attachments(payload: Any) -> list[dict]:
    found = []
    for rec in recursive_dicts(payload):
        pdf_link = clean(rec.get("별표서식PDF파일링크"))
        hwp_link = clean(rec.get("별표서식파일링크"))
        if pdf_link or hwp_link:
            found.append({
                "appendix_no": clean(rec.get("별표번호")),
                "appendix_branch": clean(rec.get("별표가지번호")),
                "appendix_kind": clean(rec.get("별표구분")),
                "appendix_title": clean(rec.get("별표제목")),
                "pdf_path": pdf_link,
                "hwp_path": hwp_link,
            })
    # de-duplicate repeated attachment records in nested JSON
    unique = []
    seen = set()
    for a in found:
        key = (a["appendix_no"], a["appendix_title"], a["pdf_path"], a["hwp_path"])
        if key not in seen:
            seen.add(key)
            unique.append(a)
    return unique


def download_attachment(path: str, timeout: int = 60) -> bytes:
    if not path:
        raise ValueError("PDF path is empty")
    url = path if path.startswith("http") else f"{LAW_BASE}{path}"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def merge_wrapped_catchall_lines(lines: list[str]) -> list[str]:
    merged, i = [], 0
    while i < len(lines):
        line = lines[i]
        marker = NUM_DASH_ONLY_RE.match(line)
        if marker and merged and not ITEM_RE.match(merged[-1]) and not DESIG_RE.match(merged[-1]):
            prev = merged.pop()
            nxt = ""
            if i + 1 < len(lines) and not ITEM_RE.match(lines[i+1]) and not DESIG_RE.match(lines[i+1]) and not NUM_DASH_ONLY_RE.match(lines[i+1]):
                nxt = lines[i+1]
                i += 1
            merged.append(f"{marker.group(1)} {prev.strip()} {nxt.strip()} -".strip())
        else:
            merged.append(line)
        i += 1
    return merged


def preprocess_lines(raw_text: str) -> list[str]:
    text = HEADER_NOISE_RE.sub("\n", raw_text)
    text = re.sub(r"(\d-\d)고유번호", r"\1\n", text)
    lines = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or re.fullmatch(r"\*+", line):
            continue
        m = INLINE_CODE_RE.match(line)
        if m:
            lines.extend([m.group(1), m.group(2)])
        else:
            lines.append(line)
    return merge_wrapped_catchall_lines(lines)


def parse_grouped_appendix(raw_text: str) -> pd.DataFrame:
    lines = preprocess_lines(raw_text)
    item_idx = [i for i, line in enumerate(lines) if ITEM_RE.match(line)]
    cols = ["designation_id", "designation_id_inferred", "item_no", "substance_name", "cas",
            "is_catch_all", "group_title", "concentration_threshold_pct", "concentration_operator"]
    if not item_idx:
        return pd.DataFrame(columns=cols)

    boundaries, last = [], 10_000
    for i in item_idx:
        no = int(ITEM_RE.match(lines[i]).group(1))
        if no <= last:
            boundaries.append(i)
        last = no
    boundaries.append(len(lines))
    spans = list(zip(boundaries, boundaries[1:]))
    preamble = lines[:boundaries[0]]

    def last_item(s, e):
        return max(i for i in item_idx if s <= i < e)

    tails = [lines[last_item(s, e)+1:e] for s, e in spans]
    next_titles = [preamble]
    own_catchalls = [[] for _ in spans]

    for k, tail in enumerate(tails):
        items, remainder = [], []
        current_no, current_parts, current_complete = None, [], True
        i = 0
        while i < len(tail):
            line = tail[i]
            if DESIG_RE.match(line):
                i += 1
                continue
            m = CATCHALL_RE.match(line)
            if m:
                if current_no is not None:
                    items.append((current_no, " ".join(current_parts)))
                current_no, current_parts = m.group(1), [m.group(2)]
                txt = " ".join(current_parts).strip()
                current_complete = line.rstrip().endswith("-") or txt.endswith("혼합물") or txt.endswith("제외한다.")
            elif current_no is not None and not current_complete:
                current_parts.append(line)
                txt = " ".join(current_parts).strip()
                current_complete = line.rstrip().endswith("-") or txt.endswith("혼합물") or txt.endswith("제외한다.")
            else:
                remainder = tail[i:]
                break
            i += 1
        if current_no is not None:
            items.append((current_no, " ".join(current_parts)))
        own_catchalls[k] = items
        next_titles.append(remainder)

    rows = []
    for k, (s, e) in enumerate(spans):
        title_lines = next_titles[k]
        code = next((line for line in [*title_lines, *lines[s:e]] if DESIG_RE.match(line)), None)
        inferred = code is None
        code = code or f"UNKNOWN-{k+1}"
        title = " ".join(line for line in title_lines if not DESIG_RE.match(line)).strip()
        tm = THRESHOLD_RE.search(title)
        thr = float(tm.group(1)) if tm else np.nan
        op = re.sub(r"\s+", "", tm.group(2)) if tm else ""
        for i in range(s, e):
            m = ITEM_RE.match(lines[i])
            if m:
                rows.append({
                    "designation_id": code, "designation_id_inferred": inferred,
                    "item_no": int(m.group(1)), "substance_name": m.group(2).strip(),
                    "cas": m.group(3), "is_catch_all": False, "group_title": title,
                    "concentration_threshold_pct": thr, "concentration_operator": op,
                })
        for item_no, name in own_catchalls[k]:
            rows.append({
                "designation_id": code, "designation_id_inferred": inferred,
                "item_no": int(item_no), "substance_name": name.strip(), "cas": "",
                "is_catch_all": True, "group_title": title,
                "concentration_threshold_pct": thr, "concentration_operator": op,
            })
    return pd.DataFrame(rows, columns=cols)


def parse_flat_appendix(raw_text: str) -> pd.DataFrame:
    cols = ["designation_id", "designation_id_inferred", "item_no", "substance_name", "cas",
            "is_catch_all", "group_title", "concentration_threshold_pct", "concentration_operator"]
    blob = FLAT_LIST_NOISE_RE.sub(" ", raw_text)
    blob = re.sub(r"\s+", " ", blob).strip()
    matches = list(FLAT_LIST_ENTRY_END_RE.finditer(blob))
    if not matches:
        return pd.DataFrame(columns=cols)
    rows, last = [], 0
    for item_no, m in enumerate(matches, start=1):
        segment = blob[last:m.end()].strip()
        last = m.end()
        bracket = FLAT_LIST_BRACKET_RE.search(segment)
        korean = segment[:bracket.start()].strip() if bracket else segment
        inside = bracket.group(1) if bracket else ""
        parts = [p.strip() for p in inside.split(";")]
        cas_candidates = [p.strip() for p in (parts[-1] if parts else "").split(",")]
        cas_list = [c for c in cas_candidates if FLAT_LIST_CAS_RE.fullmatch(c)]
        threshold, operator = float(m.group(1)), m.group(2)
        if not cas_list:
            rows.append({
                "designation_id": f"ITEM-{item_no:03d}", "designation_id_inferred": False,
                "item_no": item_no, "substance_name": korean, "cas": "", "is_catch_all": True,
                "group_title": segment, "concentration_threshold_pct": threshold,
                "concentration_operator": operator,
            })
        for cas in cas_list:
            rows.append({
                "designation_id": f"ITEM-{item_no:03d}", "designation_id_inferred": False,
                "item_no": item_no, "substance_name": korean, "cas": cas, "is_catch_all": False,
                "group_title": segment, "concentration_threshold_pct": threshold,
                "concentration_operator": operator,
            })
    return pd.DataFrame(rows, columns=cols)




# ---------------------------------------------------------------------------
# Human-hazard appendix: wide table parser
# ---------------------------------------------------------------------------
# The current 인체등유해성물질 appendix is a wide regulatory table rather than
# the grouped designation/CAS layout used by several other notices.  Parse it
# with pdfplumber's table extractor first, while retaining the legacy parsers
# as fallbacks.  Parsed rows remain CANDIDATE_REVIEW_REQUIRED and never replace
# the approved master automatically.

WIDE_DESIG_RE = re.compile(
    r"(?<!\d)(\d{2,4}-\d+-\d+(?:\s*(?:의\s*소번호|소번호)\s*\d+)?)(?!\d)"
)
WIDE_HEADER_HINTS = ("인체등유해성물질", "인체급성유해성물질", "인체만성유해성물질", "생태유해성물질")

def _cell_text(value: Any) -> str:
    return re.sub(r"\s+", " ", clean(value).replace("\n", " ")).strip()

def _header_token(value: Any) -> str:
    s = _cell_text(value).lower()
    s = s.replace("cas no.", "casno").replace("cas no", "casno").replace("cas번호", "casno")
    return re.sub(r"[\s·ㆍ._:/()\[\]{}-]+", "", s)

def _find_wide_header(rows: list[list[Any]]) -> tuple[int, dict[str, int]] | tuple[None, dict]:
    """Locate designation/name/CAS columns in a possibly multi-row header.

    PDF table headers are often split across 2--3 physical rows.  We therefore
    combine column-wise text across a sliding header window and identify the
    columns semantically rather than relying on fixed column numbers.
    """
    if not rows:
        return None, {}
    max_cols = max((len(r) for r in rows if isinstance(r, list)), default=0)
    if max_cols < 3:
        return None, {}
    for start in range(min(8, len(rows))):
        for height in (1, 2, 3):
            end = min(len(rows), start + height)
            col_text = []
            for j in range(max_cols):
                parts = []
                for i in range(start, end):
                    row = rows[i] if isinstance(rows[i], list) else []
                    if j < len(row):
                        parts.append(_cell_text(row[j]))
                col_text.append(_header_token(" ".join(parts)))
            desig_idx = next((j for j, t in enumerate(col_text) if "고유번호" in t), None)
            cas_idx = next((j for j, t in enumerate(col_text) if "casno" in t or t == "cas"), None)
            name_idx = next((j for j, t in enumerate(col_text) if ("인체등유해성물질" in t and "명칭" in t) or "물질명" in t or "화학물질명칭" in t), None)
            if desig_idx is not None and cas_idx is not None and name_idx is not None:
                return end - 1, {"designation": desig_idx, "name": name_idx, "cas": cas_idx}
    return None, {}

def _looks_like_human_hazard_table(rows: list[list[Any]]) -> bool:
    sample = " ".join(_cell_text(c) for r in rows[:8] if isinstance(r, list) for c in r)
    compact = re.sub(r"\s+", "", sample)
    return ("고유번호" in compact and "CAS" in compact and any(h.replace(" ", "") in compact for h in WIDE_HEADER_HINTS))

def parse_human_hazard_wide_tables(pdf: Any) -> pd.DataFrame:
    """Parse the current 인체등유해성물질 wide appendix table.

    The function intentionally extracts only the fields needed for source
    traceability (designation, substance name, CAS).  Hazard-category columns
    and pictograms may change independently and are therefore preserved only
    in ``group_title`` as normalized row text rather than interpreted here.
    """
    cols = ["designation_id", "designation_id_inferred", "item_no", "substance_name", "cas",
            "is_catch_all", "group_title", "concentration_threshold_pct", "concentration_operator"]
    out_rows: list[dict] = []
    global_item = 0
    last_designation = ""

    table_settings_candidates = [
        None,
        {"vertical_strategy": "lines", "horizontal_strategy": "lines", "intersection_tolerance": 5, "snap_tolerance": 3},
        {"vertical_strategy": "text", "horizontal_strategy": "text", "min_words_vertical": 1, "min_words_horizontal": 1, "text_tolerance": 3},
    ]

    for page_no, page in enumerate(pdf.pages, start=1):
        page_tables: list[list[list[Any]]] = []
        for settings in table_settings_candidates:
            try:
                tables = page.extract_tables(table_settings=settings) if settings is not None else page.extract_tables()
            except Exception:
                tables = []
            for table in tables or []:
                if table and table not in page_tables:
                    page_tables.append(table)

        for table in page_tables:
            if not _looks_like_human_hazard_table(table):
                continue
            header_end, idx = _find_wide_header(table)
            if header_end is None:
                continue
            for raw_row in table[header_end + 1:]:
                if not isinstance(raw_row, list):
                    continue
                row = [_cell_text(c) for c in raw_row]
                row_text = " | ".join(c for c in row if c)
                if not row_text:
                    continue
                if "고유번호" in row_text and "CAS" in row_text:
                    continue
                if re.fullmatch(r"(?:삭제|<삭제>|-)+", row_text.replace(" ", ""), re.I):
                    continue

                def at(key: str) -> str:
                    j = idx[key]
                    return row[j] if j < len(row) else ""

                desig_cell = at("designation")
                name_cell = at("name")
                cas_cell = at("cas")
                desig_match = WIDE_DESIG_RE.search(desig_cell) or WIDE_DESIG_RE.search(row_text)
                designation = desig_match.group(1) if desig_match else ""
                designation = re.sub(r"\s*의\s*소번호\s*", "의 소번호 ", designation).strip()
                if designation:
                    last_designation = designation
                elif last_designation and (name_cell or CAS_RE.search(cas_cell)):
                    # A vertically merged designation cell is blank on continuation rows.
                    designation = last_designation

                cas_list = list(dict.fromkeys(CAS_RE.findall(cas_cell)))
                if not cas_list:
                    # Some PDFs visually place the CAS in an adjacent cell after extraction.
                    cas_list = list(dict.fromkeys(CAS_RE.findall(row_text)))

                name = name_cell.strip()
                if not designation and not name and not cas_list:
                    continue
                if "삭제" in name.replace(" ", "") and not cas_list:
                    continue

                global_item += 1
                if cas_list:
                    for cas in cas_list:
                        out_rows.append({
                            "designation_id": designation or f"UNKNOWN-P{page_no}-{global_item}",
                            "designation_id_inferred": not bool(designation),
                            "item_no": global_item,
                            "substance_name": name,
                            "cas": cas,
                            "is_catch_all": False,
                            "group_title": row_text,
                            "concentration_threshold_pct": np.nan,
                            "concentration_operator": "",
                        })
                else:
                    # Keep a named/no-CAS regulatory row for audit rather than silently dropping it.
                    out_rows.append({
                        "designation_id": designation or f"UNKNOWN-P{page_no}-{global_item}",
                        "designation_id_inferred": not bool(designation),
                        "item_no": global_item,
                        "substance_name": name,
                        "cas": "",
                        "is_catch_all": True,
                        "group_title": row_text,
                        "concentration_threshold_pct": np.nan,
                        "concentration_operator": "",
                    })

    if not out_rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(out_rows, columns=cols)
    # De-duplicate repeated tables caused by trying multiple extraction strategies.
    key_cols = ["designation_id", "substance_name", "cas"]
    out = out.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)
    return out

def _write_pdf_debug_artifacts(pdf: Any, regime: str) -> None:
    """Write parser diagnostics only when recognition fails."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", regime or "UNKNOWN")
    try:
        text = "\n\n".join((p.extract_text() or "") for p in pdf.pages)
        (INTERMEDIATE / f"01_pdf_debug_{safe}_text.txt").write_text(text, encoding="utf-8")
        samples = []
        for pno, page in enumerate(pdf.pages[:5], start=1):
            try:
                tables = page.extract_tables() or []
            except Exception as exc:
                samples.append({"page": pno, "error": f"{type(exc).__name__}: {exc}"})
                continue
            samples.append({"page": pno, "tables": tables[:3]})
        (INTERMEDIATE / f"01_pdf_debug_{safe}_tables.json").write_text(
            json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass



HUMAN_HAZARD_PDF_STUB_RE = re.compile(
    r"(?:자세한\s*내용은.*?상단\s*메뉴|상단\s*메뉴.*?버튼.*?이용)",
    re.I | re.S,
)
HWP_DASH_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d]")
HWP_DESIG_RE = re.compile(
    r"(?<!\d)(\d{1,4})\s*-\s*(\d{1,3})\s*-\s*(\d{1,6})(?!\d)"
)

def _normalize_hwp_dashes(value: str) -> str:
    """Normalize full-width and Unicode dash/minus variants used in HWP/HWPX."""
    return HWP_DASH_RE.sub("-", value)

def _human_hazard_pdf_is_stub(text: str) -> bool:
    """Return True when the official PDF is only a viewer/help stub.

    The 2026-5 human-hazard appendix currently exposes a one-page PDF whose
    textual content tells the user to use the top-menu button for details.
    Such a file contains no regulatory rows, so treating it as an
    unrecognized table is misleading.  It is explicitly classified as a
    source stub and the audit falls back to the HWP/HWPX attachment.
    """
    compact = normalize_text(text)
    return bool(HUMAN_HAZARD_PDF_STUB_RE.search(compact))

def _strip_hwp_controls(value: str) -> str:
    # Keep ordinary whitespace; remove embedded HWP control codepoints.
    value = value.replace("\x00", " ")
    value = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", " ", value)
    return re.sub(r"[ \t]+", " ", value)

def _hwpx_section_xml_names(zf: zipfile.ZipFile) -> list[str]:
    names = sorted(
        n for n in zf.namelist()
        if re.search(r"(?:^|/)section\d+\.xml$", n, re.I)
    )
    if not names:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".xml"))
    return names

def _extract_hwpx_text(data: bytes) -> str:
    """Extract text from HWPX (ZIP/XML) using only the Python stdlib.

    Paragraph/cell boundaries are preserved with newlines because regulatory
    designation identifiers and CAS values can occupy separate table cells.
    """
    import xml.etree.ElementTree as ET
    parts: list[str] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in _hwpx_section_xml_names(zf):
            try:
                root = ET.fromstring(zf.read(name))
            except Exception:
                continue
            for elem in root.iter():
                local = str(elem.tag).split("}")[-1].lower()
                if local in {"t", "text"} and elem.text:
                    parts.append(elem.text)
                elif local in {"p", "tc", "tr"}:
                    parts.append("\n")
            parts.append("\n")
    return _normalize_hwp_dashes(_strip_hwp_controls(" ".join(parts)))

def _extract_hwpx_table_rows(data: bytes) -> list[list[str]]:
    """Return HWPX table rows as cell-text lists.

    HWPX uses namespace-qualified table-row/table-cell elements.  We match
    local tag names only so the parser remains tolerant of namespace/version
    changes.  Merged cells may produce blanks on continuation rows; callers
    carry the last seen designation id forward when appropriate.
    """
    import xml.etree.ElementTree as ET
    rows: list[list[str]] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in _hwpx_section_xml_names(zf):
            try:
                root = ET.fromstring(zf.read(name))
            except Exception:
                continue
            for tr in root.iter():
                local = str(tr.tag).split("}")[-1].lower()
                if local not in {"tr", "tablerow"}:
                    continue
                cells: list[str] = []
                for child in list(tr):
                    clocal = str(child.tag).split("}")[-1].lower()
                    if clocal not in {"tc", "tablecell"}:
                        continue
                    bits: list[str] = []
                    for elem in child.iter():
                        elocal = str(elem.tag).split("}")[-1].lower()
                        if elocal in {"t", "text"} and elem.text:
                            bits.append(elem.text)
                    cell_text = _normalize_hwp_dashes(
                        normalize_text(_strip_hwp_controls(" ".join(bits)))
                    )
                    cells.append(cell_text)
                if cells:
                    rows.append(cells)
    return rows

def _parse_human_hazard_hwpx_rows(data: bytes) -> pd.DataFrame:
    """Parse designation/CAS audit pairs directly from HWPX table rows."""
    cols = [
        "designation_id", "designation_id_inferred", "item_no", "substance_name",
        "cas", "is_catch_all", "group_title", "concentration_threshold_pct",
        "concentration_operator",
    ]
    rows_xml = _extract_hwpx_table_rows(data)
    out: list[dict] = []
    current_designation = ""
    item_no = 0

    for cells in rows_xml:
        combined = _normalize_hwp_dashes(" | ".join(cells))
        dmatch = HWP_DESIG_RE.search(combined)
        if dmatch:
            current_designation = f"{dmatch.group(1)}-{dmatch.group(2)}-{dmatch.group(3)}"

        cas_hits = list(dict.fromkeys(CAS_RE.findall(combined)))
        if not current_designation or not cas_hits:
            continue

        # Prefer a cell that contains chemical text but is not only the id/CAS/header.
        name = ""
        for cell in cells:
            c = normalize_text(cell)
            if not c:
                continue
            if HWP_DESIG_RE.fullmatch(_normalize_hwp_dashes(c)):
                continue
            if CAS_RE.fullmatch(c):
                continue
            if re.search(r"고유번호|CAS\s*No|유해성\s*구분", c, re.I):
                continue
            if len(c) > len(name):
                name = c

        item_no += 1
        for cas in cas_hits:
            out.append({
                "designation_id": current_designation,
                "designation_id_inferred": False,
                "item_no": item_no,
                "substance_name": name[:1000],
                "cas": cas,
                "is_catch_all": False,
                "group_title": combined[:3000],
                "concentration_threshold_pct": np.nan,
                "concentration_operator": "",
            })

    if not out:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(out, columns=cols).drop_duplicates(
        subset=["designation_id", "cas"], keep="first"
    ).reset_index(drop=True)

def _extract_hwp5_text(data: bytes) -> str:
    """Extract paragraph text from legacy binary HWP 5.x.

    Requires the lightweight ``olefile`` package.  No Hancom Office
    installation or COM automation is required.
    """
    try:
        import olefile  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "HWP_BINARY_REQUIRES_OLEFILE: run `pip install olefile` once in the regai environment"
        ) from exc

    parts: list[str] = []
    with olefile.OleFileIO(io.BytesIO(data)) as ole:
        if not ole.exists("FileHeader"):
            raise ValueError("Not a recognized HWP 5.x OLE file")
        header = ole.openstream("FileHeader").read()
        flags = int.from_bytes(header[36:40], "little", signed=False) if len(header) >= 40 else 0
        compressed = bool(flags & 0x01)

        section_paths = []
        for path in ole.listdir(streams=True, storages=False):
            if len(path) >= 2 and path[0] == "BodyText" and re.fullmatch(r"Section\d+", path[1]):
                section_paths.append(path)
        section_paths.sort(key=lambda p: int(re.search(r"\d+", p[1]).group()))

        for path in section_paths:
            blob = ole.openstream(path).read()
            if compressed:
                try:
                    blob = zlib.decompress(blob, -15)
                except zlib.error:
                    # A malformed/alternative stream should not silently make
                    # source verification succeed; try raw bytes only as a
                    # diagnostic fallback.
                    pass

            pos = 0
            n = len(blob)
            while pos + 4 <= n:
                rec = int.from_bytes(blob[pos:pos+4], "little", signed=False)
                pos += 4
                tag_id = rec & 0x3FF
                size = (rec >> 20) & 0xFFF
                if size == 0xFFF:
                    if pos + 4 > n:
                        break
                    size = int.from_bytes(blob[pos:pos+4], "little", signed=False)
                    pos += 4
                if size < 0 or pos + size > n:
                    break
                payload = blob[pos:pos+size]
                pos += size
                # HWP 5 record tag 67 = HWPTAG_PARA_TEXT.
                if tag_id == 67 and payload:
                    try:
                        s = payload.decode("utf-16le", errors="ignore")
                    except Exception:
                        continue
                    s = _strip_hwp_controls(s)
                    if s.strip():
                        parts.append(s.strip())
                        parts.append("\n")
    return _normalize_hwp_dashes("\n".join(parts))

def extract_hwp_or_hwpx_text(data: bytes) -> tuple[str, str]:
    """Return (text, source_format) for HWPX or legacy HWP."""
    if data[:2] == b"PK":
        return _extract_hwpx_text(data), "HWPX_XML"
    if data[:8] == bytes.fromhex("D0CF11E0A1B11AE1"):
        return _extract_hwp5_text(data), "HWP5_OLE"
    # Some servers may return XML/HTML despite an .hwp-looking URL.
    head = data[:500].lstrip().lower()
    if head.startswith(b"<?xml") or head.startswith(b"<"):
        decoded = data.decode("utf-8", errors="ignore")
        plain = re.sub(r"<[^>]+>", " ", decoded)
        return normalize_text(plain), "TEXT_XML_HTML"
    raise ValueError("Unsupported HWP/HWPX attachment format")

def parse_human_hazard_source_text(raw_text: str) -> pd.DataFrame:
    """Build source-audit rows from the official HWP/HWPX text.

    This is deliberately an audit parser, not a rule generator.  It associates
    every CAS number found between one designation identifier and the next with
    that designation.  The resulting (designation, CAS) pairs are then compared
    with the frozen approved master.  No parsed row is auto-promoted.
    """
    cols = [
        "designation_id", "designation_id_inferred", "item_no", "substance_name",
        "cas", "is_catch_all", "group_title", "concentration_threshold_pct",
        "concentration_operator",
    ]
    text = _normalize_hwp_dashes(raw_text.replace("\u00a0", " "))
    matches = list(HWP_DESIG_RE.finditer(text))
    if not matches:
        return pd.DataFrame(columns=cols)

    rows: list[dict] = []
    item_no = 0
    for i, m in enumerate(matches):
        designation = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[m.end():end]
        # Limit pathological spillover while keeping long multi-line entries.
        block = block[:10000]
        cas_hits = list(dict.fromkeys(CAS_RE.findall(block)))

        first_cas = CAS_RE.search(block)
        name_chunk = block[:first_cas.start()] if first_cas else block
        name_chunk = re.sub(
            r"(?:인체급성유해성물질|인체만성유해성물질|생태유해성물질|"
            r"유해성구분|고유번호|CAS\s*No\.?)",
            " ",
            name_chunk,
            flags=re.I,
        )
        name = normalize_text(name_chunk).strip(" |-:;")
        if len(name) > 1000:
            name = name[:1000]

        item_no += 1
        if cas_hits:
            for cas in cas_hits:
                rows.append({
                    "designation_id": designation,
                    "designation_id_inferred": False,
                    "item_no": item_no,
                    "substance_name": name,
                    "cas": cas,
                    "is_catch_all": False,
                    "group_title": normalize_text(block)[:3000],
                    "concentration_threshold_pct": np.nan,
                    "concentration_operator": "",
                })
        else:
            # Preserve no-CAS designation blocks for traceability, but pair
            # cross-check metrics use only nonblank CAS values.
            rows.append({
                "designation_id": designation,
                "designation_id_inferred": False,
                "item_no": item_no,
                "substance_name": name,
                "cas": "",
                "is_catch_all": True,
                "group_title": normalize_text(block)[:3000],
                "concentration_threshold_pct": np.nan,
                "concentration_operator": "",
            })

    out = pd.DataFrame(rows, columns=cols)
    return out.drop_duplicates(
        subset=["designation_id", "substance_name", "cas"], keep="first"
    ).reset_index(drop=True)

def parse_human_hazard_hwp_bytes(data: bytes) -> tuple[pd.DataFrame, str, str]:
    """Parse the official HWP/HWPX attachment for source verification.

    For HWPX, try structural row/cell parsing first and then fall back to the
    flattened source-text parser.  This avoids losing identifiers when merged
    table cells or HWPX namespaces break ordinary text extraction.
    """
    text, source_format = extract_hwp_or_hwpx_text(data)
    if source_format == "HWPX_XML":
        structured = _parse_human_hazard_hwpx_rows(data)
        if not structured.empty:
            return structured, "HWPX_TABLE_DESIGNATION_CAS_AUDIT", text
    parsed = parse_human_hazard_source_text(text)
    return parsed, f"{source_format}_DESIGNATION_CAS_AUDIT", text

def _write_hwp_debug_artifact(raw_text: str, regime: str) -> None:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", regime or "UNKNOWN")
    try:
        (INTERMEDIATE / f"01_hwp_debug_{safe}_text.txt").write_text(
            raw_text, encoding="utf-8"
        )
    except Exception:
        pass

def _write_hwpx_structure_debug(data: bytes, regime: str) -> None:
    """Persist a compact HWPX XML/table diagnostic only when parsing yields no rows."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", regime or "UNKNOWN")
    try:
        table_rows = _extract_hwpx_table_rows(data)
        payload = {
            "table_row_count": len(table_rows),
            "sample_rows": table_rows[:30],
        }
        (INTERMEDIATE / f"01_hwpx_debug_{safe}_structure.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        try:
            (INTERMEDIATE / f"01_hwpx_debug_{safe}_structure.json").write_text(
                json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass


def parse_appendix_pdf_bytes(data: bytes, regime: str = "") -> tuple[pd.DataFrame, str]:
    import pdfplumber
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        # The human-hazard "PDF" can be a one-page viewer/help stub rather
        # than the actual appendix table.  Detect that state explicitly.
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        if regime == "HUMAN_HAZARD" and _human_hazard_pdf_is_stub(text):
            _write_pdf_debug_artifacts(pdf, regime)
            return pd.DataFrame(), "PDF_STUB_NO_REGULATORY_ROWS"

        if regime == "HUMAN_HAZARD":
            wide = parse_human_hazard_wide_tables(pdf)
            if not wide.empty:
                return wide, "HUMAN_HAZARD_WIDE_TABLE"

        grouped = parse_grouped_appendix(text)
        if not grouped.empty:
            return grouped, "GROUPED_TABLE"
        flat = parse_flat_appendix(text)
        if not flat.empty:
            return flat, "FLAT_LIST"
        _write_pdf_debug_artifacts(pdf, regime)
    return pd.DataFrame(), "TABLE_FORMAT_NOT_RECOGNIZED"


def audit_notice_pdf(info: dict, payload: Any, timeout: int = 90) -> tuple[pd.DataFrame, list[dict]]:
    """Audit official appendix sources.

    PDF remains the first-choice source.  For HUMAN_HAZARD only, an official
    PDF that is a viewer/help stub (or otherwise contains no parseable rows)
    falls back to the HWP/HWPX attachment carried in the same API record.
    Parsed source rows remain candidates and never overwrite the approved
    master.
    """
    attachments = find_appendix_attachments(payload)
    frames: list[pd.DataFrame] = []
    statuses: list[dict] = []
    regime = clean(info.get("regime"))

    def _decorate(parsed: pd.DataFrame, label: str, parser: str, source_type: str) -> pd.DataFrame:
        parsed = parsed.copy()
        parsed["regime"] = regime
        parsed["official_notice"] = clean(info.get("official_notice"))
        parsed["official_title"] = clean(info.get("official_title"))
        parsed["official_serial"] = clean(info.get("official_serial"))
        parsed["appendix_title"] = label
        parsed["parser_type"] = parser
        parsed["source_type"] = source_type
        parsed["approval_status"] = "CANDIDATE_REVIEW_REQUIRED"
        return parsed

    for idx, a in enumerate(attachments, start=1):
        label = clean(a.get("appendix_title")) or clean(a.get("appendix_no")) or f"appendix_{idx}"
        if re.fullmatch(r"<?\s*삭\s*제\s*>?", label):
            statuses.append({"appendix": label, "status": "ABOLISHED_APPENDIX", "rows": 0})
            continue

        pdf_status = "PDF_NOT_ATTEMPTED"
        pdf_sha256 = ""
        hwp_sha256 = ""

        # 1) Preferred PDF path.
        if clean(a.get("pdf_path")):
            try:
                pdf_data = download_attachment(a["pdf_path"], timeout=timeout)
                pdf_sha256 = hashlib.sha256(pdf_data).hexdigest()
                parsed, parser = parse_appendix_pdf_bytes(pdf_data, regime)
                pdf_status = parser
                if not parsed.empty:
                    parsed = _decorate(parsed, label, parser, "PDF")
                    frames.append(parsed)
                    statuses.append({
                        "appendix": label,
                        "status": "PARSED",
                        "source_type": "PDF",
                        "parser_type": parser,
                        "rows": int(len(parsed)),
                        "inferred_designation_rows": int(parsed["designation_id_inferred"].sum()),
                        "pdf_status": parser,
                        "pdf_sha256": pdf_sha256,
                    })
                    continue
            except Exception as exc:
                pdf_status = f"PDF_ERROR:{type(exc).__name__}:{exc}"
        else:
            pdf_status = "PDF_LINK_MISSING"

        # 2) HUMAN_HAZARD fallback to the actual HWP/HWPX source when the PDF
        #    is a stub or otherwise not data-bearing.
        if regime == "HUMAN_HAZARD" and clean(a.get("hwp_path")):
            try:
                hwp_data = download_attachment(a["hwp_path"], timeout=timeout)
                hwp_sha256 = hashlib.sha256(hwp_data).hexdigest()

                # Keep a byte-for-byte snapshot of the official appendix source.
                ext = ".hwpx" if hwp_data[:2] == b"PK" else ".hwp"
                safe_label = re.sub(r"[^A-Za-z0-9가-힣_-]+", "_", label)[:80] or f"appendix_{idx}"
                snapshot = INTERMEDIATE / f"01_official_HUMAN_HAZARD_{safe_label}{ext}"
                snapshot.write_bytes(hwp_data)

                parsed, parser, hwp_text = parse_human_hazard_hwp_bytes(hwp_data)
                _write_hwp_debug_artifact(hwp_text, regime)
                if not parsed.empty:
                    parsed = _decorate(parsed, label, parser, "HWP/HWPX")
                    frames.append(parsed)
                    statuses.append({
                        "appendix": label,
                        "status": "PARSED",
                        "source_type": "HWP/HWPX",
                        "parser_type": parser,
                        "rows": int(len(parsed)),
                        "inferred_designation_rows": int(parsed["designation_id_inferred"].sum()),
                        "pdf_status": pdf_status,
                        "pdf_sha256": pdf_sha256,
                        "hwp_sha256": hwp_sha256,
                        "official_source_snapshot": snapshot.name,
                    })
                    continue

                if hwp_data[:2] == b"PK":
                    _write_hwpx_structure_debug(hwp_data, regime)
                statuses.append({
                    "appendix": label,
                    "status": "HWP_FALLBACK_NO_DESIGNATION_CAS_ROWS",
                    "source_type": "HWP/HWPX",
                    "rows": 0,
                    "pdf_status": pdf_status,
                    "pdf_sha256": pdf_sha256,
                    "hwp_sha256": hwp_sha256,
                    "parser_type": parser,
                    "official_source_snapshot": snapshot.name,
                })
                continue
            except Exception as exc:
                statuses.append({
                    "appendix": label,
                    "status": "HWP_FALLBACK_ERROR",
                    "rows": 0,
                    "pdf_status": pdf_status,
                    "pdf_sha256": pdf_sha256,
                    "hwp_sha256": hwp_sha256,
                    "message": f"{type(exc).__name__}: {exc}",
                })
                continue

        statuses.append({
            "appendix": label,
            "status": pdf_status,
            "rows": 0,
            "pdf_status": pdf_status,
            "pdf_sha256": pdf_sha256,
            "hwp_available": bool(clean(a.get("hwp_path"))),
        })

    out = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return out, statuses


# Versioned operational reference for the paper.
# Positive = the regulatory scope cannot be represented safely by direct-CAS equality alone.
REFERENCE_RULE_VERSION = "cas-only-scope-refined-v3-20260910"
BENCHMARK_POPULATION_VERSION = "active-current-scope-v1-20260910"

EXCLUSION_RE = re.compile(
    r"(?:다만|단[\s,，:]|제외(?:한다|함|되는|한)?|excluding|except(?:ion)?|with\s+the\s+exception)",
    re.I,
)
BROAD_SALT_REL_RE = re.compile(
    r"(?:와|과|및)\s*(?:그\s*)?염(?:류)?\b"
    r"|\band\s+(?:all\s+)?(?:its|their)\s+salts?\b",
    re.I,
)
SALT_FAMILY_RE = re.compile(
    r"(?:염류|염\s*(?:및|또는|등)|salts?\b)",
    re.I,
)
COMPOUND_REL_RE = re.compile(
    r"(?:와|과|및)\s*(?:그\s*)?화합물(?:군|류)?\b"
    r"|\band\s+(?:all\s+)?(?:its|their)\s+compounds?\b",
    re.I,
)
GENERIC_COMPOUND_RE = re.compile(
    r"(?:화합물군|화합물류|화합물|\bcompounds?\b)",
    re.I,
)
DERIVATIVE_REL_RE = re.compile(
    r"(?:와|과|및)\s*(?:그\s*)?유도체(?:류)?\b"
    r"|\band\s+(?:all\s+)?(?:its|their)\s+derivatives?\b",
    re.I,
)
GENERIC_DERIVATIVE_RE = re.compile(r"(?:유도체류|\bderivatives\b)", re.I)
REACTION_PRODUCT_RE = re.compile(r"reaction\s+(?:product|products)|반응\s*생성물", re.I)
MIXTURE_UVCB_RE = re.compile(r"reaction\s+mixture|\bmixtures?\b|혼합물|\buvcb\b|\bpolymers?\b|고분자|중합체", re.I)
STRUCTURAL_RANGE_RE = re.compile(
    r"C\s*=?\s*\d+\s*(?:~|[-–—]|to|∼)\s*C?\s*=?\s*\d+"
    r"|탄소수\s*\d+\s*(?:~|[-–—]|에서|∼)\s*\d+",
    re.I,
)
GROUP_LIST_RE = re.compile(r"/\s*구체적\s*목록|총칭으로\s*지정", re.I)


def _group_scope_text(source_text: str) -> str:
    """Return the generic designation text before a specific listed CAS item, if present."""
    src = clean(source_text)
    m = re.search(r"/\s*구체적\s*목록", src, flags=re.I)
    if m:
        return src[:m.start()].strip()
    return src


def _matched_phrase(pattern: re.Pattern, text: str) -> str:
    m = pattern.search(text)
    return normalize_text(m.group(0)) if m else ""


def classify_scope(
    source_text: str, substance_name: str, direct_cas: str
) -> tuple[str, bool, list[str], str]:
    """
    Refined source-derived operational reference.

    Core rule:
    - No direct CAS -> CAS-only screening is insufficient.
    - A specific salt / polymer / derivative / reaction product / structural-range
      substance that HAS its own direct CAS is NOT automatically a CAS gap.
    - With a direct CAS, positive classification requires explicit evidence that the
      legal scope extends beyond that exact CAS (e.g., "A and its salts", a generic
      designation with a specific-item list, or an explicit exclusion clause).
    """
    name = clean(substance_name)
    source = clean(source_text)
    text = f"{name} {source}".strip()
    has_direct = bool(split_cas(direct_cas))
    group_text = _group_scope_text(source)
    has_group_list = bool(GROUP_LIST_RE.search(source))

    # 1) Explicit exclusions are intrinsically not representable by CAS equality alone:
    # CAS-only can over-screen an explicitly excluded form/condition.
    ev = _matched_phrase(EXCLUSION_RE, text)
    if ev:
        return "EXPLICIT_EXCEPTION", True, ["EXPLICIT_EXCEPTION_CLAUSE"], ev

    # 2) No direct CAS is always a positive CAS-only gap; semantic subtype is secondary.
    if not has_direct:
        for scope, pattern, reason in [
            ("REACTION_PRODUCT", REACTION_PRODUCT_RE, "NO_DIRECT_CAS_REACTION_PRODUCT"),
            ("MIXTURE_OR_UVCB", MIXTURE_UVCB_RE, "NO_DIRECT_CAS_MIXTURE_OR_UVCB"),
            ("BROAD_SALT", SALT_FAMILY_RE, "NO_DIRECT_CAS_SALT_FAMILY"),
            ("DERIVATIVE_FAMILY", GENERIC_DERIVATIVE_RE, "NO_DIRECT_CAS_DERIVATIVE_FAMILY"),
            ("COMPOUND_GROUP", GENERIC_COMPOUND_RE, "NO_DIRECT_CAS_COMPOUND_GROUP"),
            ("STRUCTURAL_RANGE", STRUCTURAL_RANGE_RE, "NO_DIRECT_CAS_STRUCTURAL_RANGE"),
        ]:
            ev = _matched_phrase(pattern, text)
            if ev:
                return scope, True, ["NO_DIRECT_CAS", reason], ev
        return "NO_DIRECT_CAS_OTHER", True, ["NO_DIRECT_CAS"], ""

    # 3) With a direct CAS, require an explicit relational/general scope.
    # "A and its salts" is positive, while "magnesium salt (1:1), CAS xxxx" is not.
    ev = _matched_phrase(BROAD_SALT_REL_RE, group_text)
    if ev:
        return "BROAD_SALT", True, ["DIRECT_CAS_PLUS_EXPLICIT_BROAD_SALT_SCOPE"], ev

    # Generic designation + specific listed row: direct CAS of one listed member does
    # not necessarily exhaust the designation's total scope.
    if has_group_list:
        if SALT_FAMILY_RE.search(group_text):
            ev = _matched_phrase(SALT_FAMILY_RE, group_text)
            return "BROAD_SALT", True, ["GENERIC_SALT_DESIGNATION_WITH_SPECIFIC_CAS_LIST"], ev
        if REACTION_PRODUCT_RE.search(group_text):
            ev = _matched_phrase(REACTION_PRODUCT_RE, group_text)
            return "REACTION_PRODUCT", True, ["GENERIC_REACTION_PRODUCT_DESIGNATION_WITH_SPECIFIC_CAS_LIST"], ev
        if MIXTURE_UVCB_RE.search(group_text):
            ev = _matched_phrase(MIXTURE_UVCB_RE, group_text)
            return "MIXTURE_OR_UVCB", True, ["GENERIC_MIXTURE_UVCB_DESIGNATION_WITH_SPECIFIC_CAS_LIST"], ev
        if DERIVATIVE_REL_RE.search(group_text) or GENERIC_DERIVATIVE_RE.search(group_text):
            ev = _matched_phrase(DERIVATIVE_REL_RE, group_text) or _matched_phrase(GENERIC_DERIVATIVE_RE, group_text)
            return "DERIVATIVE_FAMILY", True, ["GENERIC_DERIVATIVE_DESIGNATION_WITH_SPECIFIC_CAS_LIST"], ev
        if COMPOUND_REL_RE.search(group_text) or GENERIC_COMPOUND_RE.search(group_text):
            ev = _matched_phrase(COMPOUND_REL_RE, group_text) or _matched_phrase(GENERIC_COMPOUND_RE, group_text)
            return "COMPOUND_GROUP", True, ["GENERIC_COMPOUND_DESIGNATION_WITH_SPECIFIC_CAS_LIST"], ev
        if STRUCTURAL_RANGE_RE.search(group_text):
            ev = _matched_phrase(STRUCTURAL_RANGE_RE, group_text)
            return "STRUCTURAL_RANGE", True, ["GENERIC_STRUCTURAL_RANGE_WITH_SPECIFIC_CAS_LIST"], ev

    # Explicit relational compound/derivative scope can also occur without a "/ specific list".
    ev = _matched_phrase(COMPOUND_REL_RE, text)
    if ev:
        return "COMPOUND_GROUP", True, ["DIRECT_CAS_PLUS_EXPLICIT_COMPOUND_FAMILY_SCOPE"], ev
    ev = _matched_phrase(DERIVATIVE_REL_RE, text)
    if ev:
        return "DERIVATIVE_FAMILY", True, ["DIRECT_CAS_PLUS_EXPLICIT_DERIVATIVE_FAMILY_SCOPE"], ev

    # 4) IMPORTANT NEGATIVE RULE:
    # A complex chemical name can contain salt/compound/derivative/polymer/reaction
    # product/range terminology yet still denote one CAS-assigned substance.
    complexity_markers = []
    for label, pattern in [
        ("SALT_WORD_IN_SPECIFIC_NAME", SALT_FAMILY_RE),
        ("REACTION_PRODUCT_WORD_IN_SPECIFIC_NAME", REACTION_PRODUCT_RE),
        ("MIXTURE_UVCB_WORD_IN_SPECIFIC_NAME", MIXTURE_UVCB_RE),
        ("DERIVATIVE_WORD_IN_SPECIFIC_NAME", re.compile(r"deriv(?:ative)?s?\.?|유도체", re.I)),
        ("COMPOUND_WORD_IN_SPECIFIC_NAME", re.compile(r"\bcompd\.?\b|\bcompound\b|화합물|착화합물", re.I)),
        ("STRUCTURAL_RANGE_IN_SPECIFIC_NAME", STRUCTURAL_RANGE_RE),
    ]:
        if pattern.search(name) or pattern.search(source):
            complexity_markers.append(label)

    reason = ["DIRECT_CAS_ASSIGNED_AND_NO_EXPLICIT_SCOPE_EXTENSION"]
    if complexity_markers:
        reason.extend(complexity_markers)
    return "DIRECT_CAS_ONLY", False, reason, ""


def inactive_deleted_row(name: str, source: str, direct_cas: str, raw_row: pd.Series) -> tuple[bool, str]:
    """Identify historical/deleted placeholder rows that are not active regulatory scopes.

    The approved master can retain rows whose current content is only '삭제'.  They are
    useful for historical change tracking but must not enter the denominator of a
    *current* CAS-gap prevalence estimate or the LLM benchmark.

    The test is deliberately strict so that a legitimate clause merely containing the
    word '삭제' is not discarded.
    """
    name_n = normalize_text(name)
    source_n = normalize_text(source)
    direct = bool(split_cas(direct_cas))

    explicit_deleted_tokens = {"삭제", "<삭제>", "삭 제", "deleted", "abolished"}
    if name_n.lower() in explicit_deleted_tokens:
        return True, "SUBSTANCE_NAME_DELETED_PLACEHOLDER"
    if source_n.lower() in explicit_deleted_tokens:
        return True, "SOURCE_TEXT_DELETED_PLACEHOLDER"

    # Some master versions carry a dedicated current/status/change column.
    for col in [
        "status", "current_status", "row_status", "change_type", "change_status",
        "revision_status", "designation_status", "현행여부", "상태", "변경유형",
    ]:
        if col in raw_row.index:
            value = normalize_text(raw_row.get(col, "")).lower()
            if value in explicit_deleted_tokens:
                return True, f"STATUS_COLUMN_DELETED:{col}"

    # Typical historical placeholder: no real name, no CAS, and source cell only says 삭제.
    name_placeholder = name_n in {"", "-", "–", "—"}
    if name_placeholder and not direct and re.fullmatch(r"<?\s*삭\s*제\s*>?", source_n):
        return True, "EMPTY_IDENTITY_DELETED_PLACEHOLDER"
    return False, ""


def build_benchmark(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    notice_col = first_existing(master, ["notice", "source_reference", "official_notice"])
    latest = latest_notice_from_master(master)
    x = master.copy()
    if notice_col and latest:
        selected = x[x[notice_col].astype(str).str.strip().eq(latest)].copy()
        if len(selected):
            x = selected

    name_col = first_existing(x, ["substance_name_ko", "substance_name", "chemical_name", "name"])
    source_col = first_existing(x, ["source_text", "regulatory_text", "group_title"])
    direct_col = first_existing(x, ["direct_cas", "direct_cas_final", "cas", "CAS"])
    rule_col = first_existing(x, ["rule_id", "designation_id"])
    desig_col = first_existing(x, ["designation_id", "rule_id"])
    subno_col = first_existing(x, ["sub_no", "item_no"])
    acute_col = first_existing(x, ["acute_threshold_pct", "acute_threshold_raw"])
    chronic_col = first_existing(x, ["chronic_threshold_pct", "chronic_threshold_raw"])
    eco_col = first_existing(x, ["eco_threshold_pct", "eco_threshold_raw"])

    rows = []
    for i, r in x.reset_index(drop=True).iterrows():
        name = clean(r.get(name_col, "")) if name_col else ""
        source = clean(r.get(source_col, "")) if source_col else ""
        direct = clean(r.get(direct_col, "")) if direct_col else ""
        scope, positive, reasons, evidence = classify_scope(source, name, direct)
        source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16] if source else ""
        rows.append({
            "benchmark_id": f"CASGAP_{i+1:05d}",
            "notice": latest,
            "rule_id": clean(r.get(rule_col, "")) if rule_col else "",
            "designation_id": clean(r.get(desig_col, "")) if desig_col else "",
            "sub_no": clean(r.get(subno_col, "")) if subno_col else "",
            "substance_name_ko": name,
            "source_text": source,
            "source_text_sha256_16": source_hash,
            "direct_cas": ";".join(split_cas(direct)),
            "has_direct_cas": bool(split_cas(direct)),
            "reference_scope_type": scope,
            # New precise name plus compatibility alias for scripts already built.
            "reference_cas_only_insufficient": bool(positive),
            "reference_requires_extended_identity": bool(positive),
            "reference_reason": ";".join(reasons),
            "reference_evidence_phrase": evidence,
            "reference_rule_version": REFERENCE_RULE_VERSION,
            "acute_threshold": clean(r.get(acute_col, "")) if acute_col else "",
            "chronic_threshold": clean(r.get(chronic_col, "")) if chronic_col else "",
            "eco_threshold": clean(r.get(eco_col, "")) if eco_col else "",
            "reference_type": "SOURCE_DERIVED_PREDEFINED_OPERATIONAL_REFERENCE_NOT_EXPERT_GOLD",
            "inactive_deleted_row": inactive_deleted_row(name, source, direct, r)[0],
            "benchmark_exclusion_reason": inactive_deleted_row(name, source, direct, r)[1],
        })
    full_all = pd.DataFrame(rows)

    # IMPORTANT: preserve benchmark_id assigned above, then exclude inactive/deleted
    # placeholders.  This keeps IDs stable across reference refinements and makes cached
    # LLM results reusable for unchanged active rows.
    excluded = full_all[full_all["inactive_deleted_row"].astype(bool)].copy()
    full = full_all[~full_all["inactive_deleted_row"].astype(bool)].copy().reset_index(drop=True)

    # Primary balanced benchmark: all ACTIVE source-defined CAS-only-insufficient rows
    # + reproducible sample of direct-CAS-only controls.
    pos = full[full["reference_cas_only_insufficient"]].copy()
    neg = full[~full["reference_cas_only_insufficient"]].copy()
    n_neg = min(len(neg), int(round(len(pos) * max(0.0, NEGATIVE_RATIO))))
    sampled_neg = neg.sample(n=n_neg, random_state=BENCHMARK_SEED) if n_neg and len(neg) else neg.iloc[0:0]
    primary = pd.concat([pos, sampled_neg], ignore_index=True)
    if len(primary):
        primary = primary.sample(frac=1, random_state=BENCHMARK_SEED).reset_index(drop=True)
    primary["benchmark_set"] = "PRIMARY_BALANCED_CAS_ONLY_INSUFFICIENCY"

    summary = (
        full.groupby(["reference_scope_type", "reference_cas_only_insufficient"], dropna=False)
        .size().reset_index(name="n_full_rows")
        .rename(columns={"reference_cas_only_insufficient": "reference_requires_extended_identity"})
    )
    pcounts = primary.groupby("reference_scope_type").size().rename("n_primary_rows")
    summary = summary.merge(pcounts, left_on="reference_scope_type", right_index=True, how="left")
    summary["n_primary_rows"] = summary["n_primary_rows"].fillna(0).astype(int)
    return full, primary, summary, excluded


def reference_reclassification_audit(full: pd.DataFrame) -> pd.DataFrame:
    """Flag complex CAS-assigned names that are intentionally NOT treated as gaps."""
    if full.empty:
        return pd.DataFrame()
    x = full.copy()
    direct_complex = (
        x["has_direct_cas"].astype(bool)
        & ~x["reference_cas_only_insufficient"].astype(bool)
        & x["reference_reason"].astype(str).str.contains(
            "SALT_WORD|REACTION_PRODUCT_WORD|MIXTURE_UVCB_WORD|DERIVATIVE_WORD|COMPOUND_WORD|STRUCTURAL_RANGE",
            regex=True,
        )
    )
    cols = [
        "benchmark_id", "rule_id", "designation_id", "substance_name_ko", "direct_cas",
        "source_text", "reference_scope_type", "reference_cas_only_insufficient",
        "reference_reason", "reference_rule_version",
    ]
    return x.loc[direct_complex, [c for c in cols if c in x.columns]].copy()

def source_traceability_audit(benchmark: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in benchmark.iterrows():
        src = clean(r.get("source_text"))
        direct = split_cas(r.get("direct_cas"))
        rows.append({
            "benchmark_id": r.get("benchmark_id"),
            "source_text_present": bool(src),
            "direct_cas_count": len(direct),
            "all_direct_cas_traceable_in_source": all(c in src for c in direct) if direct else True,
            "reference_reason": r.get("reference_reason"),
            "reference_scope_type": r.get("reference_scope_type"),
        })
    return pd.DataFrame(rows)





def _norm_designation_for_audit(v: Any) -> str:
    s = normalize_text(v)
    s = _normalize_hwp_dashes(s)
    return re.sub(r"\s*의\s*소번호\s*", "의 소번호 ", s).strip()


def _unique_nonempty(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        s = normalize_text(value)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def master_for_official_source_audit(master: pd.DataFrame, parsed: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Restrict the frozen master to the same regulatory snapshot as the official source.

    The approved master can contain several historical notices. Comparing a current
    official appendix (e.g. 2026-5) against every historical master row inflates the
    apparent number of master-only pairs. The benchmark already uses the latest
    notice only; source audit must use the same temporal scope.
    """
    m = master.copy()
    meta: dict[str, Any] = {
        "master_rows_before_source_filter": int(len(m)),
        "master_source_notice_filter": "",
        "master_source_regime_filter": "",
    }

    parsed_notice = ""
    if parsed is not None and not parsed.empty and "official_notice" in parsed.columns:
        vals = _unique_nonempty(parsed["official_notice"].tolist())
        if vals:
            parsed_notice = vals[0]

    notice_col = first_existing(m, ["notice", "source_reference", "official_notice"])
    if parsed_notice and notice_col:
        target_key = notice_key(parsed_notice)
        mask = m[notice_col].map(lambda v: notice_key(v) == target_key)
        candidate = m.loc[mask].copy()
        if len(candidate):
            m = candidate
            meta["master_source_notice_filter"] = parsed_notice

    # Regime filtering is applied only when the master actually uses the same label.
    # This avoids forcing an incompatible vocabulary across legacy master versions.
    parsed_regime = ""
    if parsed is not None and not parsed.empty and "regime" in parsed.columns:
        vals = _unique_nonempty(parsed["regime"].tolist())
        if vals:
            parsed_regime = vals[0]
    regime_col = first_existing(m, ["regime", "regulatory_regime", "category", "regulatory_category"])
    if parsed_regime and regime_col:
        norm_target = normalize_text(parsed_regime).upper()
        mask = m[regime_col].map(lambda v: normalize_text(v).upper() == norm_target)
        candidate = m.loc[mask].copy()
        if len(candidate):
            m = candidate
            meta["master_source_regime_filter"] = parsed_regime

    meta["master_rows_compared_to_official_source"] = int(len(m))
    return m, meta


def _pair_frame_from_official(parsed: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "designation_id", "cas", "official_substance_name", "official_sub_no",
        "official_hazard_type", "official_source_excel_row",
    ]
    if parsed is None or parsed.empty:
        return pd.DataFrame(columns=cols)
    rows: list[dict[str, Any]] = []
    for _, r in parsed.iterrows():
        d = _norm_designation_for_audit(r.get("designation_id"))
        for c in split_cas(r.get("cas")):
            if d and c:
                rows.append({
                    "designation_id": d,
                    "cas": c,
                    "official_substance_name": normalize_text(r.get("substance_name")),
                    "official_sub_no": normalize_text(r.get("sub_no")),
                    "official_hazard_type": normalize_text(r.get("hazard_type")),
                    "official_source_excel_row": normalize_text(r.get("source_excel_row")),
                })
    if not rows:
        return pd.DataFrame(columns=cols)
    x = pd.DataFrame(rows)
    agg = {
        c: (lambda s: " | ".join(_unique_nonempty(s.tolist())))
        for c in x.columns if c not in {"designation_id", "cas"}
    }
    return x.groupby(["designation_id", "cas"], as_index=False, dropna=False).agg(agg)


def _pair_frame_from_master(master: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "designation_id", "cas", "master_substance_name", "master_rule_id",
        "master_notice", "master_source_text",
    ]
    if master is None or master.empty:
        return pd.DataFrame(columns=cols)
    desig_col = first_existing(master, ["designation_id", "rule_id"])
    cas_col = first_existing(master, ["direct_cas", "direct_cas_final", "cas", "CAS"])
    if desig_col is None or cas_col is None:
        return pd.DataFrame(columns=cols)
    name_col = first_existing(master, ["substance_name_ko", "substance_name", "chemical_name", "name"])
    rule_col = first_existing(master, ["rule_id", "designation_id"])
    notice_col = first_existing(master, ["notice", "source_reference", "official_notice"])
    source_col = first_existing(master, ["source_text", "regulatory_text", "group_title"])
    rows: list[dict[str, Any]] = []
    for _, r in master.iterrows():
        d = _norm_designation_for_audit(r.get(desig_col))
        for c in split_cas(r.get(cas_col)):
            if d and c:
                rows.append({
                    "designation_id": d,
                    "cas": c,
                    "master_substance_name": normalize_text(r.get(name_col)) if name_col else "",
                    "master_rule_id": normalize_text(r.get(rule_col)) if rule_col else "",
                    "master_notice": normalize_text(r.get(notice_col)) if notice_col else "",
                    "master_source_text": normalize_text(r.get(source_col)) if source_col else "",
                })
    if not rows:
        return pd.DataFrame(columns=cols)
    x = pd.DataFrame(rows)
    agg = {
        c: (lambda s: " | ".join(_unique_nonempty(s.tolist())))
        for c in x.columns if c not in {"designation_id", "cas"}
    }
    return x.groupby(["designation_id", "cas"], as_index=False, dropna=False).agg(agg)


def appendix_master_pair_differences(parsed: pd.DataFrame, master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return auditable pair-level differences after temporal/regime alignment."""
    audit_master, filter_meta = master_for_official_source_audit(master, parsed)
    official = _pair_frame_from_official(parsed)
    frozen = _pair_frame_from_master(audit_master)

    official_keys = set(map(tuple, official[["designation_id", "cas"]].to_numpy())) if len(official) else set()
    master_keys = set(map(tuple, frozen[["designation_id", "cas"]].to_numpy())) if len(frozen) else set()
    master_only_keys = master_keys - official_keys
    official_only_keys = official_keys - master_keys

    mo = frozen[frozen.apply(lambda r: (r["designation_id"], r["cas"]) in master_only_keys, axis=1)].copy() if len(frozen) else frozen.copy()
    oo = official[official.apply(lambda r: (r["designation_id"], r["cas"]) in official_only_keys, axis=1)].copy() if len(official) else official.copy()

    official_designations = set(official["designation_id"]) if len(official) else set()
    official_cas = set(official["cas"]) if len(official) else set()
    if len(mo):
        mo["same_designation_exists_in_official"] = mo["designation_id"].isin(official_designations)
        mo["same_cas_exists_elsewhere_in_official"] = mo["cas"].isin(official_cas)
        def _master_status(r: pd.Series) -> str:
            if r["same_designation_exists_in_official"] and r["same_cas_exists_elsewhere_in_official"]:
                return "PAIRING_DIFFERS_FROM_CURRENT_OFFICIAL_XLSX"
            if r["same_designation_exists_in_official"]:
                return "ADDITIONAL_MASTER_CAS_FOR_EXISTING_DESIGNATION"
            if r["same_cas_exists_elsewhere_in_official"]:
                return "MASTER_DESIGNATION_ABSENT_BUT_CAS_EXISTS_ELSEWHERE"
            return "PAIR_NOT_FOUND_IN_CURRENT_OFFICIAL_XLSX"
        mo["audit_status"] = mo.apply(_master_status, axis=1)
        mo["manual_source_review_required"] = True
    else:
        mo = pd.DataFrame(columns=list(frozen.columns) + [
            "same_designation_exists_in_official", "same_cas_exists_elsewhere_in_official",
            "audit_status", "manual_source_review_required",
        ])

    master_designations = set(frozen["designation_id"]) if len(frozen) else set()
    master_cas = set(frozen["cas"]) if len(frozen) else set()
    if len(oo):
        oo["same_designation_exists_in_master"] = oo["designation_id"].isin(master_designations)
        oo["same_cas_exists_elsewhere_in_master"] = oo["cas"].isin(master_cas)
        oo["audit_status"] = "OFFICIAL_PAIR_MISSING_FROM_FROZEN_MASTER"
        oo["manual_source_review_required"] = True
    else:
        oo = pd.DataFrame(columns=list(official.columns) + [
            "same_designation_exists_in_master", "same_cas_exists_elsewhere_in_master",
            "audit_status", "manual_source_review_required",
        ])

    overlap = official_keys & master_keys
    summary = pd.DataFrame([{
        **filter_meta,
        "official_unique_pairs": len(official_keys),
        "master_unique_pairs": len(master_keys),
        "pair_overlap": len(overlap),
        "master_only_pairs": len(master_only_keys),
        "official_only_pairs": len(official_only_keys),
        "official_pair_coverage_by_master": (len(overlap) / len(official_keys)) if official_keys else np.nan,
        "master_pair_coverage_by_official": (len(overlap) / len(master_keys)) if master_keys else np.nan,
        "bidirectional_exact_pair_match": bool(not master_only_keys and not official_only_keys),
    }])
    return mo.reset_index(drop=True), oo.reset_index(drop=True), summary


def appendix_master_crosscheck(parsed: pd.DataFrame, master: pd.DataFrame) -> dict[str, Any]:
    """Cross-check official designation/CAS pairs against the SAME master snapshot.

    IMPORTANT: the frozen master may contain historical notices. The audit first
    aligns the master to the official source's notice (and regime when safely
    possible), preventing obsolete historical pairs from being counted as current
    source discrepancies.
    """
    if parsed is None or parsed.empty or master is None or master.empty:
        return {
            "pdf_unique_pairs": 0, "master_unique_pairs": 0, "pair_overlap": 0,
            "pdf_pair_match_rate": np.nan, "master_pair_coverage_rate": np.nan,
            "master_rows_before_source_filter": int(len(master)) if master is not None else 0,
            "master_rows_compared_to_official_source": 0,
        }

    audit_master, filter_meta = master_for_official_source_audit(master, parsed)
    official = _pair_frame_from_official(parsed)
    frozen = _pair_frame_from_master(audit_master)
    official_pairs = set(map(tuple, official[["designation_id", "cas"]].to_numpy())) if len(official) else set()
    master_pairs = set(map(tuple, frozen[["designation_id", "cas"]].to_numpy())) if len(frozen) else set()
    overlap = official_pairs & master_pairs
    return {
        "pdf_unique_pairs": len(official_pairs),
        "master_unique_pairs": len(master_pairs),
        "pair_overlap": len(overlap),
        "pdf_pair_match_rate": (len(overlap) / len(official_pairs)) if official_pairs else np.nan,
        "master_pair_coverage_rate": (len(overlap) / len(master_pairs)) if master_pairs else np.nan,
        **filter_meta,
    }

def main() -> None:
    print("=" * 80)
    print("01 / OFFICIAL DATA + SOURCE AUDIT + CAS-GAP BENCHMARK")
    print("=" * 80)

    ok, seed_status = download_seed(SEED_MASTER_URL, SEED_MASTER_LOCAL)
    if not ok:
        raise FileNotFoundError(
            "Approved regulatory master is unavailable. Put regulatory_rules_master_APPROVED.csv "
            f"in {DATA_DIR} or check internet access. Detail: {seed_status}"
        )
    master = read_csv_robust(SEED_MASTER_LOCAL)
    latest_local = latest_notice_from_master(master)
    print(f"Approved master: {SEED_MASTER_LOCAL.name} | rows={len(master):,} | latest={latest_local}")

    # Optional source-supported broad-salt materials from the user's existing project.
    bs_rule_ok, bs_rule_status = download_seed(BROAD_SALT_RULES_URL, BROAD_SALT_RULES_LOCAL)
    bs_val_ok, bs_val_status = download_seed(BROAD_SALT_VALIDATION_URL, BROAD_SALT_VALIDATION_LOCAL)

    human_xlsx_path = find_human_hazard_official_xlsx()
    if human_xlsx_path:
        print(f"Official HUMAN_HAZARD Excel audit source: {human_xlsx_path.name}")
    else:
        print("Official HUMAN_HAZARD Excel audit source: not found; PDF/HWP fallback will be used")

    api_status_rows = []
    appendix_frames = []
    human_master_only_pairs = pd.DataFrame()
    human_official_only_pairs = pd.DataFrame()
    human_pair_diff_summary = pd.DataFrame()
    for spec in NOTICE_SPECS:
        info, status = query_notice(spec)
        status["approved_master_latest_notice"] = latest_local
        if info and spec["regime"] == "HUMAN_HAZARD":
            status["master_sync_status"] = (
                "CURRENT" if notice_key(info.get("official_notice")) == notice_key(latest_local)
                else "OFFICIAL_NEWER" if notice_key(info.get("official_notice")) > notice_key(latest_local)
                else "LOCAL_NEWER"
            )
        payload, body_status = fetch_notice_body(info) if info else ({}, "NOT_AVAILABLE")
        status["body_status"] = body_status
        if payload:
            pending_json = INTERMEDIATE / f"01_{spec['regime']}_{clean(info.get('official_notice'))}_official.json"
            pending_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            # HUMAN_HAZARD: prefer the official machine-readable full-list Excel.
            # The law.go.kr PDF is a viewer/help stub and is not the data-bearing source.
            if spec["regime"] == "HUMAN_HAZARD" and human_xlsx_path:
                try:
                    parsed, xlsx_meta = parse_human_hazard_official_xlsx(human_xlsx_path)
                    parsed = parsed.copy()
                    parsed["regime"] = spec["regime"]
                    parsed["official_notice"] = clean(info.get("official_notice"))
                    parsed["official_title"] = clean(info.get("official_title"))
                    parsed["official_serial"] = clean(info.get("official_serial"))
                    parsed["appendix_title"] = "인체등유해성물질 목록(전체)"
                    parsed["parser_type"] = "HUMAN_HAZARD_OFFICIAL_XLSX"
                    parsed["source_type"] = "OFFICIAL_XLSX_LOCAL"
                    parsed["approval_status"] = "SOURCE_AUDIT_ONLY"
                    appendix_frames.append(parsed)

                    cross = appendix_master_crosscheck(parsed, master)
                    status.update(cross)
                    # Generic aliases; legacy pdf_* fields remain for 03 compatibility.
                    status["official_source_unique_pairs"] = cross.get("pdf_unique_pairs")
                    status["master_unique_pairs_official_audit"] = cross.get("master_unique_pairs")
                    status["official_source_pair_overlap"] = cross.get("pair_overlap")
                    status["official_source_pair_match_rate"] = cross.get("pdf_pair_match_rate")
                    status["master_pair_coverage_rate_official_source"] = cross.get("master_pair_coverage_rate")

                    human_master_only_pairs, human_official_only_pairs, human_pair_diff_summary = appendix_master_pair_differences(parsed, master)
                    if len(human_pair_diff_summary):
                        q = human_pair_diff_summary.iloc[0]
                        status["master_only_pair_count"] = int(q.get("master_only_pairs", 0))
                        status["official_only_pair_count"] = int(q.get("official_only_pairs", 0))
                        status["bidirectional_exact_pair_match"] = bool(q.get("bidirectional_exact_pair_match", False))
                    status.update(xlsx_meta)
                    status["official_xlsx_notice_matches_api"] = (
                        not xlsx_meta.get("official_xlsx_notice")
                        or notice_key(xlsx_meta.get("official_xlsx_notice")) == notice_key(info.get("official_notice"))
                    )
                    status["appendix_attachment_count"] = 1
                    status["appendix_parsed_count"] = 1
                    status["appendix_rows"] = int(len(parsed))
                    appendix_status = [{
                        "appendix": "인체등유해성물질 목록(전체)",
                        "status": "PARSED",
                        "source_type": "OFFICIAL_XLSX_LOCAL",
                        "parser_type": "HUMAN_HAZARD_OFFICIAL_XLSX",
                        "rows": int(len(parsed)),
                        "active_rows": int(xlsx_meta["official_xlsx_active_rows"]),
                        "deleted_rows": int(xlsx_meta["official_xlsx_deleted_rows"]),
                        "inferred_designation_rows": int(xlsx_meta["official_xlsx_inferred_designation_rows"]),
                        "sha256": xlsx_meta["official_xlsx_sha256"],
                        "source_file": xlsx_meta["official_xlsx_name"],
                        "notice_matches_api": bool(status["official_xlsx_notice_matches_api"]),
                    }]
                    status["appendix_status_json"] = json.dumps(appendix_status, ensure_ascii=False)
                except Exception as exc:
                    status["official_xlsx_error"] = f"{type(exc).__name__}: {exc}"
                    parsed, appendix_status = audit_notice_pdf(info, payload)
                    if len(parsed):
                        appendix_frames.append(parsed)
                        status.update(appendix_master_crosscheck(parsed, master))
                    status["appendix_attachment_count"] = len(appendix_status)
                    status["appendix_parsed_count"] = sum(s.get("status") == "PARSED" for s in appendix_status)
                    status["appendix_rows"] = int(sum(int(s.get("rows", 0) or 0) for s in appendix_status))
                    status["appendix_status_json"] = json.dumps(appendix_status, ensure_ascii=False)
            else:
                parsed, appendix_status = audit_notice_pdf(info, payload)
                if len(parsed):
                    appendix_frames.append(parsed)
                    if spec["regime"] == "HUMAN_HAZARD":
                        status.update(appendix_master_crosscheck(parsed, master))
                status["appendix_attachment_count"] = len(appendix_status)
                status["appendix_parsed_count"] = sum(s.get("status") == "PARSED" for s in appendix_status)
                status["appendix_rows"] = int(sum(int(s.get("rows", 0) or 0) for s in appendix_status))
                status["appendix_status_json"] = json.dumps(appendix_status, ensure_ascii=False)
        else:
            status["appendix_attachment_count"] = 0
            status["appendix_parsed_count"] = 0
            status["appendix_rows"] = 0
            status["appendix_status_json"] = "[]"
        api_status_rows.append(status)

    api_status = pd.DataFrame(api_status_rows)
    appendix_candidates = pd.concat(appendix_frames, ignore_index=True, sort=False) if appendix_frames else pd.DataFrame()

    full, primary, scope_summary, excluded_rows = build_benchmark(master)
    trace = source_traceability_audit(full)
    reclass_audit = reference_reclassification_audit(full)

    # Save all machine-readable outputs. Newly parsed API/PDF data stay separate from approved master.
    master.to_csv(INTERMEDIATE / "01_approved_master_snapshot.csv", index=False, encoding="utf-8-sig")
    full.to_csv(INTERMEDIATE / "01_cas_gap_benchmark_full.csv", index=False, encoding="utf-8-sig")
    primary.to_csv(INTERMEDIATE / "01_cas_gap_benchmark_primary.csv", index=False, encoding="utf-8-sig")
    scope_summary.to_csv(INTERMEDIATE / "01_benchmark_scope_summary.csv", index=False, encoding="utf-8-sig")
    trace.to_csv(INTERMEDIATE / "01_source_traceability_audit.csv", index=False, encoding="utf-8-sig")
    reclass_audit.to_csv(INTERMEDIATE / "01_reference_reclassification_audit.csv", index=False, encoding="utf-8-sig")
    excluded_rows.to_csv(INTERMEDIATE / "01_benchmark_excluded_inactive_rows.csv", index=False, encoding="utf-8-sig")
    api_status.to_csv(INTERMEDIATE / "01_official_api_pdf_status.csv", index=False, encoding="utf-8-sig")
    appendix_candidates.to_csv(INTERMEDIATE / "01_official_appendix_pdf_candidates.csv", index=False, encoding="utf-8-sig")
    human_master_only_pairs.to_csv(INTERMEDIATE / "01_master_only_pairs_vs_official_xlsx.csv", index=False, encoding="utf-8-sig")
    human_official_only_pairs.to_csv(INTERMEDIATE / "01_official_only_pairs_vs_master.csv", index=False, encoding="utf-8-sig")
    human_pair_diff_summary.to_csv(INTERMEDIATE / "01_official_master_pair_diff_summary.csv", index=False, encoding="utf-8-sig")

    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "latest_approved_master_notice": latest_local,
        "approved_master_source": seed_status,
        "broad_salt_rules_source": bs_rule_status if bs_rule_ok else "UNAVAILABLE",
        "broad_salt_validation_source": bs_val_status if bs_val_ok else "UNAVAILABLE",
        "negative_ratio": NEGATIVE_RATIO,
        "seed": BENCHMARK_SEED,
        "primary_endpoint": "regulatory scope not safely representable by direct-CAS equality alone",
        "reference_rule_version": REFERENCE_RULE_VERSION,
        "benchmark_population_version": BENCHMARK_POPULATION_VERSION,
        "reference_type": "source-derived predefined operational reference; not expert gold",
        "api_used": bool(get_law_oc()),
        "human_hazard_official_xlsx": str(human_xlsx_path) if human_xlsx_path else "",
        "human_hazard_official_xlsx_used": bool(human_xlsx_path),
        "new_pdf_rows_promoted_to_approved_rules": False,
        "inactive_deleted_rows_excluded": int(len(excluded_rows)),
        "current_scope_denominator_excludes_deleted_placeholders": True,
    }
    (INTERMEDIATE / "01_run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Active full benchmark rows: {len(full):,}")
    print(f"Excluded inactive/deleted placeholder rows: {len(excluded_rows):,}")
    print(f"Primary balanced benchmark rows: {len(primary):,}")
    print(f"Specific CAS-assigned complex names reclassified as DIRECT_CAS_ONLY: {len(reclass_audit):,}")
    if len(scope_summary):
        print("\n[Scope composition]")
        print(scope_summary.to_string(index=False))
    print("\n[Official source audit]")
    show = [c for c in ["regime", "status", "official_notice", "master_sync_status", "body_status", "appendix_attachment_count", "appendix_parsed_count", "appendix_rows"] if c in api_status.columns]
    print(api_status[show].to_string(index=False) if len(api_status) else "No API status")
    if len(human_pair_diff_summary):
        q = human_pair_diff_summary.iloc[0]
        print("\n[Current official Excel vs current approved master pair QC]")
        print(f"Master rows compared after source alignment: {int(q.get('master_rows_compared_to_official_source', 0)):,}")
        print(f"Official unique designation/CAS pairs: {int(q.get('official_unique_pairs', 0)):,}")
        print(f"Master unique designation/CAS pairs: {int(q.get('master_unique_pairs', 0)):,}")
        print(f"Pair overlap: {int(q.get('pair_overlap', 0)):,}")
        print(f"Master-only pairs: {int(q.get('master_only_pairs', 0)):,}")
        print(f"Official-only pairs: {int(q.get('official_only_pairs', 0)):,}")
        print(f"Bidirectional exact pair match: {bool(q.get('bidirectional_exact_pair_match', False))}")

    print("\nSaved to: intermediate/")
    print("Next: python 03_make_paper_results_CLAUDE_FINAL_QC.py  # 02 rerun not needed if primary benchmark remains 210")


if __name__ == "__main__":
    main()
