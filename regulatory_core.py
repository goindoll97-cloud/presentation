# -*- coding: utf-8 -*-
"""
CHEMICAL REGULATORY AUTOMATION FRAMEWORK
========================================

연구의 중심
-----------
본 연구의 core performance benchmark는 인체급성유해성물질,
인체만성유해성물질 및 생태유해성물질 지정체계를 사용한다.
이는 해당 체계만이 기업의 규제대상이라는 의미가 아니다.

인체등유해성물질 지정체계는 하나의 규제체계 안에서
- direct CAS,
- salts / chemical groups / mixtures / reaction products / structural ranges,
- concentration thresholds,
- hazard-specific criteria
가 함께 존재하므로 chemical regulatory screening의 automation boundary를
통제된 조건에서 평가하기 위한 core test bed로 사용한다.

기업의 실제 chemical compliance 업무는 사고대비물질, 제한물질,
금지물질, 허가물질 등 추가적인 regulated chemical categories를 함께
검토해야 한다. 이들 체계에서는 identity 외에도 use, quantity,
temporal condition, authorization status와 같은 추가 정보가
regulatory applicability를 결정할 수 있다.

따라서 본 코드는 두 층으로 구성한다.

CORE PERFORMANCE LAYER
----------------------
STAGE 0  CAS-only / CAS+concentration failure-mode evidence
STAGE 1  Approved regulatory-version comparison and enterprise backbone
STAGE 2  Local LLM semantic scope parsing
STAGE 3  Controlled automation-boundary benchmark with selective abstention
STAGE 4  Rule/update routing and explainable enterprise alerts
STAGE 5  Source traceability and paper figures

CROSS-REGIME GENERALIZABILITY LAYER
-----------------------------------
STAGE 6  Major regulated-chemical categories를 공통 machine-readable schema로 변환
         applicability dimensions:
         identity / concentration / use / quantity / time / authorization

         routing principles:
         - DETERMINISTIC_AUTO_CANDIDATE
         - AI_ASSISTED_IDENTITY_CANDIDATE
         - AI_ASSISTED_CONDITION_CANDIDATE
         - ADDITIONAL_ENTERPRISE_DATA_REQUIRED
         - REVIEW_REQUIRED

중요한 해석 원칙
----------------
- AI는 최종 법적 위반 여부를 판정하지 않는다.
- 명확한 CAS/수치조건은 deterministic logic이 처리한다.
- Local LLM은 semantic scope/condition interpretation에 제한적으로 사용한다.
- 필요한 기업측 입력정보(use, quantity, authorization 등)가 없으면
  모델성능과 무관하게 ADDITIONAL_ENTERPRISE_DATA_REQUIRED로 분류한다.
- 복잡한 identity membership을 근거 없이 추론하지 않고 REVIEW_REQUIRED로 보낸다.
- cross-regime 외부 공식 rule 파일이 없으면 결과를 임의 생성하지 않고 INPUT_REQUIRED로 기록한다.
- controlled benchmark 비율은 실제 기업의 오류율 또는 업무시간 절감률이 아니다.
"""


from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


# ============================================================
# 0. 경로
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent

MASTER_FILE = (
    SCRIPT_DIR
    / "data"
    / "master_v2"
    / "01_regulatory_rules_master_DRAFT_V2.csv"
)

def resolve_writable_research_output_dir():
    import os

    candidates = []

    env_dir = os.getenv(
        "REGINTEL_OUTPUT_DIR",
        "",
    ).strip()

    if env_dir:
        candidates.append(
            Path(env_dir)
        )

    candidates += [
        SCRIPT_DIR / "paper_output",
        SCRIPT_DIR / "paper_output_regintel_boundary",
    ]

    for cand in candidates:
        try:
            cand.mkdir(
                parents=True,
                exist_ok=True,
            )

            probe = (
                cand
                / ".regintel_write_test"
            )
            probe.write_text(
                "ok",
                encoding="utf-8",
            )
            probe.unlink(
                missing_ok=True
            )

            return cand
        except Exception:
            continue

    raise PermissionError(
        "쓰기 가능한 연구 output 폴더를 찾지 못했습니다. "
        "REGINTEL_OUTPUT_DIR 환경변수를 지정하세요."
    )


OUT_DIR = resolve_writable_research_output_dir()
OUT_XLSX = OUT_DIR / "LEGACY_MONOLITH_OUTPUT_NOT_USED.xlsx"

FIGURE_DIR = OUT_DIR / "paper_figures"
FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 0-1. 기업형/자동업데이트 설정
# ============================================================
REGULATION_TITLE = (
    "인체급성유해성물질, 인체만성유해성물질 및 "
    "생태유해성물질의 지정고시"
)

# 공식 최신 고시 확인: 실패해도 현재 로컬 master로 계속 진행
CHECK_OFFICIAL_UPDATE = (
    __import__("os").getenv("CHECK_OFFICIAL_UPDATE", "1")
    .strip().lower()
    not in {"0", "false", "no", "off"}
)

# 새 고시는 자동 발견/다운로드/후보파싱까지만 수행.
# 전문가 확인 후 아래 환경변수에 해당 고시번호를 넣어야 master에 승인 반영.
# 예: $env:APPROVED_NOTICE="2027-3"
APPROVED_NOTICE = (
    __import__("os").getenv("APPROVED_NOTICE", "").strip()
)

# 회사 inventory 경로.
# 미지정 시 script 폴더의 company_inventory.xlsx/csv를 자동 탐색.
COMPANY_INVENTORY_FILE = (
    __import__("os").getenv("COMPANY_INVENTORY_FILE", "").strip()
)

# 검증된 identity relation 파일(선택).
# 예: member CAS가 특정 규제 designation의 salt/group member임을 전문가가 검증한 표.
IDENTITY_RELATION_FILE = (
    __import__("os").getenv("IDENTITY_RELATION_FILE", "").strip()
)


# 선택적 regulatory structure reference.
# 열 예:
# designation_id, reference_smiles, relation_type, validated, evidence_source
REGULATORY_STRUCTURE_FILE = (
    __import__("os").getenv("REGULATORY_STRUCTURE_FILE", "").strip()
)

# 실제 회사 inventory가 없으면 논문/프로토타입 확인을 위한 synthetic demo 생성.
USE_DEMO_INVENTORY_IF_MISSING = (
    __import__("os").getenv("USE_DEMO_INVENTORY_IF_MISSING", "1")
    .strip().lower()
    not in {"0", "false", "no", "off"}
)

RAW_DIR = SCRIPT_DIR / "data" / "raw"

# ------------------------------------------------------------
# Cross-regime generalizability input
# ------------------------------------------------------------
# Optional standardized official-rule file for categories outside the
# core human-hazard benchmark. XLSX/CSV supported.
# Recommended regimes:
#   ACCIDENT_PREPAREDNESS, RESTRICTED, PROHIBITED, AUTHORIZED
# The code never fabricates missing official rules; absent regimes are
# explicitly reported as INPUT_REQUIRED.
CROSS_REGIME_RULES_FILE = (
    __import__("os").getenv("CROSS_REGIME_RULES_FILE", "").strip()
)

CROSS_REGIME_DIR = SCRIPT_DIR / "data" / "cross_regime"

CROSS_REGIME_EXPECTED = [
    ("HUMAN_HAZARD", "인체등유해성물질", "CONTROLLED_BENCHMARK_AND_UNIFIED_SCOPE"),
    ("ACCIDENT_PREPAREDNESS", "사고대비물질", "UNIFIED_OPERATIONAL_SCOPE"),
    ("RESTRICTED", "제한물질", "UNIFIED_OPERATIONAL_SCOPE"),
    ("PROHIBITED", "금지물질", "UNIFIED_OPERATIONAL_SCOPE"),
    ("AUTHORIZED", "허가물질", "UNIFIED_OPERATIONAL_SCOPE"),
]

CROSS_REGIME_RULE_COLUMNS = [
    "regime",
    "rule_id",
    "designation_id",
    "substance_name",
    "direct_cas",
    "source_text",
    "concentration_threshold_pct",
    "concentration_threshold_raw",
    "use_condition",
    "quantity_threshold",
    "quantity_unit",
    "temporal_condition",
    "authorization_condition",
    "source_name",
    "source_reference",
    "source_locator",
    "official_serial",
    "official_issue_date",
    "official_effective_date",
    "extraction_status",
]

# Cross-regime official API ingestion is ON by default.
AUTO_FETCH_CROSS_REGIME_API = (
    __import__("os").getenv("AUTO_FETCH_CROSS_REGIME_API", "1")
    .strip().lower()
    not in {"0", "false", "no", "off"}
)

# The human-hazard notice remains the controlled benchmark master.
# The four additional regimes are fetched from the current official
# administrative-rule API on every run (unless AUTO_FETCH...=0).
CROSS_REGIME_NOTICE_SPECS = [
    {
        "regime": "ACCIDENT_PREPAREDNESS",
        "title": "사고대비물질의 지정",
        "query_terms": ["사고대비물질", "지정"],
        "split_mode": "DIRECT",
    },
    {
        "regime": "RESTRICTED_PROHIBITED",
        "title": "제한물질·금지물질의 지정",
        "query_terms": ["제한물질", "금지물질", "지정"],
        "split_mode": "RESTRICTED_PROHIBITED",
    },
    {
        "regime": "AUTHORIZED",
        "title": "허가물질 지정 등에 관한 규정",
        "query_terms": ["허가물질", "지정"],
        "split_mode": "DIRECT",
    },
]

CROSS_REGIME_API_RAW_DIR = CROSS_REGIME_DIR / "official_api_raw"

# Optional external override; Stage 0 also has an embedded fallback.
FACTORIAL_PIPELINE_FILE = (
    SCRIPT_DIR / "paper_pipeline_CONTROLLED_BENCHMARK_V2_FACTORIAL.py"
)

# Automation-boundary controlled challenge benchmark.
# Each semantic rule receives one below-threshold and one above-threshold case.
CHALLENGE_FACTORS = [0.80, 1.20]

# Existing AI results can be reused without loading the model again.
# This is useful for re-running deterministic/automation-boundary analysis.
AI_CACHE_CANDIDATES = [
    SCRIPT_DIR / "LEGACY_MONOLITH_OUTPUT_NOT_USED.xlsx",
    SCRIPT_DIR / "LOCAL_AI_REGULATORY_AUTOMATION_BOUNDARY_FINAL_FIXED.xlsx",
    SCRIPT_DIR / "LOCAL_AI_REGULATORY_AUTOMATION_BOUNDARY_FINAL_FIXED(1).xlsx",
    SCRIPT_DIR / "LOCAL_AI_REGULATORY_INTELLIGENCE_RESEARCH_FINAL.xlsx",
    SCRIPT_DIR / "paper_output" / "LOCAL_AI_REGULATORY_INTELLIGENCE_RESEARCH_FINAL.xlsx",
]


# ============================================================
# 1. 공통
# ============================================================
CAS_RE = re.compile(r"(?<!\d)(\d{2,7}-\d{2}-\d)(?!\d)")

BROAD_SALT_PATTERNS = [
    r"(?:와|과|및)\s*그\s*염류",
    r"(?:와|과|및)\s*그\s*염\b",
    r"\band\s+(?:all\s+)?(?:its|their)\s+salts\b",
]

MIXTURE_PATTERNS = [
    r"혼합물",
    r"반응혼합물",
    r"반응생성물",
    r"\bmixture\b",
    r"reaction mixture",
    r"reaction product",
    r"공중합체",
    r"\bcopolymer\b",
]

GENERIC_GROUP_PATTERNS = [
    r"화합물",
    r"\bcompounds?\b",
    r"유도체",
    r"\bderivatives?\b",
]

SALT_FAMILY_PATTERNS = [
    r"염류",
    r"\bsalts\b",
]

STRUCTURAL_RANGE_RE = re.compile(
    r"(?:C\s*=\s*\d+\s*[~∼\-]\s*\d+|탄소수\s*\d+\s*[~∼\-]\s*\d+)",
    flags=re.I,
)


def clean(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return "" if s.lower() in {"nan", "none", "null"} else s


def splitsemi(x) -> List[str]:
    s = clean(x)
    return [z.strip() for z in s.split(";") if z.strip()] if s else []


def parse_threshold(x) -> float:
    s = clean(x)
    if not s or s == "-":
        return np.nan
    s = s.replace("%", "").strip()
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)", s)
    return float(m.group(1)) if m else np.nan


def active_new_thresholds(row: pd.Series):
    vals = []
    for label, col in [
        ("acute", "acute_threshold_raw"),
        ("chronic", "chronic_threshold_raw"),
        ("eco", "eco_threshold_raw"),
    ]:
        v = parse_threshold(row.get(col))
        if np.isfinite(v):
            vals.append((label, v))
    return vals


def min_threshold(row: pd.Series) -> float:
    vals = active_new_thresholds(row)
    return min(v for _, v in vals) if vals else np.nan


def source_parts(text: str):
    return [p.strip() for p in clean(text).split("|")]


def find_category_index(parts) -> Optional[int]:
    for i, p in enumerate(parts):
        if any(k in p for k in ["급성", "만성", "생태"]) and len(p) <= 40:
            return i
    return None


def recover_cas_field(source_text: str) -> Tuple[str, List[str]]:
    parts = source_parts(source_text)
    cat_idx = find_category_index(parts)

    if cat_idx is None or cat_idx < 4:
        return "", []

    cas_field = parts[cat_idx - 4]

    if cas_field.strip() in {"-", "부여되지 않음", "미부여"}:
        return cas_field, []

    hits = list(dict.fromkeys(CAS_RE.findall(cas_field)))
    return cas_field, hits


def broad_salt_scope(text: str) -> bool:
    t = clean(text)
    return any(re.search(p, t, flags=re.I) for p in BROAD_SALT_PATTERNS)


def mixture_scope(text: str) -> bool:
    t = clean(text)
    return any(re.search(p, t, flags=re.I) for p in MIXTURE_PATTERNS)


def generic_group_scope(text: str) -> bool:
    t = clean(text)
    return any(re.search(p, t, flags=re.I) for p in GENERIC_GROUP_PATTERNS)


def salt_family_scope(text: str) -> bool:
    t = clean(text)
    return any(re.search(p, t, flags=re.I) for p in SALT_FAMILY_PATTERNS)


def structural_range_scope(text: str) -> bool:
    return bool(STRUCTURAL_RANGE_RE.search(clean(text)))



# ============================================================
# 1-1. 규제 버전 / 최신 고시 자동 인식
# ============================================================
NOTICE_RE = re.compile(r"(?P<year>\d{4})\s*[-－]\s*(?P<num>\d+)")


def notice_key(label: str):
    m = NOTICE_RE.search(clean(label))
    if not m:
        return (-1, -1)
    return (int(m.group("year")), int(m.group("num")))


def ordered_notices(master: pd.DataFrame):
    vals = [
        clean(x)
        for x in master["notice"].dropna().unique()
        if clean(x)
    ]
    return sorted(vals, key=notice_key)


def latest_two_notices(master: pd.DataFrame):
    vals = ordered_notices(master)
    if not vals:
        raise ValueError("master에 notice 값이 없습니다.")
    latest = vals[-1]
    previous = vals[-2] if len(vals) >= 2 else ""
    return previous, latest


def resolve_master_file() -> Path:
    """
    기존 프로젝트 경로를 우선 사용하되,
    같은 폴더에 master CSV가 있으면 fallback.
    """
    if MASTER_FILE.exists():
        return MASTER_FILE

    candidates = [
        SCRIPT_DIR / "01_regulatory_rules_master_DRAFT_V2.csv",
        SCRIPT_DIR / "data" / "master_v2" / "01_regulatory_rules_master_DRAFT.csv",
        SCRIPT_DIR.parent / "data" / "master_v2" / "01_regulatory_rules_master_DRAFT_V2.csv",
        SCRIPT_DIR.parent / "01_regulatory_rules_master_DRAFT_V2.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    return MASTER_FILE


def substantive_rule_rows_for_notice(
    master: pd.DataFrame,
    notice: str,
) -> pd.DataFrame:
    g = master[
        master["notice"].astype(str).eq(str(notice))
    ].copy()

    if "row_type" in g.columns:
        g = g[
            g["row_type"].astype(str).ne("continuation_detail")
        ].copy()

    # active threshold가 있는 규칙을 중심으로 사용.
    active_mask = g.apply(
        lambda r: len(active_new_thresholds(r)) > 0,
        axis=1,
    )
    if active_mask.any():
        g = g[active_mask].copy()

    return g


def direct_cas_list_from_row(row: pd.Series):
    vals = splitsemi(row.get("direct_cas"))
    if vals:
        return vals

    _, recovered = recover_cas_field(row.get("source_text"))
    return recovered


def normalized_rule_signature(row: pd.Series):
    """
    새/직전 고시 간 규칙의 실질적 변경을 탐지하기 위한 fingerprint.
    단순 서식 변경보다 CAS/함량/category/exception/본문 변화에 초점.
    """
    direct = sorted(direct_cas_list_from_row(row))
    exc = sorted(splitsemi(row.get("exception_cas")))

    sig = (
        clean(row.get("sub_no")),
        clean(row.get("substance_name_ko")),
        tuple(direct),
        clean(row.get("acute_threshold_raw")),
        clean(row.get("chronic_threshold_raw")),
        clean(row.get("eco_threshold_raw")),
        clean(row.get("category")),
        tuple(exc),
        re.sub(r"\s+", " ", clean(row.get("source_text"))),
    )
    return sig


def designation_summary(g: pd.DataFrame):
    if not len(g):
        return {
            "n_rows": 0,
            "substance_names": "",
            "direct_cas": "",
            "min_threshold_pct": np.nan,
            "has_extended_scope": False,
            "source_text": "",
        }

    all_direct = []
    thresholds = []
    extended = False

    for _, r in g.iterrows():
        all_direct.extend(direct_cas_list_from_row(r))
        t = min_threshold(r)
        if np.isfinite(t):
            thresholds.append(t)

        text = clean(r.get("source_text"))
        direct_here = direct_cas_list_from_row(r)

        # 기업 screening에서 extended identity는 보수적으로 정의:
        # 1) direct CAS가 없거나
        # 2) direct CAS 하나로 전체 "A와 그 염류" 범위를 대표할 수 없는 경우.
        #
        # 특정 direct-CAS 물질명에 '화합물', '유도체', 'salt' 등의
        # 단어가 포함됐다는 이유만으로 자동 review 대상으로 보내지 않는다.
        if (
            not direct_here
            or broad_salt_scope(text)
        ):
            extended = True

    return {
        "n_rows": len(g),
        "substance_names": "; ".join(
            g["substance_name_ko"]
            .dropna().astype(str).unique()[:6]
        ),
        "direct_cas": ";".join(dict.fromkeys(all_direct)),
        "min_threshold_pct": (
            min(thresholds) if thresholds else np.nan
        ),
        "has_extended_scope": extended,
        "source_text": " || ".join(
            g["source_text"]
            .dropna().astype(str).unique()[:4]
        ),
    }


def build_regulatory_update_diff(
    master: pd.DataFrame,
    previous_notice: str,
    latest_notice: str,
):
    """
    특정 정책개편을 연구주제로 두지 않고,
    '직전 승인 규제 버전 -> 최신 규제 버전'의 변경만 탐지.
    """
    old_all = substantive_rule_rows_for_notice(
        master, previous_notice
    )
    new_all = substantive_rule_rows_for_notice(
        master, latest_notice
    )

    ids = sorted(
        set(old_all["designation_id"].dropna().astype(str))
        | set(new_all["designation_id"].dropna().astype(str))
    )

    rows = []

    for did in ids:
        old = old_all[
            old_all["designation_id"].astype(str).eq(did)
        ].copy()
        new = new_all[
            new_all["designation_id"].astype(str).eq(did)
        ].copy()

        old_sigs = {
            normalized_rule_signature(r)
            for _, r in old.iterrows()
        }
        new_sigs = {
            normalized_rule_signature(r)
            for _, r in new.iterrows()
        }

        if not len(old) and len(new):
            change = "ADDED"
        elif len(old) and not len(new):
            change = "REMOVED"
        elif old_sigs == new_sigs:
            change = "UNCHANGED"
        else:
            change = "MODIFIED"

        old_sum = designation_summary(old)
        new_sum = designation_summary(new)

        changed_fields = []
        for key in [
            "substance_names",
            "direct_cas",
            "min_threshold_pct",
            "has_extended_scope",
        ]:
            a = old_sum[key]
            b = new_sum[key]
            if (
                (pd.isna(a) and pd.isna(b))
                or str(a) == str(b)
            ):
                continue
            changed_fields.append(key)

        # source text 자체가 바뀐 경우도 audit 표시
        if old_sigs != new_sigs and "rule_text_or_row_structure" not in changed_fields:
            changed_fields.append("rule_text_or_row_structure")

        rows.append({
            "previous_notice": previous_notice,
            "latest_notice": latest_notice,
            "designation_id": did,
            "update_change_type": change,
            "changed_fields": ";".join(changed_fields),
            "old_n_rows": old_sum["n_rows"],
            "new_n_rows": new_sum["n_rows"],
            "old_substance_names": old_sum["substance_names"],
            "new_substance_names": new_sum["substance_names"],
            "old_direct_cas": old_sum["direct_cas"],
            "new_direct_cas": new_sum["direct_cas"],
            "old_min_threshold_pct": old_sum["min_threshold_pct"],
            "new_min_threshold_pct": new_sum["min_threshold_pct"],
            "old_has_extended_scope": old_sum["has_extended_scope"],
            "new_has_extended_scope": new_sum["has_extended_scope"],
            "human_review_required": (
                change != "UNCHANGED"
                and (
                    old_sum["has_extended_scope"]
                    or new_sum["has_extended_scope"]
                )
            ),
            "old_rule_text": old_sum["source_text"],
            "new_rule_text": new_sum["source_text"],
        })

    return pd.DataFrame(rows)


# ============================================================
# 2. Stage 3 통합: 규제 변화
# ============================================================
NEW_COLS = [
    "acute_threshold_raw",
    "chronic_threshold_raw",
    "eco_threshold_raw",
]


def parse_old_threshold_information(row: pd.Series):
    old_general = parse_threshold(row.get("old_or_reference_threshold_raw"))

    text = clean(row.get("source_text"))
    parts = [p.strip() for p in text.split("|")]

    cat_idx = find_category_index(parts)
    tail = " ".join(parts[cat_idx + 1:]) if cat_idx is not None else ""

    if not np.isfinite(old_general):
        m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)", tail)
        if m:
            old_general = float(m.group(1))

    specials = []

    for m in re.finditer(
        r"([^:]{1,120}):\s*([0-9]+(?:\.[0-9]+)?)",
        tail
    ):
        name = m.group(1).strip()
        name = re.sub(
            r"^\s*\d+(?:\.\d+)?\s*",
            "",
            name
        ).strip()

        specials.append(
            (name, float(m.group(2)))
        )

    return old_general, specials


def compare_direction(old: float, new: float) -> str:
    if not np.isfinite(old) or not np.isfinite(new):
        return "review"

    if new < old - 1e-12:
        return "strengthened"

    if new > old + 1e-12:
        return "relaxed"

    return "maintained"


def row_transition(row: pd.Series):
    active = active_new_thresholds(row)

    if not active:
        return {
            "row_change": "deleted",
            "old_general_threshold": np.nan,
            "old_special_thresholds": "",
            "new_min_threshold": np.nan,
            "direction_components": "deleted",
        }

    new_min = min(v for _, v in active)

    old_general, specials = parse_old_threshold_information(row)

    dirs = []

    if np.isfinite(old_general):
        dirs.append(
            ("general", old_general, compare_direction(old_general, new_min))
        )
    else:
        dirs.append(
            ("general", np.nan, "review")
        )

    for name, old_val in specials:
        dirs.append(
            (name or "special", old_val, compare_direction(old_val, new_min))
        )

    dvals = [x[2] for x in dirs]

    if "review" in dvals:
        change = "review"
    elif len(set(dvals)) > 1:
        change = "mixed"
    else:
        change = dvals[0]

    return {
        "row_change": change,
        "old_general_threshold": old_general,
        "old_special_thresholds": ";".join(
            f"{name}:{val:g}" for name, val in specials
        ),
        "new_min_threshold": new_min,
        "direction_components": ";".join(
            f"{name}[{old if np.isfinite(old) else 'NA'}->{new_min:g}]={d}"
            for name, old, d in dirs
        ),
    }


def substantive_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df[
        df["row_type"].astype(str) != "continuation_detail"
    ].copy()

    return out if len(out) else df.copy()


def aggregate_designation(changes: List[str]) -> str:
    s = set(changes)

    if "review" in s:
        return "review"

    if s == {"deleted"}:
        return "deleted"

    if s == {"maintained"}:
        return "maintained"

    if "mixed" in s:
        return "mixed"

    if "strengthened" in s and "relaxed" in s:
        return "mixed"

    if "deleted" in s and len(s) > 1:
        return "mixed"

    if s.issubset({"maintained", "strengthened"}) and "strengthened" in s:
        return "strengthened"

    if s == {"relaxed"}:
        return "relaxed"

    if s.issubset({"maintained", "relaxed"}) and "relaxed" in s:
        return "mixed"

    return "mixed"


def build_transition(master: pd.DataFrame, target_notice: str):
    union = sorted(
        set(
            master.loc[
                master["notice"].eq("2025-2"),
                "designation_id"
            ].dropna().astype(str)
        )
        |
        set(
            master.loc[
                master["notice"].eq(target_notice),
                "designation_id"
            ].dropna().astype(str)
        )
    )

    designation_rows = []
    sub_rows = []

    for did in union:
        old = master[
            master["notice"].eq("2025-2")
            & master["designation_id"].eq(did)
        ].copy()

        target = master[
            master["notice"].eq(target_notice)
            & master["designation_id"].eq(did)
        ].copy()

        target_sub = substantive_rows(target)

        if len(target_sub) == 0:
            final = "deleted" if len(old) else "not_applicable"
            row_results = []

        elif len(old) == 0:
            final = "new"
            row_results = []

        else:
            row_results = []

            for _, r in target_sub.iterrows():
                row_results.append(
                    row_transition(r)
                )

            final = aggregate_designation(
                [z["row_change"] for z in row_results]
            )

        designation_rows.append({
            "comparison": f"2025-2_to_{target_notice}",
            "designation_id": did,
            "change_type": final,
            "old_exists": len(old) > 0,
            "target_exists": len(target_sub) > 0,
            "n_target_substantive_rows": len(target_sub),
            "old_substance_name": "; ".join(
                old["substance_name_ko"]
                .dropna()
                .astype(str)
                .unique()[:3]
            ),
            "target_substance_name": "; ".join(
                target_sub["substance_name_ko"]
                .dropna()
                .astype(str)
                .unique()[:5]
            ),
        })

        for (_, r), z in zip(
            target_sub.iterrows(),
            row_results
        ):
            sub_rows.append({
                "comparison": f"2025-2_to_{target_notice}",
                "designation_id": did,
                "sub_no": r.get("sub_no"),
                "substance_name_ko": r.get("substance_name_ko"),
                "direct_cas": r.get("direct_cas"),
                "category": r.get("category"),
                "old_threshold_raw": r.get("old_or_reference_threshold_raw"),
                "acute_threshold_raw": r.get("acute_threshold_raw"),
                "chronic_threshold_raw": r.get("chronic_threshold_raw"),
                "eco_threshold_raw": r.get("eco_threshold_raw"),
                **z,
            })

    return (
        pd.DataFrame(designation_rows),
        pd.DataFrame(sub_rows),
    )


def transition_counts(t2519: pd.DataFrame, t265: pd.DataFrame):
    rows = []

    for label, df in [
        ("2025-2_to_2025-19", t2519),
        ("2025-2_to_2026-5", t265),
    ]:
        vc = df["change_type"].value_counts()

        for k, v in vc.items():
            rows.append({
                "comparison": label,
                "change_type": k,
                "n_designations": int(v),
                "proportion": float(v / len(df)),
                "total_designations": len(df),
            })

    return pd.DataFrame(rows)


# ============================================================
# 3. Stage 4 V3 통합: direct CAS / RQ2 / RQ3
# ============================================================
def build_identity_registry(master: pd.DataFrame, notice: str = ""):
    if not notice:
        _, notice = latest_two_notices(master)
    g = master[
        master["notice"].eq(notice)
        & (
            master["row_type"]
            .astype(str)
            .ne("continuation_detail")
        )
    ].copy()

    rows = []

    for _, r in g.iterrows():
        thr = min_threshold(r)

        if not np.isfinite(thr):
            continue

        direct_stage2 = splitsemi(r.get("direct_cas"))
        cas_field, recovered = recover_cas_field(r.get("source_text"))

        direct_final = direct_stage2 if direct_stage2 else recovered

        rows.append({
            "rule_id": r.get("rule_id"),
            "designation_id": r.get("designation_id"),
            "sub_no": r.get("sub_no"),
            "substance_name_ko": r.get("substance_name_ko"),
            "substance_name_en": r.get("substance_name_en"),
            "category": r.get("category"),
            "min_threshold_pct": thr,
            "direct_cas_final": ";".join(direct_final),
            "has_direct_cas_final": bool(direct_final),
            "cas_field_raw": cas_field,
            "exception_cas": clean(r.get("exception_cas")),
            "source_text": r.get("source_text"),
        })

    reg = pd.DataFrame(rows)

    challenge = reg[
        ~reg["has_direct_cas_final"]
    ].copy()

    challenge["primary_identity_class"] = challenge.apply(
        identity_class,
        axis=1,
    )

    return reg, challenge


def identity_class(row: pd.Series) -> str:
    text = clean(row["source_text"])

    if broad_salt_scope(text):
        return "BROAD_SALT_SCOPE"

    if mixture_scope(text):
        return "MIXTURE_OR_REACTION_SCOPE"

    if structural_range_scope(text):
        return "STRUCTURAL_RANGE_NO_DIRECT_CAS"

    if generic_group_scope(text):
        return "GENERIC_COMPOUND_OR_DERIVATIVE_SCOPE"

    if salt_family_scope(text):
        return "SALT_FAMILY_NO_DIRECT_CAS"

    return "OTHER_NO_DIRECT_CAS"


def build_identity_summary(challenge: pd.DataFrame):
    rows = [{
        "summary_type": "TOTAL",
        "identity_class": "ALL_TRUE_NO_DIRECT_CAS",
        "n_rule_rows": len(challenge),
        "n_unique_designations": challenge["designation_id"].nunique(),
    }]

    for cls, g in challenge.groupby("primary_identity_class"):
        rows.append({
            "summary_type": "PRIMARY_CLASS",
            "identity_class": cls,
            "n_rule_rows": len(g),
            "n_unique_designations": g["designation_id"].nunique(),
        })

    return pd.DataFrame(rows)


def build_rq2_summary(registry: pd.DataFrame):
    n = int(
        registry["has_direct_cas_final"].sum()
    )

    return pd.DataFrame([
        {
            "scenario": "DIRECT_BELOW",
            "n_rules": n,
            "TierA_CAS_only_vs_TierB_CAS_plus_concentration_disagreement_n": n,
            "disagreement_rate": 1.0,
            "interpretation": (
                "CAS 일치하지만 함량기준 미만: "
                "CAS-only는 모두 과잉판정"
            ),
        },
        {
            "scenario": "DIRECT_EQUAL",
            "n_rules": n,
            "TierA_CAS_only_vs_TierB_CAS_plus_concentration_disagreement_n": 0,
            "disagreement_rate": 0.0,
            "interpretation": (
                "CAS 일치 + 함량기준과 동일: 판정 일치"
            ),
        },
        {
            "scenario": "DIRECT_ABOVE",
            "n_rules": n,
            "TierA_CAS_only_vs_TierB_CAS_plus_concentration_disagreement_n": 0,
            "disagreement_rate": 0.0,
            "interpretation": (
                "CAS 일치 + 함량기준 초과: 판정 일치"
            ),
        },
        {
            "scenario": "ALL_DESIGN_DEPENDENT",
            "n_rules": n * 3,
            "TierA_CAS_only_vs_TierB_CAS_plus_concentration_disagreement_n": n,
            "disagreement_rate": 1/3,
            "interpretation": (
                "세 scenario를 동일하게 구성한 설계값이며 "
                "실제 기업 오류율로 해석하지 않음"
            ),
        },
    ])


# ============================================================
# 4. Stage 5 통합: Tier C 케이스북
# ============================================================
REPRESENTATIVE_N = {
    "BROAD_SALT_SCOPE": 2,
    "SALT_FAMILY_NO_DIRECT_CAS": 2,
    "MIXTURE_OR_REACTION_SCOPE": 2,
    "GENERIC_COMPOUND_OR_DERIVATIVE_SCOPE": 2,
    "STRUCTURAL_RANGE_NO_DIRECT_CAS": 2,
    "OTHER_NO_DIRECT_CAS": 2,
}


def case_score(row: pd.Series):
    score = 0.0

    text = clean(row["source_text"])

    if CAS_RE.search(text):
        score += 2

    if "제외" in text or "다만" in text:
        score += 2

    if structural_range_scope(text):
        score += 3

    if clean(row.get("sub_no")):
        score += 1

    return score


def proposed_tierC_logic(cls: str):
    return {
        "BROAD_SALT_SCOPE": "VALIDATED_MEMBER_CAS_RELATION",
        "SALT_FAMILY_NO_DIRECT_CAS": "VALIDATED_MEMBER_CAS_RELATION",
        "MIXTURE_OR_REACTION_SCOPE": "VALIDATED_MIXTURE_RULE",
        "GENERIC_COMPOUND_OR_DERIVATIVE_SCOPE": "VALIDATED_GROUP_OR_STRUCTURE_RULE",
        "STRUCTURAL_RANGE_NO_DIRECT_CAS": "VALIDATED_STRUCTURE_RULE",
        "OTHER_NO_DIRECT_CAS": "VALIDATED_EXTERNAL_IDENTIFIER_OR_NAME_RULE",
    }.get(
        cls,
        "MANUAL_VALIDATION"
    )


def build_casebook(challenge: pd.DataFrame):
    x = challenge.copy()
    x["_score"] = x.apply(
        case_score,
        axis=1,
    )

    reps = []

    for cls, g in x.groupby("primary_identity_class"):
        n = REPRESENTATIVE_N.get(cls, 2)

        g = g.sort_values(
            ["_score", "min_threshold_pct"],
            ascending=[False, True],
        )

        used = set()

        for _, r in g.iterrows():
            did = clean(r["designation_id"])

            if did in used:
                continue

            reps.append(r)
            used.add(did)

            if len(
                [z for z in reps if z["primary_identity_class"] == cls]
            ) >= n:
                break

    rep = pd.DataFrame(reps)

    if len(rep):
        rep = rep.drop(
            columns=["_score"],
            errors="ignore",
        )

        rep = rep.sort_values(
            ["primary_identity_class", "designation_id"]
        ).reset_index(drop=True)

        rep.insert(
            0,
            "case_id",
            [f"C{i+1:02d}" for i in range(len(rep))]
        )

        rep["proposed_tierC_logic"] = rep[
            "primary_identity_class"
        ].map(proposed_tierC_logic)

        rep["validation_status"] = "NOT_VALIDATED"

    return rep


def build_validation_queue(challenge: pd.DataFrame):
    q = challenge.copy()

    q["proposed_tierC_logic"] = q[
        "primary_identity_class"
    ].map(proposed_tierC_logic)

    q["validated_member_cas"] = ""
    q["validated_relation_type"] = ""
    q["structure_rule"] = ""
    q["evidence_source"] = ""
    q["validated"] = 0
    q["notes"] = ""

    return q



# ============================================================
# 4-1. 공식 최신 고시 확인 / human-in-the-loop update
# ============================================================
def _recursive_dicts(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _recursive_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _recursive_dicts(v)


def _extract_notice_label(record: dict):
    number = clean(
        record.get("발령번호")
        or record.get("행정규칙발령번호")
        or ""
    )
    date = clean(record.get("발령일자") or "")

    # '제2027-3호' 같은 형태
    m = NOTICE_RE.search(number)
    if m:
        return f"{m.group('year')}-{int(m.group('num'))}"

    # 발령번호가 단순 숫자라면 발령일자의 연도와 결합
    nums = re.findall(r"\d+", number)
    year = date[:4] if re.match(r"^\d{4}", date) else ""

    if year and nums:
        return f"{year}-{int(nums[-1])}"

    return ""


def get_law_oc_with_source():
    """
    공식 API 인증값과 '인증값을 어디에서 찾았는지'만 반환한다.

    검색 순서
    --------
    1) 환경변수 LAW_OC
    2) LAW_OC_FILE 환경변수로 지정된 파일
    3) SCRIPT_DIR / oneclick_chemical_notice_pipeline.py
    4) 현재 작업폴더 / oneclick_chemical_notice_pipeline.py
    5) SCRIPT_DIR/.env, 현재 작업폴더/.env

    보안 원칙
    --------
    - 인증값 자체는 로그/결과파일에 기록하지 않는다.
    - source 문자열만 기록한다.
    """
    import os

    oc = os.getenv("LAW_OC", "").strip()
    if oc:
        return oc, "environment:LAW_OC"

    candidate_files = []

    explicit_file = os.getenv(
        "LAW_OC_FILE",
        "",
    ).strip()
    if explicit_file:
        candidate_files.append(
            Path(explicit_file)
        )

    candidate_files.extend([
        SCRIPT_DIR / "oneclick_chemical_notice_pipeline.py",
        Path.cwd() / "oneclick_chemical_notice_pipeline.py",
        SCRIPT_DIR.parent / "oneclick_chemical_notice_pipeline.py",
        SCRIPT_DIR / ".env",
        Path.cwd() / ".env",
        SCRIPT_DIR.parent / ".env",
    ])

    # 중복경로 제거
    unique_files = []
    seen = set()
    for f in candidate_files:
        try:
            key = str(f.resolve())
        except Exception:
            key = str(f)
        if key not in seen:
            unique_files.append(f)
            seen.add(key)

    for f in unique_files:
        if not f.exists() or not f.is_file():
            continue

        try:
            txt = f.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        # Python helper: OC = "..." 또는 LAW_OC = "..."
        if f.suffix.lower() == ".py":
            for var in ["OC", "LAW_OC"]:
                m = re.search(
                    rf'(?m)^\s*{var}\s*=\s*["\']([^"\']+)["\']',
                    txt,
                )
                if m:
                    oc = clean(m.group(1))
                    if oc:
                        return oc, f"python_file:{f.name}:{var}"

            # 상수 직접 파싱이 안 되는 특수 helper는 import fallback
            try:
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    "_oneclick_notice_helper",
                    f,
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                for var in ["OC", "LAW_OC"]:
                    oc = clean(
                        getattr(mod, var, "")
                    )
                    if oc:
                        return oc, f"python_import:{f.name}:{var}"
            except Exception:
                pass

        # .env / config-like text: LAW_OC=... or OC=...
        else:
            for var in ["LAW_OC", "OC"]:
                m = re.search(
                    rf'(?m)^\s*{var}\s*=\s*["\']?([^\s"\']+)["\']?\s*$',
                    txt,
                )
                if m:
                    oc = clean(m.group(1))
                    if oc:
                        return oc, f"config_file:{f.name}:{var}"

    return "", "NOT_FOUND"


def get_law_oc():
    """Backward-compatible wrapper returning only the API credential."""
    oc, _ = get_law_oc_with_source()
    return oc

def check_official_current_notice():
    """
    국가법령정보 공동활용 행정규칙 목록 API에서 현행 고시를 검색.
    실패 시 예외를 바깥으로 던지지 않고 상태표로 반환.
    """
    if not CHECK_OFFICIAL_UPDATE:
        return (
            pd.DataFrame([{
                "status": "OFFICIAL_CHECK_DISABLED",
                "message": "환경설정에 의해 공식 최신 고시 확인 생략",
            }]),
            {},
        )

    oc, oc_source = get_law_oc_with_source()
    if not oc:
        return (
            pd.DataFrame([{
                "status": "SKIPPED_NO_LAW_OC",
                "auth_source": oc_source,
                "message": (
                    "공식 API 인증값을 찾지 못해 최신 고시 확인 생략. "
                    "LAW_OC 환경변수 또는 LAW_OC_FILE/oneclick helper를 확인하세요."
                ),
            }]),
            {},
        )

    try:
        import requests

        url = "https://www.law.go.kr/DRF/lawSearch.do"
        r = requests.get(
            url,
            params={
                "OC": oc,
                "target": "admrul",
                "type": "JSON",
                "nw": 1,
                "search": 1,
                "query": REGULATION_TITLE,
                "display": 100,
            },
            timeout=45,
        )
        r.raise_for_status()
        data = r.json()

        candidates = []
        for d in _recursive_dicts(data):
            title = clean(
                d.get("행정규칙명")
                or d.get("행정규칙제목")
                or ""
            )
            serial = clean(
                d.get("행정규칙일련번호")
                or d.get("행정규칙 일련번호")
                or ""
            )
            if not title or not serial:
                continue

            if (
                title == REGULATION_TITLE
                or (
                    "인체급성유해성물질" in title
                    and "생태유해성물질" in title
                )
            ):
                label = _extract_notice_label(d)
                candidates.append({
                    "label": label,
                    "serial": serial,
                    "title": title,
                    "issue_date": clean(d.get("발령일자")),
                    "effective_date": clean(d.get("시행일자")),
                    "revision_type": clean(d.get("제개정구분명")),
                    "ministry": clean(d.get("소관부처명")),
                })

        if not candidates:
            return (
                pd.DataFrame([{
                    "status": "OFFICIAL_QUERY_NO_MATCH",
                    "auth_source": oc_source,
                    "message": "현행 행정규칙 검색 결과에서 대상 고시를 찾지 못함",
                }]),
                {},
            )

        candidates = sorted(
            candidates,
            key=lambda x: notice_key(x["label"]),
        )
        best = candidates[-1]

        return (
            pd.DataFrame([{
                "status": "OFFICIAL_NOTICE_FOUND",
                "auth_source": oc_source,
                **best,
                "message": "국가법령정보센터 현행 행정규칙 검색 결과",
            }]),
            best,
        )

    except Exception as e:
        # requests 예외에는 query string 전체가 포함될 수 있으므로
        # API credential을 결과/로그에 절대 남기지 않는다.
        err_text = f"{type(e).__name__}: {e}"
        if oc:
            err_text = err_text.replace(oc, "***REDACTED***")
        err_text = re.sub(
            r"([?&]OC=)[^&\s]+",
            r"\1***REDACTED***",
            err_text,
            flags=re.I,
        )

        return (
            pd.DataFrame([{
                "status": "OFFICIAL_CHECK_FAILED_CONTINUE_OFFLINE",
                "auth_source": oc_source,
                "message": err_text,
            }]),
            {},
        )


def download_official_notice_candidate(info: dict):
    """
    새 고시가 로컬 master보다 최신일 때 원문/첨부를 자동 다운로드.
    여기까지는 자동. master 승인 반영은 별도.
    """
    if not info or not clean(info.get("serial")):
        return [], pd.DataFrame()

    oc = get_law_oc()
    if not oc:
        return [], pd.DataFrame()

    import requests

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    label = clean(info["label"])
    serial = clean(info["serial"])

    saved = []
    manifest = []

    try:
        r = requests.get(
            "https://www.law.go.kr/DRF/lawService.do",
            params={
                "OC": oc,
                "target": "admrul",
                "ID": serial,
                "type": "JSON",
            },
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()

        json_path = RAW_DIR / f"admrul_{label}_raw.json"
        import json as _json
        json_path.write_text(
            _json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        saved.append(json_path)

        manifest.append({
            "notice": label,
            "serial": serial,
            "file_type": "JSON",
            "local_file": str(json_path),
            "status": "DOWNLOADED",
        })

        svc = data.get("AdmRulService", {}) or {}
        attach = svc.get("첨부파일", {}) or {}
        links = attach.get("첨부파일링크", []) or []
        names = attach.get("첨부파일명", []) or []

        if isinstance(links, str):
            links = [links]
        if isinstance(names, str):
            names = [names]

        for i, link in enumerate(links):
            name = (
                names[i]
                if i < len(names)
                else f"attachment_{i+1}"
            )
            safe = re.sub(r'[\\/:*?"<>|]', "_", str(name))
            fname = f"{label}_{safe}"
            fpath = RAW_DIR / fname

            url = (
                str(link)
                if str(link).startswith("http")
                else "https://www.law.go.kr" + str(link)
            )

            fr = requests.get(url, timeout=120)
            fr.raise_for_status()
            fpath.write_bytes(fr.content)

            saved.append(fpath)
            manifest.append({
                "notice": label,
                "serial": serial,
                "file_type": fpath.suffix.lower(),
                "local_file": str(fpath),
                "status": "DOWNLOADED",
            })

    except Exception as e:
        manifest.append({
            "notice": label,
            "serial": serial,
            "file_type": "",
            "local_file": "",
            "status": f"DOWNLOAD_ERROR: {type(e).__name__}: {e}",
        })

    return saved, pd.DataFrame(manifest)


def choose_authoritative_new_excel(files):
    excels = [
        Path(p) for p in files
        if Path(p).suffix.lower() in {".xlsx", ".xls"}
    ]
    if not excels:
        return None

    full = [
        p for p in excels
        if any(
            k in p.name.lower()
            for k in ["목록(전체)", "전체", "full"]
        )
        and "개정사항" not in p.name
    ]
    if full:
        return full[0]

    non_change = [
        p for p in excels
        if "개정사항" not in p.name
    ]
    return non_change[0] if non_change else excels[0]


def parse_new_notice_candidate(
    notice_label: str,
    downloaded_files,
):
    """
    기존 Stage2 V2의 generic Excel parser를 재사용.
    새 고시가 기존과 같은 표 형식이면 자동 후보구조화 가능.
    형식이 바뀌면 PENDING_MANUAL_PARSER_REVIEW로 남긴다.
    """
    excel = choose_authoritative_new_excel(downloaded_files)
    if excel is None:
        return (
            pd.DataFrame(),
            "NO_SUPPORTED_EXCEL_ATTACHMENT",
        )

    helper = SCRIPT_DIR / "stage2_build_regulatory_master_V2_FIXED.py"
    if not helper.exists():
        return (
            pd.DataFrame(),
            "STAGE2_HELPER_NOT_FOUND",
        )

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_stage2_dynamic_helper",
            helper,
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        rows = mod.parse_excel(
            excel,
            notice_label,
            audit_only=False,
        )
        cand = pd.DataFrame(rows)

        if not len(cand):
            return cand, "PARSER_RETURNED_ZERO_ROWS"

        cand = mod.add_ids(cand)
        return cand, "PARSED_CANDIDATE_RULES"

    except Exception as e:
        return (
            pd.DataFrame(),
            f"PARSER_ERROR: {type(e).__name__}: {e}",
        )


def maybe_apply_approved_update(
    master: pd.DataFrame,
    candidate: pd.DataFrame,
    notice_label: str,
    master_path: Path,
):
    """
    자동 발견 ≠ 자동 법적 승인.
    APPROVED_NOTICE와 일치할 때에만 candidate를 승인 master에 반영.
    """
    if not len(candidate):
        return master, "NO_CANDIDATE_TO_APPLY"

    if clean(APPROVED_NOTICE) != clean(notice_label):
        return master, "PENDING_EXPERT_APPROVAL"

    # 기존 동일 notice가 있으면 교체하여 재승인 가능
    out = master[
        ~master["notice"].astype(str).eq(str(notice_label))
    ].copy()

    # 컬럼 정렬
    all_cols = list(dict.fromkeys(
        list(out.columns) + list(candidate.columns)
    ))
    out = out.reindex(columns=all_cols)
    candidate = candidate.reindex(columns=all_cols)

    out = pd.concat(
        [out, candidate],
        ignore_index=True,
    )

    # 동일 source row 중복 제거
    dedupe_cols = [
        c for c in [
            "notice", "designation_id", "sub_no",
            "substance_name_ko", "direct_cas",
            "acute_threshold_raw", "chronic_threshold_raw",
            "eco_threshold_raw", "source_text",
        ]
        if c in out.columns
    ]
    if dedupe_cols:
        out = out.drop_duplicates(
            dedupe_cols,
            keep="last",
        ).reset_index(drop=True)

    # helper가 있으면 rule_id/heterogeneity를 전체 재생성
    helper = SCRIPT_DIR / "stage2_build_regulatory_master_V2_FIXED.py"
    if helper.exists():
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "_stage2_apply_helper",
                helper,
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            out = mod.add_ids(out)
        except Exception:
            pass

    master_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(
        master_path,
        index=False,
        encoding="utf-8-sig",
    )

    return out, "APPROVED_AND_APPLIED_TO_MASTER"


def perform_official_update_check(
    master: pd.DataFrame,
    master_path: Path,
):
    """
    공식 최신 고시 확인과 로컬 master 비교.

    중요
    ----
    공식 조회를 수행하지 못한 상태에서
    NO_NEW_OFFICIAL_NOTICE를 기록하지 않는다.
    """
    local_prev, local_latest = latest_two_notices(master)

    official_status, info = check_official_current_notice()

    pending = pd.DataFrame()
    manifest = pd.DataFrame()

    status_values = (
        official_status["status"].astype(str).tolist()
        if (
            official_status is not None
            and len(official_status)
            and "status" in official_status.columns
        )
        else []
    )
    primary_status = (
        status_values[0]
        if status_values
        else "UNKNOWN_OFFICIAL_STATUS"
    )

    # 공식 조회 자체가 성공하지 않은 경우에는
    # '신규 고시 없음'이라는 결론을 내리지 않는다.
    if primary_status == "OFFICIAL_CHECK_DISABLED":
        approval_status = "OFFICIAL_CHECK_NOT_PERFORMED"
    elif primary_status == "SKIPPED_NO_LAW_OC":
        approval_status = "OFFICIAL_CHECK_NOT_PERFORMED"
    elif primary_status == "OFFICIAL_CHECK_FAILED_CONTINUE_OFFLINE":
        approval_status = "OFFICIAL_CHECK_FAILED"
    elif primary_status == "OFFICIAL_QUERY_NO_MATCH":
        approval_status = "OFFICIAL_QUERY_INCONCLUSIVE"
    elif primary_status == "OFFICIAL_NOTICE_FOUND":
        approval_status = "NO_NEW_OFFICIAL_NOTICE"
    else:
        approval_status = "OFFICIAL_STATUS_UNRESOLVED"

    if (
        primary_status == "OFFICIAL_NOTICE_FOUND"
        and info
        and clean(info.get("label"))
        and notice_key(info["label"]) > notice_key(local_latest)
    ):
        files, manifest = download_official_notice_candidate(info)
        pending, parse_status = parse_new_notice_candidate(
            info["label"],
            files,
        )

        master2, approval_status = maybe_apply_approved_update(
            master,
            pending,
            info["label"],
            master_path,
        )

        extra = pd.DataFrame([{
            "status": "NEW_OFFICIAL_NOTICE_DETECTED",
            "local_latest_notice": local_latest,
            "official_latest_notice": info["label"],
            "candidate_parse_status": parse_status,
            "approval_status": approval_status,
            "instruction": (
                "전문가가 신규 고시 후보를 검토한 후 "
                f'PowerShell에서 $env:APPROVED_NOTICE="{info["label"]}" '
                "설정 후 재실행하면 승인 master에 반영"
            ),
        }])

        official_status = pd.concat(
            [official_status, extra],
            ignore_index=True,
            sort=False,
        )
        return master2, official_status, pending, manifest

    official_status = official_status.copy()
    official_status["local_latest_notice"] = local_latest
    official_status["approval_status"] = approval_status

    if (
        primary_status == "OFFICIAL_NOTICE_FOUND"
        and info
        and clean(info.get("label"))
    ):
        official_status["official_latest_notice"] = clean(
            info.get("label")
        )

    return master, official_status, pending, manifest


# ============================================================
# 4-2. 회사 Chemical Inventory
# ============================================================
INVENTORY_COLUMNS = [
    "inventory_id",
    "product_name",
    "chemical_name",
    "cas",
    "concentration_pct",
    "use_description",
    "annual_quantity_kg",
    "authorization_status",
    "as_of_date",
    "smiles",
    "facility",
    "supplier",
]


def normalize_cas_value(x):
    m = CAS_RE.search(clean(x))
    return m.group(1) if m else ""


def load_company_inventory():
    candidates = []

    if COMPANY_INVENTORY_FILE:
        candidates.append(Path(COMPANY_INVENTORY_FILE))

    candidates += [
        SCRIPT_DIR / "company_inventory.xlsx",
        SCRIPT_DIR / "company_inventory.csv",
    ]

    for p in candidates:
        if not p.exists():
            continue

        if p.suffix.lower() in {".xlsx", ".xls"}:
            inv = pd.read_excel(p)
        else:
            inv = pd.read_csv(p)

        for c in INVENTORY_COLUMNS:
            if c not in inv.columns:
                inv[c] = ""

        inv = inv[INVENTORY_COLUMNS].copy()
        inv["cas"] = inv["cas"].map(normalize_cas_value)
        inv["concentration_pct"] = pd.to_numeric(
            inv["concentration_pct"],
            errors="coerce",
        )
        inv["inventory_source"] = str(p)
        return inv, "REAL_OR_USER_SUPPLIED_INVENTORY"

    return pd.DataFrame(columns=INVENTORY_COLUMNS), "NO_COMPANY_INVENTORY"


def build_rule_cas_index(
    master: pd.DataFrame,
    notice: str,
):
    g = substantive_rule_rows_for_notice(
        master,
        notice,
    )
    rows = []

    for _, r in g.iterrows():
        direct = direct_cas_list_from_row(r)
        thr = min_threshold(r)
        exc = set(splitsemi(r.get("exception_cas")))

        for cas in direct:
            rows.append({
                "notice": notice,
                "designation_id": clean(r.get("designation_id")),
                "rule_id": clean(r.get("rule_id")),
                "substance_name_ko": clean(r.get("substance_name_ko")),
                "cas": cas,
                "min_threshold_pct": thr,
                "exception_cas": ";".join(exc),
                "source_text": clean(r.get("source_text")),
            })

    return pd.DataFrame(rows)


def build_demo_company_inventory(
    master: pd.DataFrame,
    previous_notice: str,
    latest_notice: str,
    update_diff: pd.DataFrame,
    max_designations: int = 15,
):
    """
    실제 기업자료가 없을 때 논문/프로토타입 작동 확인용 synthetic inventory.
    실제 기업 오류율로 해석하면 안 됨.
    """
    changed = update_diff[
        update_diff["update_change_type"].ne("UNCHANGED")
    ].copy()

    latest_idx = build_rule_cas_index(
        master,
        latest_notice,
    )
    previous_idx = build_rule_cas_index(
        master,
        previous_notice,
    )

    rows = []
    counter = 1

    # 최신 changed direct-CAS rules: below/above 2개씩
    dids = changed["designation_id"].astype(str).tolist()

    used_dids = set()

    for did in dids:
        g = latest_idx[
            latest_idx["designation_id"].astype(str).eq(did)
        ]
        if not len(g):
            continue

        r = g.iloc[0]
        thr = r["min_threshold_pct"]
        if not np.isfinite(thr):
            continue

        for label, factor in [
            ("BELOW", 0.8),
            ("ABOVE", 1.2),
        ]:
            conc = min(
                100.0,
                max(0.000001, float(thr) * factor)
            )
            rows.append({
                "inventory_id": f"DEMO_{counter:03d}",
                "product_name": f"Demo product {counter:03d}",
                "chemical_name": r["substance_name_ko"],
                "cas": r["cas"],
                "concentration_pct": conc,
                "smiles": "",
                "facility": "DEMO_SITE",
                "supplier": "SYNTHETIC",
                "inventory_source": "SYNTHETIC_DEMO_NOT_REAL_COMPANY_DATA",
                "demo_scenario": f"CHANGED_RULE_{label}",
            })
            counter += 1

        used_dids.add(did)
        if len(used_dids) >= max_designations:
            break

    # 삭제된 규칙 direct CAS가 있으면 일부 추가
    removed = changed[
        changed["update_change_type"].eq("REMOVED")
    ]["designation_id"].astype(str)

    for did in removed.head(5):
        g = previous_idx[
            previous_idx["designation_id"].astype(str).eq(did)
        ]
        if not len(g):
            continue

        r = g.iloc[0]
        thr = r["min_threshold_pct"]
        conc = (
            min(100.0, float(thr) * 1.2)
            if np.isfinite(thr)
            else 1.0
        )
        rows.append({
            "inventory_id": f"DEMO_{counter:03d}",
            "product_name": f"Demo product {counter:03d}",
            "chemical_name": r["substance_name_ko"],
            "cas": r["cas"],
            "concentration_pct": conc,
            "smiles": "",
            "facility": "DEMO_SITE",
            "supplier": "SYNTHETIC",
            "inventory_source": "SYNTHETIC_DEMO_NOT_REAL_COMPANY_DATA",
            "demo_scenario": "REMOVED_RULE_CONTROL",
        })
        counter += 1

    return pd.DataFrame(rows)


def load_identity_relations():
    """
    선택적 검증 relation table.
    필수 열:
      inventory_cas, designation_id, relation_type, validated
    """
    candidates = []
    if IDENTITY_RELATION_FILE:
        candidates.append(Path(IDENTITY_RELATION_FILE))
    candidates += [
        SCRIPT_DIR / "company_identity_relations.xlsx",
        SCRIPT_DIR / "company_identity_relations.csv",
    ]

    for p in candidates:
        if not p.exists():
            continue

        if p.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(p)
        else:
            df = pd.read_csv(p)

        for c in [
            "inventory_cas",
            "designation_id",
            "relation_type",
            "validated",
            "evidence_source",
        ]:
            if c not in df.columns:
                df[c] = ""

        df["inventory_cas"] = df["inventory_cas"].map(
            normalize_cas_value
        )
        df["validated_bool"] = (
            df["validated"].astype(str).str.strip().str.lower()
            .isin({"1", "true", "yes", "y", "validated"})
        )
        return df

    return pd.DataFrame(columns=[
        "inventory_cas",
        "designation_id",
        "relation_type",
        "validated",
        "evidence_source",
        "validated_bool",
    ])


def evaluate_inventory_for_notice(
    inventory: pd.DataFrame,
    master: pd.DataFrame,
    notice: str,
    relations: pd.DataFrame,
):
    index = build_rule_cas_index(
        master,
        notice,
    )

    rel = relations[
        relations.get(
            "validated_bool",
            pd.Series(dtype=bool)
        ).eq(True)
    ].copy()

    rows = []

    for _, inv in inventory.iterrows():
        cas = normalize_cas_value(inv.get("cas"))
        conc = pd.to_numeric(
            pd.Series([inv.get("concentration_pct")]),
            errors="coerce",
        ).iloc[0]

        matches = (
            index[index["cas"].eq(cas)].copy()
            if cas and len(index)
            else index.head(0).copy()
        )

        # direct-CAS final screening
        if len(matches):
            for _, rule in matches.iterrows():
                exc = set(splitsemi(rule.get("exception_cas")))
                excluded = cas in exc

                thr = rule["min_threshold_pct"]
                if excluded:
                    applicable = False
                    status = "DIRECT_CAS_EXCLUDED"
                elif not np.isfinite(conc):
                    applicable = None
                    status = "REVIEW_MISSING_CONCENTRATION"
                elif not np.isfinite(thr):
                    applicable = None
                    status = "REVIEW_MISSING_THRESHOLD"
                elif float(conc) >= float(thr):
                    applicable = True
                    status = "DIRECT_CAS_THRESHOLD_TRIGGER"
                else:
                    applicable = False
                    status = "DIRECT_CAS_BELOW_THRESHOLD"

                rows.append({
                    **inv.to_dict(),
                    "assessment_notice": notice,
                    "designation_id": rule["designation_id"],
                    "matching_mode": "DIRECT_CAS",
                    "relation_type": "",
                    "rule_substance_name": rule["substance_name_ko"],
                    "rule_threshold_pct": thr,
                    "screening_status": status,
                    "screening_triggered": applicable,
                    "source_text": rule["source_text"],
                })

        # validated relation matching for salt/group/etc.
        rel_matches = (
            rel[rel["inventory_cas"].eq(cas)]
            if cas and len(rel)
            else rel.head(0)
        )

        for _, rr in rel_matches.iterrows():
            did = clean(rr["designation_id"])

            rg = substantive_rule_rows_for_notice(
                master,
                notice,
            )
            rg = rg[
                rg["designation_id"].astype(str).eq(did)
            ].copy()

            if not len(rg):
                continue

            summary = designation_summary(rg)
            thr = summary["min_threshold_pct"]

            if not np.isfinite(conc):
                applicable = None
                status = "REVIEW_MISSING_CONCENTRATION"
            elif not np.isfinite(thr):
                applicable = None
                status = "REVIEW_MISSING_THRESHOLD"
            elif float(conc) >= float(thr):
                applicable = True
                status = "VALIDATED_RELATION_THRESHOLD_TRIGGER"
            else:
                applicable = False
                status = "VALIDATED_RELATION_BELOW_THRESHOLD"

            rows.append({
                **inv.to_dict(),
                "assessment_notice": notice,
                "designation_id": did,
                "matching_mode": "VALIDATED_IDENTITY_RELATION",
                "relation_type": clean(rr["relation_type"]),
                "rule_substance_name": summary["substance_names"],
                "rule_threshold_pct": thr,
                "screening_status": status,
                "screening_triggered": applicable,
                "source_text": summary["source_text"],
            })

        if not len(matches) and not len(rel_matches):
            rows.append({
                **inv.to_dict(),
                "assessment_notice": notice,
                "designation_id": "",
                "matching_mode": "NONE",
                "relation_type": "",
                "rule_substance_name": "",
                "rule_threshold_pct": np.nan,
                "screening_status": "NO_DIRECT_OR_VALIDATED_RELATION_MATCH",
                "screening_triggered": False,
                "source_text": "",
            })

    return pd.DataFrame(rows)


def build_company_update_impact(
    inventory: pd.DataFrame,
    previous_eval: pd.DataFrame,
    latest_eval: pd.DataFrame,
    update_diff: pd.DataFrame,
):
    changed = update_diff[
        update_diff["update_change_type"].ne("UNCHANGED")
    ].copy()

    changed_ids = set(
        changed["designation_id"].astype(str)
    )

    old = previous_eval[
        previous_eval["designation_id"]
        .astype(str).isin(changed_ids)
    ].copy()

    new = latest_eval[
        latest_eval["designation_id"]
        .astype(str).isin(changed_ids)
    ].copy()

    keys = set()

    for df in [old, new]:
        for _, r in df.iterrows():
            did = clean(r.get("designation_id"))
            if did:
                keys.add((
                    clean(r.get("inventory_id")),
                    did,
                ))

    rows = []

    def triggered_for(df, iid, did):
        g = df[
            df["inventory_id"].astype(str).eq(str(iid))
            & df["designation_id"].astype(str).eq(str(did))
        ]
        vals = [
            bool(x)
            for x in g["screening_triggered"].tolist()
            if isinstance(x, (bool, np.bool_))
        ]
        return any(vals) if vals else False

    for iid, did in sorted(keys):
        old_t = triggered_for(old, iid, did)
        new_t = triggered_for(new, iid, did)

        if (not old_t) and new_t:
            impact = "NEWLY_TRIGGERED_BY_UPDATE"
        elif old_t and (not new_t):
            impact = "NO_LONGER_TRIGGERED_AFTER_UPDATE"
        elif old_t and new_t:
            impact = "STILL_TRIGGERED"
        else:
            impact = "NOT_TRIGGERED_BY_DIRECT_OR_VALIDATED_RULE"

        invrow = inventory[
            inventory["inventory_id"].astype(str).eq(str(iid))
        ]
        invd = invrow.iloc[0].to_dict() if len(invrow) else {}

        ch = changed[
            changed["designation_id"].astype(str).eq(str(did))
        ]
        chd = ch.iloc[0].to_dict() if len(ch) else {}

        rows.append({
            **invd,
            "designation_id": did,
            "regulatory_update_type": chd.get(
                "update_change_type", ""
            ),
            "old_triggered": old_t,
            "new_triggered": new_t,
            "company_impact": impact,
            "human_review_required": chd.get(
                "human_review_required", False
            ),
            "changed_fields": chd.get("changed_fields", ""),
            "old_rule_text": chd.get("old_rule_text", ""),
            "new_rule_text": chd.get("new_rule_text", ""),
        })

    return pd.DataFrame(rows)


def build_extended_scope_review_queue(
    update_diff: pd.DataFrame,
    latest_notice: str,
):
    q = update_diff[
        update_diff["update_change_type"].ne("UNCHANGED")
        & (
            update_diff["old_has_extended_scope"].fillna(False)
            | update_diff["new_has_extended_scope"].fillna(False)
        )
    ].copy()

    if not len(q):
        return q

    q["review_reason"] = (
        "UPDATED_RULE_NOT_FULLY_RESOLVED_BY_DIRECT_CAS"
    )
    q["recommended_resolution"] = (
        "Local LLM scope interpretation + validated member CAS relation "
        "/ RDKit candidate structure analysis / expert confirmation"
    )
    q["review_status"] = "PENDING_HUMAN_REVIEW"
    q["latest_notice"] = latest_notice
    return q



# ============================================================
# 4-3. Optional RDKit parent-structure candidate layer
# ============================================================
def load_regulatory_structure_reference():
    candidates = []
    if REGULATORY_STRUCTURE_FILE:
        candidates.append(Path(REGULATORY_STRUCTURE_FILE))

    candidates += [
        SCRIPT_DIR / "regulatory_structure_reference.xlsx",
        SCRIPT_DIR / "regulatory_structure_reference.csv",
    ]

    for p in candidates:
        if not p.exists():
            continue

        if p.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(p)
        else:
            df = pd.read_csv(p)

        for c in [
            "designation_id",
            "reference_smiles",
            "relation_type",
            "validated",
            "evidence_source",
        ]:
            if c not in df.columns:
                df[c] = ""

        df["validated_bool"] = (
            df["validated"].astype(str).str.strip().str.lower()
            .isin({"1", "true", "yes", "y", "validated"})
        )
        return df

    return pd.DataFrame(columns=[
        "designation_id",
        "reference_smiles",
        "relation_type",
        "validated",
        "evidence_source",
        "validated_bool",
    ])


def rdkit_parent_key(smiles: str):
    """
    Salt/counter-ion이 포함된 SMILES를 parent-like canonical key로 표준화.
    결과는 법적 identity가 아니라 cheminformatics candidate 생성용이다.
    """
    smi = clean(smiles)
    if not smi:
        return ""

    try:
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""

        # 가장 대표적인 parent fragment 선택
        parent = rdMolStandardize.FragmentParent(mol)

        try:
            uncharger = rdMolStandardize.Uncharger()
            parent = uncharger.uncharge(parent)
        except Exception:
            pass

        return Chem.MolToSmiles(
            parent,
            canonical=True,
            isomericSmiles=True,
        )
    except Exception:
        return ""


def build_rdkit_structure_candidates(
    inventory: pd.DataFrame,
    structure_ref: pd.DataFrame,
    update_diff: pd.DataFrame,
):
    """
    회사 SMILES와 규제 기준 SMILES의 parent key가 같으면 후보관계 생성.

    중요:
    - RDKIT_PARENT_MATCH_CANDIDATE는 최종 법적 match가 아니다.
    - 전문가 확인 후 validated identity relation으로 승격해야
      deterministic screening에 사용할 수 있다.
    """
    if not len(inventory) or not len(structure_ref):
        return pd.DataFrame()

    refs = structure_ref[
        structure_ref["validated_bool"].eq(True)
        & structure_ref["reference_smiles"].fillna("").astype(str).str.strip().ne("")
    ].copy()

    if not len(refs):
        return pd.DataFrame()

    refs["_parent_key"] = refs["reference_smiles"].map(
        rdkit_parent_key
    )
    refs = refs[refs["_parent_key"].ne("")].copy()

    if not len(refs):
        return pd.DataFrame()

    changed_ids = set(
        update_diff.loc[
            update_diff["update_change_type"].ne("UNCHANGED"),
            "designation_id",
        ].astype(str)
    ) if len(update_diff) else set()

    rows = []

    for _, inv in inventory.iterrows():
        inv_key = rdkit_parent_key(inv.get("smiles"))
        if not inv_key:
            continue

        hits = refs[
            refs["_parent_key"].eq(inv_key)
        ]

        for _, rr in hits.iterrows():
            did = clean(rr["designation_id"])
            rows.append({
                "inventory_id": clean(inv.get("inventory_id")),
                "product_name": clean(inv.get("product_name")),
                "chemical_name": clean(inv.get("chemical_name")),
                "inventory_cas": normalize_cas_value(inv.get("cas")),
                "inventory_smiles": clean(inv.get("smiles")),
                "inventory_parent_key": inv_key,
                "designation_id": did,
                "reference_smiles": clean(rr["reference_smiles"]),
                "reference_parent_key": clean(rr["_parent_key"]),
                "reference_relation_type": clean(rr["relation_type"]),
                "regulatory_rule_changed_in_latest_update": did in changed_ids,
                "candidate_status": (
                    "RDKIT_PARENT_STRUCTURE_CANDIDATE_NOT_LEGAL_MATCH"
                ),
                "required_action": (
                    "Expert confirms salt/group/structure membership; "
                    "if confirmed, add to company_identity_relations as validated."
                ),
                "evidence_source": clean(rr["evidence_source"]),
            })

    return pd.DataFrame(rows)


def regulatory_structure_template():
    return pd.DataFrame([
        {
            "designation_id": "regulated designation id",
            "reference_smiles": "validated parent/reference SMILES",
            "relation_type": "SALT_PARENT / GROUP_REFERENCE / STRUCTURE_REFERENCE",
            "validated": 0,
            "evidence_source": "",
        }
    ])


def enterprise_inventory_template():
    return pd.DataFrame([
        {
            "inventory_id": "INV-001",
            "product_name": "Example product",
            "chemical_name": "Example chemical",
            "cas": "123-45-6",
            "concentration_pct": 1.0,
            "smiles": "",
            "facility": "Plant A",
            "supplier": "Supplier",
        }
    ])


def identity_relation_template():
    return pd.DataFrame([
        {
            "inventory_cas": "member CAS",
            "designation_id": "regulated designation id",
            "relation_type": "SALT_OF / GROUP_MEMBER / STRUCTURE_MEMBER",
            "validated": 0,
            "evidence_source": "",
        }
    ])


# ============================================================
# 5. Excel 저장
# ============================================================
def autosize_sheet(ws, max_width=60):
    for col_cells in ws.columns:
        length = 0
        col_letter = col_cells[0].column_letter

        for cell in col_cells:
            value = "" if cell.value is None else str(cell.value)
            length = max(length, len(value))

        ws.column_dimensions[col_letter].width = min(
            max(length + 2, 10),
            max_width
        )

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions


def save_enterprise_base_excel(
    system_overview,
    official_update_status,
    official_download_manifest,
    pending_update_rules,
    update_diff,
    inventory,
    current_screening,
    company_update_impact,
    extended_review_queue,
    identity_summary,
    identity_detail,
    rq2,
    casebook,
    tierc_queue,
    inventory_template_df,
    relation_template_df,
    structure_candidate_queue,
    structure_template_df,
    historical_case_study,
    metadata,
):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(
        OUT_XLSX,
        engine="openpyxl",
    ) as writer:

        sheets = [
            ("System_overview", system_overview),
            ("Official_update_status", official_update_status),
            ("Official_download_manifest", official_download_manifest),
            ("Pending_update_rules", pending_update_rules),
            ("Regulatory_update_diff", update_diff),
            ("Company_inventory", inventory),
            ("Company_current_screening", current_screening),
            ("Company_update_impact", company_update_impact),
            ("Human_review_queue", extended_review_queue),
            ("Identity_scope_summary", identity_summary),
            ("Identity_scope_detail", identity_detail),
            ("CAS_vs_concentration_test", rq2),
            ("TierC_casebook", casebook),
            ("TierC_validation_queue", tierc_queue),
            ("Company_inventory_template", inventory_template_df),
            ("Identity_relation_template", relation_template_df),
            ("Structure_candidate_queue", structure_candidate_queue),
            ("Reg_structure_template", structure_template_df),
            ("Historical_case_study", historical_case_study),
            ("Analysis_metadata", metadata),
        ]

        for name, df in sheets:
            if df is None:
                df = pd.DataFrame()
            df.to_excel(
                writer,
                sheet_name=name[:31],
                index=False,
            )

        for ws in writer.book.worksheets:
            autosize_sheet(ws)


def run_deterministic_analysis():
    master_path = resolve_master_file()

    if not master_path.exists():
        raise FileNotFoundError(
            f"입력 마스터 파일이 없습니다:\n{master_path}\n"
            "먼저 규제 마스터 builder를 실행하세요."
        )

    master = pd.read_csv(master_path)

    # --------------------------------------------------------
    # A. 공식 사이트 업데이트 확인
    # --------------------------------------------------------
    (
        master,
        official_update_status,
        pending_update_rules,
        official_download_manifest,
    ) = perform_official_update_check(
        master,
        master_path,
    )

    previous_notice, latest_notice = latest_two_notices(
        master
    )

    # --------------------------------------------------------
    # B. 연구의 핵심: 직전 규제 -> 최신 규제 update diff
    # --------------------------------------------------------
    if previous_notice:
        update_diff = build_regulatory_update_diff(
            master,
            previous_notice,
            latest_notice,
        )
    else:
        update_diff = pd.DataFrame()

    # --------------------------------------------------------
    # C. 최신 규제의 identity scope
    # --------------------------------------------------------
    registry, challenge = build_identity_registry(
        master,
        latest_notice,
    )

    identity_summary = build_identity_summary(
        challenge
    )
    rq2 = build_rq2_summary(
        registry
    )
    casebook = build_casebook(
        challenge
    )
    tierc_queue = build_validation_queue(
        challenge
    )

    # --------------------------------------------------------
    # D. 회사 inventory
    # --------------------------------------------------------
    inventory, inventory_mode = load_company_inventory()

    if (
        not len(inventory)
        and USE_DEMO_INVENTORY_IF_MISSING
        and previous_notice
        and len(update_diff)
    ):
        inventory = build_demo_company_inventory(
            master,
            previous_notice,
            latest_notice,
            update_diff,
        )
        inventory_mode = "SYNTHETIC_DEMO_INVENTORY"

    relations = load_identity_relations()
    structure_ref = load_regulatory_structure_reference()

    structure_candidate_queue = build_rdkit_structure_candidates(
        inventory,
        structure_ref,
        update_diff,
    )

    if len(inventory):
        current_screening = evaluate_inventory_for_notice(
            inventory,
            master,
            latest_notice,
            relations,
        )

        previous_screening = (
            evaluate_inventory_for_notice(
                inventory,
                master,
                previous_notice,
                relations,
            )
            if previous_notice
            else pd.DataFrame()
        )

        company_update_impact = (
            build_company_update_impact(
                inventory,
                previous_screening,
                current_screening,
                update_diff,
            )
            if previous_notice and len(update_diff)
            else pd.DataFrame()
        )
    else:
        current_screening = pd.DataFrame()
        company_update_impact = pd.DataFrame()

    extended_review_queue = (
        build_extended_scope_review_queue(
            update_diff,
            latest_notice,
        )
        if len(update_diff)
        else pd.DataFrame()
    )

    # --------------------------------------------------------
    # E. 과거 2025 개편 비교는 본 연구의 중심이 아니라 case study
    # --------------------------------------------------------
    historical_case_study = pd.DataFrame()

    notices = ordered_notices(master)
    if len(notices) >= 2:
        earliest = notices[0]
        # 기존 transition logic은 2025-2를 기준으로 작성돼 있으므로
        # 해당 자료가 있을 때만 과거 사례표를 유지.
        if earliest == "2025-2":
            try:
                hist, _ = build_transition(
                    master,
                    latest_notice,
                )
                historical_case_study = hist
            except Exception:
                historical_case_study = pd.DataFrame()

    # --------------------------------------------------------
    # F. 연구설계 / 기업 적용 개요
    # --------------------------------------------------------
    system_overview = pd.DataFrame([
        {
            "step": 1,
            "module": "Regulatory update detection",
            "role": (
                "국가법령정보센터에서 현행 행정규칙을 확인하고 "
                "로컬 승인 master보다 최신 고시인지 탐지"
            ),
            "automation_level": "AUTOMATED_CHECK",
        },
        {
            "step": 2,
            "module": "Candidate rule ingestion",
            "role": (
                "신규 고시 원문/첨부 다운로드 및 machine-readable "
                "rule 후보 구조화"
            ),
            "automation_level": "AUTOMATED_CANDIDATE_EXTRACTION",
        },
        {
            "step": 3,
            "module": "Human approval gate",
            "role": (
                "신규 규제 후보를 전문가가 검토한 뒤에만 "
                "승인 regulatory master에 반영"
            ),
            "automation_level": "HUMAN_IN_THE_LOOP",
        },
        {
            "step": 4,
            "module": "Local LLM semantic parser",
            "role": (
                "염류·혼합물·물질군·구조범위 등 자연어 "
                "regulatory scope 및 명시적 제외조건 해석"
            ),
            "automation_level": "LOCAL_AI_ASSISTED",
        },
        {
            "step": 5,
            "module": "Deterministic rule engine",
            "role": (
                "CAS, 함량기준, 검증된 relation을 이용해 "
                "재현 가능한 screening 수행"
            ),
            "automation_level": "DETERMINISTIC",
        },
        {
            "step": 6,
            "module": "Enterprise impact screening",
            "role": (
                "전체 inventory를 재검토하는 대신 변경된 규칙에 "
                "영향받는 회사 물질을 우선 선별"
            ),
            "automation_level": "AUTOMATED_TRIAGE",
        },
        {
            "step": 7,
            "module": "Expert review queue",
            "role": (
                "direct CAS로 해결할 수 없는 관계/구조 범위와 "
                "불확실한 사례만 담당자 검토로 이관"
            ),
            "automation_level": "TARGETED_HUMAN_REVIEW",
        },
    ])

    metadata = pd.DataFrame([
        {
            "item": "research_primary_question",
            "value": (
                "공식 화학물질 규제의 개정사항을 자동 감지·구조화하고, "
                "기업 chemical inventory와 연결하여 규제 적용 가능성이 "
                "새로 발생하거나 변경된 물질을 Local LLM + deterministic "
                "rules로 자동 triage할 수 있는가?"
            ),
        },
        {
            "item": "previous_approved_notice",
            "value": previous_notice,
        },
        {
            "item": "latest_approved_notice",
            "value": latest_notice,
        },
        {
            "item": "inventory_mode",
            "value": inventory_mode,
        },
        {
            "item": "inventory_n",
            "value": len(inventory),
        },
        {
            "item": "changed_designations_latest_update",
            "value": int(
                update_diff[
                    "update_change_type"
                ].ne("UNCHANGED").sum()
            ) if len(update_diff) else 0,
        },
        {
            "item": "extended_scope_update_review_n",
            "value": len(extended_review_queue),
        },
        {
            "item": "current_direct_identity_challenge_rows",
            "value": len(challenge),
        },
        {
            "item": "important_legal_safeguard",
            "value": (
                "새 고시는 자동 발견/다운로드 가능하지만 "
                "APPROVED_NOTICE 승인 전에는 승인 master에 반영하지 않음"
            ),
        },
        {
            "item": "AI_role",
            "value": (
                "자연어 regulatory scope와 explicit exclusion 해석 보조. "
                "최종 법적 판정은 수행하지 않음."
            ),
        },
        {
            "item": "cheminformatics_status",
            "value": (
                "검증된 규제 reference SMILES가 있을 경우 RDKit parent-structure "
                "candidate를 자동 생성함. candidate는 법적 match가 아니며 "
                "전문가 검증 후 validated relation으로 승격해야 함."
            ),
        },
        {
            "item": "historical_reform_role",
            "value": (
                "2025 개편 및 Factorial controlled benchmark는 "
                "단순 CAS 기반 screening의 한계를 입증하는 preliminary evidence로 사용."
            ),
        },
    ])

    save_enterprise_base_excel(
        system_overview=system_overview,
        official_update_status=official_update_status,
        official_download_manifest=official_download_manifest,
        pending_update_rules=pending_update_rules,
        update_diff=update_diff,
        inventory=inventory,
        current_screening=current_screening,
        company_update_impact=company_update_impact,
        extended_review_queue=extended_review_queue,
        identity_summary=identity_summary,
        identity_detail=challenge,
        rq2=rq2,
        casebook=casebook,
        tierc_queue=tierc_queue,
        inventory_template_df=enterprise_inventory_template(),
        relation_template_df=identity_relation_template(),
        structure_candidate_queue=structure_candidate_queue,
        structure_template_df=regulatory_structure_template(),
        historical_case_study=historical_case_study,
        metadata=metadata,
    )

    print("=" * 76)
    print("ENTERPRISE REGULATORY HYBRID - deterministic/update layer 완료")
    print("=" * 76)
    print(f"승인 규제 버전: {previous_notice} -> {latest_notice}")
    print(
        "변경 designation: "
        f"{int(update_diff['update_change_type'].ne('UNCHANGED').sum()) if len(update_diff) else 0:,}"
    )
    print(f"회사 inventory: {len(inventory):,} rows | {inventory_mode}")
    print(f"저장: {OUT_XLSX}")

    return {
        "master": master,
        "master_path": master_path,
        "previous_notice": previous_notice,
        "latest_notice": latest_notice,
        "update_diff": update_diff,
        "inventory": inventory,
        "current_screening": current_screening,
        "company_update_impact": company_update_impact,
        "extended_review_queue": extended_review_queue,
        "official_update_status": official_update_status,
        "update_diff": update_diff,
    }


# ============================================================
# 7. HYBRID LOCAL AI - enterprise semantic layer
#    Rule engine = CAS / concentration / deterministic decision
#    Local LLM   = semantic regulatory interpretation only
# ============================================================
import os
import re
import json
import time
import hashlib
from typing import Literal

AI_MODEL = os.getenv(
    "LOCAL_LLM_MODEL",
    "Qwen/Qwen3-1.7B"
).strip()

# 최종 논문 기본값:
# 2026-5 direct-CAS 부재 challenge 전부 + direct-CAS 비교군
AI_SAMPLE_SIZE = int(os.getenv("AI_SAMPLE_SIZE", "130"))
AI_RANDOM_SEED = int(os.getenv("AI_RANDOM_SEED", "20260828"))
AI_RUN = os.getenv("RUN_AI", "1").strip().lower() not in {
    "0", "false", "no", "off"
}

AI_PROMPT_VERSION = "enterprise_hybrid_local_scope_v5.0_2026-08-29"

SCOPE_TYPES = [
    "SINGLE_OR_DIRECT_CAS",
    "BROAD_SALT_SCOPE",
    "SALT_FAMILY",
    "MIXTURE_OR_REACTION",
    "GENERIC_GROUP_OR_DERIVATIVE",
    "STRUCTURAL_RANGE",
    "OTHER",
]


# ============================================================
# 7-1. AI evaluation reference
# ============================================================
def sha256_text(text: str) -> str:
    return hashlib.sha256(
        clean(text).encode("utf-8")
    ).hexdigest()


def semantic_reference_scope(
    text: str,
    has_direct_cas: bool,
) -> str:
    """
    Rule-based semantic reference.

    중요:
    '화합물', '유도체', '염'이라는 단어가 물질명에 들어갔다는 이유만으로
    direct CAS가 있는 특정 화합물을 generic group으로 과분류하지 않는다.

    명시적 broad-salt / mixture / structural-range 문구는 우선 인정.
    generic group / salt family는 direct CAS가 없는 경우에 한해 분류.
    """
    t = clean(text)

    if broad_salt_scope(t):
        return "BROAD_SALT_SCOPE"

    if mixture_scope(t):
        return "MIXTURE_OR_REACTION"

    if structural_range_scope(t):
        return "STRUCTURAL_RANGE"

    if not has_direct_cas:
        if generic_group_scope(t):
            return "GENERIC_GROUP_OR_DERIVATIVE"

        if salt_family_scope(t):
            return "SALT_FAMILY"

        return "OTHER"

    return "SINGLE_OR_DIRECT_CAS"


def semantic_reference_extended_identity(
    text: str,
    has_direct_cas: bool,
) -> bool:
    """
    현재 연구의 operational definition.

    1) direct CAS가 없으면 extended identity가 필요.
    2) 'A와 그 염류'처럼 direct CAS가 parent에 있더라도
       전체 salts 범위를 표현하려면 추가 관계정보가 필요.

    그 외 direct-CAS가 부여된 특정 mixture/UVCB/structural-range 항목은
    전문가 검증 전에는 자동으로 extended=True로 확대하지 않는다.
    """
    if not has_direct_cas:
        return True

    if broad_salt_scope(text):
        return True

    return False


def build_semantic_ai_reference(
    master: pd.DataFrame,
    notice: str = "",
) -> pd.DataFrame:

    if not notice:
        _, notice = latest_two_notices(master)

    g = master[
        master["notice"].eq(notice)
        & master["row_type"].astype(str).ne("continuation_detail")
    ].copy()

    rows = []

    for _, r in g.iterrows():
        active = active_new_thresholds(r)

        # 실제 활성 threshold가 있는 substantive rule만
        if not active:
            continue

        direct_stage2 = splitsemi(r.get("direct_cas"))
        cas_field_raw, recovered = recover_cas_field(
            r.get("source_text")
        )

        direct_final = (
            direct_stage2
            if direct_stage2
            else recovered
        )

        has_direct = bool(direct_final)
        text = clean(r.get("source_text"))
        exc = splitsemi(r.get("exception_cas"))

        scope = semantic_reference_scope(
            text,
            has_direct
        )

        has_exclusion = (
            bool(exc)
            or ("제외" in text)
            or ("다만" in text)
        )

        extended = semantic_reference_extended_identity(
            text,
            has_direct
        )

        rows.append({
            "rule_id": clean(r.get("rule_id")),
            "designation_id": clean(
                r.get("designation_id")
            ),
            "sub_no": clean(r.get("sub_no")),
            "substance_name_ko": clean(
                r.get("substance_name_ko")
            ),
            "source_text": text,

            # deterministic facts
            "direct_cas_final": ";".join(
                direct_final
            ),
            "has_direct_cas": has_direct,
            "acute_threshold_pct": parse_threshold(
                r.get("acute_threshold_raw")
            ),
            "chronic_threshold_pct": parse_threshold(
                r.get("chronic_threshold_raw")
            ),
            "eco_threshold_pct": parse_threshold(
                r.get("eco_threshold_raw")
            ),

            # semantic rule-based reference
            "ref_scope_type": scope,
            "ref_has_exclusion": has_exclusion,
            "ref_requires_extended_identity": extended,

            "_is_identity_challenge": not has_direct,
            "_is_semantically_complex": (
                scope != "SINGLE_OR_DIRECT_CAS"
                or has_exclusion
                or extended
            ),
            "_source_hash": sha256_text(text),
        })

    return pd.DataFrame(rows)


# ============================================================
# 7-2. Stratified sample
# ============================================================
def select_semantic_ai_sample(
    ref: pd.DataFrame,
    n: int,
) -> pd.DataFrame:
    """
    논문용 기본 표본:
    - direct-CAS 부재 identity challenge 전부 우선 포함
    - 나머지는 direct-CAS 규칙에서 비교군 추출
      · semantic complex direct 약 절반
      · ordinary direct 약 절반

    기본 n=130이면 현재 데이터에서는 대략:
      challenge 87 rule rows + direct controls 약 43 rows
    """
    if len(ref) <= n:
        x = ref.copy()
        x["sample_reason"] = "ALL_RULES"
        return x.reset_index(drop=True)

    rng = np.random.default_rng(AI_RANDOM_SEED)

    challenge = ref[
        ref["_is_identity_challenge"]
    ].copy()

    direct = ref[
        ~ref["_is_identity_challenge"]
    ].copy()

    # n이 challenge보다 작을 때도 scope별 층화
    if n < len(challenge):
        parts = []

        groups = list(
            challenge.groupby("ref_scope_type")
        )

        quota = max(
            1,
            n // max(len(groups), 1)
        )

        used = set()

        for scope, g in groups:
            take_n = min(
                quota,
                len(g),
                n - len(used)
            )

            if take_n <= 0:
                continue

            idx = rng.choice(
                g.index.to_numpy(),
                size=take_n,
                replace=False
            )

            x = g.loc[idx].copy()
            x["sample_reason"] = (
                "STRATIFIED_IDENTITY_"
                + str(scope)
            )
            parts.append(x)
            used.update(idx.tolist())

        out = (
            pd.concat(parts, ignore_index=False)
            if parts
            else challenge.head(0)
        )

        if len(out) < n:
            pool = challenge.loc[
                ~challenge.index.isin(used)
            ]

            need = min(
                n - len(out),
                len(pool)
            )

            if need:
                idx = rng.choice(
                    pool.index.to_numpy(),
                    size=need,
                    replace=False
                )

                extra = pool.loc[idx].copy()
                extra["sample_reason"] = (
                    "IDENTITY_FILL"
                )

                out = pd.concat(
                    [out, extra],
                    ignore_index=True
                )

        return out.head(n).reset_index(drop=True)

    # challenge 전부
    challenge["sample_reason"] = (
        "ALL_IDENTITY_CHALLENGE"
    )

    direct_need = n - len(challenge)

    complex_direct = direct[
        direct["_is_semantically_complex"]
    ].copy()

    ordinary_direct = direct[
        ~direct["_is_semantically_complex"]
    ].copy()

    # direct 비교군 절반씩
    complex_n = min(
        len(complex_direct),
        int(np.ceil(direct_need / 2))
    )

    ordinary_n = min(
        len(ordinary_direct),
        direct_need - complex_n
    )

    parts = [challenge]

    used_direct = set()

    if complex_n:
        idx = rng.choice(
            complex_direct.index.to_numpy(),
            size=complex_n,
            replace=False
        )
        x = complex_direct.loc[idx].copy()
        x["sample_reason"] = (
            "COMPLEX_DIRECT_CONTROL"
        )
        parts.append(x)
        used_direct.update(idx.tolist())

    if ordinary_n:
        idx = rng.choice(
            ordinary_direct.index.to_numpy(),
            size=ordinary_n,
            replace=False
        )
        x = ordinary_direct.loc[idx].copy()
        x["sample_reason"] = (
            "ORDINARY_DIRECT_CONTROL"
        )
        parts.append(x)
        used_direct.update(idx.tolist())

    out = pd.concat(
        parts,
        ignore_index=True
    )

    # 부족하면 direct pool에서 추가
    if len(out) < n:
        pool = direct.loc[
            ~direct.index.isin(used_direct)
        ]

        need = min(
            n - len(out),
            len(pool)
        )

        if need:
            idx = rng.choice(
                pool.index.to_numpy(),
                size=need,
                replace=False
            )
            extra = pool.loc[idx].copy()
            extra["sample_reason"] = (
                "DIRECT_CONTROL_FILL"
            )
            out = pd.concat(
                [out, extra],
                ignore_index=True
            )

    return out.head(n).reset_index(drop=True)


# ============================================================
# 7-3. Local model
# ============================================================
def get_direct_local_model():
    """
    Hugging Face Transformers direct local inference.
    Ollama/API/server 불필요.

    첫 실행 시 model weight만 다운로드.
    이후 Hugging Face cache에서 로컬 재사용.
    """
    try:
        import torch
        import transformers
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
        )
    except Exception as e:
        raise RuntimeError(
            "regai conda 환경에서 transformers/torch가 필요합니다.\n"
            '예: conda install -c conda-forge '
            'python=3.11 pytorch transformers accelerate '
            'pydantic pandas openpyxl -y'
        ) from e

    print(
        f"[LOCAL AI] model loading: {AI_MODEL}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        AI_MODEL
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            AI_MODEL,
            torch_dtype="auto",
            device_map="auto",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            AI_MODEL,
            torch_dtype="auto",
        )

        if torch.cuda.is_available():
            model = model.to("cuda")

    model.eval()

    try:
        device = next(model.parameters()).device
    except Exception:
        device = "unknown"

    print(
        f"[LOCAL AI] inference device: {device}"
    )

    return tokenizer, model


def extract_first_json_object(text: str) -> str:
    t = clean(text)

    t = re.sub(
        r"<think>.*?</think>",
        "",
        t,
        flags=re.I | re.S,
    ).strip()

    start = t.find("{")

    if start < 0:
        raise ValueError(
            "JSON 시작 '{'를 찾지 못했습니다."
        )

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(t)):
        ch = t[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1

            if depth == 0:
                return t[start:i + 1]

    raise ValueError(
        "완전한 JSON object를 찾지 못했습니다."
    )


def make_semantic_ai_model():
    from pydantic import BaseModel

    class AISemanticExtraction(BaseModel):
        rule_id: str

        scope_type: Literal[
            "SINGLE_OR_DIRECT_CAS",
            "BROAD_SALT_SCOPE",
            "SALT_FAMILY",
            "MIXTURE_OR_REACTION",
            "GENERIC_GROUP_OR_DERIVATIVE",
            "STRUCTURAL_RANGE",
            "OTHER",
        ]

        has_exclusion: bool
        extraction_note: str

    return AISemanticExtraction


AI_SYSTEM_PROMPT = """
You are a conservative semantic parser for Korean chemical regulations.

This is a HYBRID system.

The deterministic Python rule engine already extracts:
- exact CAS numbers,
- acute/chronic/ecological concentration thresholds,
- numerical threshold comparisons.

DO NOT extract CAS numbers or thresholds.
DO NOT make the final legal/compliance decision.

Your task is ONLY to interpret natural-language regulatory scope.

You must classify exactly two semantic fields:

1. scope_type

SINGLE_OR_DIRECT_CAS
- a specific substance/rule adequately represented as a specific item.
- A specific compound name containing words such as chloride, sodium,
  borate, salt, compound, derivative, ammonium, etc. is NOT automatically
  a generic family.

BROAD_SALT_SCOPE
- explicit wording such as:
  "A와 그 염류", "A와 그 염", "A and its salts".

SALT_FAMILY
- a family/class of salts itself is regulated,
  e.g. chromates, nitrite salts, generic "... 염류",
  rather than one specifically named salt compound.

MIXTURE_OR_REACTION
- mixture, reaction mixture, reaction product,
  composition-defined mixture, copolymer, or similar.

GENERIC_GROUP_OR_DERIVATIVE
- generic compound family, derivatives, or chemical group,
  not one specific direct-CAS item.

STRUCTURAL_RANGE
- regulation is defined by molecular/structural range,
  such as C12-C14, C=10~16, alkyl-chain range,
  carbon-number range, or related structural definition.

OTHER
- a rule without direct CAS that does not fit the above classes.

2. has_exclusion

true ONLY when the regulation explicitly excludes a substance/group
or contains an explicit exception such as "제외", "다만 ... 제외".

Do not infer an exclusion merely from complex wording.

Be conservative.
Use only the supplied regulatory text and deterministic facts.
Return JSON only.
""".strip()


def local_generate_semantic_json(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 220,
):
    import torch

    messages = [
        {
            "role": "system",
            "content": AI_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
    except TypeError:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    try:
        device = next(model.parameters()).device
        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }
    except Exception:
        pass

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[-1]

    generated = outputs[0][input_len:]

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )

    return extract_first_json_object(text)


def normalize_ai_scope_aliases(raw_json: str) -> str:
    """
    Small local models sometimes emit semantically obvious label aliases.
    These are normalized before schema validation so formatting variation is
    scored as a classification result rather than discarded as a parse error.

    IMPORTANT:
    DIRECT_CAS -> SINGLE_OR_DIRECT_CAS only fixes the label vocabulary.
    If the semantic classification itself is wrong, it remains a measured error.
    """
    text = str(raw_json or "")

    aliases = {
        "DIRECT_CAS": "SINGLE_OR_DIRECT_CAS",
        "SINGLE_DIRECT_CAS": "SINGLE_OR_DIRECT_CAS",
        "DIRECT_OR_SINGLE_CAS": "SINGLE_OR_DIRECT_CAS",
        "REACTION_PRODUCT": "MIXTURE_OR_REACTION",
        "REACTION_OR_MIXTURE": "MIXTURE_OR_REACTION",
        "GENERIC_GROUP": "GENERIC_GROUP_OR_DERIVATIVE",
        "GENERIC_DERIVATIVE": "GENERIC_GROUP_OR_DERIVATIVE",
        "STRUCTURE_RANGE": "STRUCTURAL_RANGE",
    }

    for old, new in aliases.items():
        text = re.sub(
            rf'("scope_type"\s*:\s*")'
            rf'{re.escape(old)}'
            rf'(")',
            rf'\1{new}\2',
            text,
            flags=re.I,
        )

    return text


# ============================================================
# 7-4. Cache
# ============================================================
def load_sheet_if_exists(
    path: Path,
    sheet_name: str,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_excel(
            path,
            sheet_name=sheet_name,
        )
    except Exception:
        return pd.DataFrame()


def run_semantic_ai(
    sample: pd.DataFrame,
    prior_cache: pd.DataFrame,
):
    tokenizer, model = get_direct_local_model()
    AISemanticExtraction = (
        make_semantic_ai_model()
    )

    cache = {}

    if len(prior_cache):
        required = [
            "rule_id",
            "ai_model",
            "prompt_version",
            "source_hash",
            "ai_status",
        ]

        for c in required:
            if c not in prior_cache.columns:
                prior_cache[c] = ""

        for _, r in prior_cache.iterrows():
            if clean(r["ai_status"]) != "OK":
                continue

            key = (
                clean(r["rule_id"]),
                clean(r["ai_model"]),
                clean(r["prompt_version"]),
                clean(r["source_hash"]),
            )

            cache[key] = r.to_dict()

    out_rows = []

    for pos, (_, r) in enumerate(
        sample.iterrows(),
        start=1,
    ):
        rid = clean(r["rule_id"])

        key = (
            rid,
            AI_MODEL,
            AI_PROMPT_VERSION,
            clean(r["_source_hash"]),
        )

        if key in cache:
            z = cache[key].copy()
            z["cache_reused"] = True
            out_rows.append(z)

            print(
                f"[AI {pos}/{len(sample)}] "
                f"cache {rid}"
            )
            continue

        print(
            f"[AI {pos}/{len(sample)}] {rid}"
        )

        prompt = f"""
RULE_ID: {rid}

DETERMINISTIC_FACTS:
DIRECT_CAS_PRESENT: {str(bool(r['has_direct_cas'])).lower()}
DIRECT_CAS: {clean(r['direct_cas_final']) or 'NONE'}

REGULATORY_SUBSTANCE_NAME:
{clean(r['substance_name_ko'])}

REGULATORY_ROW:
{clean(r['source_text'])}

Return exactly this JSON structure:
{{
  "rule_id": "{rid}",
  "scope_type": "ONE_ALLOWED_SCOPE_TYPE",
  "has_exclusion": true_or_false,
  "extraction_note": "brief reason"
}}
""".strip()

        parsed = None
        last_error = ""

        for attempt in range(1, 4):
            try:
                raw = local_generate_semantic_json(
                    tokenizer,
                    model,
                    prompt,
                )

                raw = normalize_ai_scope_aliases(
                    raw
                )

                parsed = (
                    AISemanticExtraction
                    .model_validate_json(raw)
                )

                break

            except Exception as e:
                last_error = (
                    f"{type(e).__name__}: {e}"
                )

                prompt += (
                    "\nReturn valid JSON only. "
                    "No markdown, no thinking, no prose."
                )

        if parsed is None:
            out_rows.append({
                "rule_id": rid,
                "ai_status": "ERROR",
                "ai_error": last_error,
                "ai_model": AI_MODEL,
                "prompt_version": AI_PROMPT_VERSION,
                "source_hash": clean(
                    r["_source_hash"]
                ),
                "cache_reused": False,
            })

            continue

        d = parsed.model_dump()

        out_rows.append({
            "rule_id": rid,
            "ai_status": "OK",
            "ai_error": "",
            "ai_model": AI_MODEL,
            "prompt_version": AI_PROMPT_VERSION,
            "source_hash": clean(
                r["_source_hash"]
            ),
            "cache_reused": False,

            "ai_scope_type": d["scope_type"],
            "ai_has_exclusion": bool(
                d["has_exclusion"]
            ),
            "ai_extraction_note": clean(
                d["extraction_note"]
            ),
        })

    ai = pd.DataFrame(out_rows)

    ref_cols = [
        "rule_id",
        "designation_id",
        "sub_no",
        "substance_name_ko",
        "source_text",
        "direct_cas_final",
        "has_direct_cas",
        "acute_threshold_pct",
        "chronic_threshold_pct",
        "eco_threshold_pct",
        "ref_scope_type",
        "ref_has_exclusion",
        "ref_requires_extended_identity",
        "sample_reason",
        "_source_hash",
    ]

    refs = sample[
        ref_cols
    ].copy().rename(
        columns={
            "_source_hash":
            "sample_source_hash"
        }
    )

    # cache에 과거 ref 열이 있어도 AI 열만 사용
    ai_cols = [
        c for c in ai.columns
        if c in {
            "rule_id",
            "ai_status",
            "ai_error",
            "ai_model",
            "prompt_version",
            "source_hash",
            "cache_reused",
            "ai_scope_type",
            "ai_has_exclusion",
            "ai_extraction_note",
        }
    ]

    merged = refs.merge(
        ai[ai_cols],
        on="rule_id",
        how="left",
    )

    return merged


# ============================================================
# 7-5. Metrics
# ============================================================
def bool_value(x):
    if isinstance(x, bool):
        return x

    s = clean(x).lower()

    if s in {
        "1", "true", "yes", "y",
        "예", "맞음",
    }:
        return True

    if s in {
        "0", "false", "no", "n",
        "아니오", "아님",
    }:
        return False

    return None


def binary_metrics(
    truth: pd.Series,
    pred: pd.Series,
):
    pairs = []

    for t, p in zip(truth, pred):
        tt = bool_value(t)
        pp = bool_value(p)

        if tt is None or pp is None:
            continue

        pairs.append(
            (tt, pp)
        )

    if not pairs:
        return {
            "n": 0,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }

    tp = sum(t and p for t, p in pairs)
    tn = sum((not t) and (not p) for t, p in pairs)
    fp = sum((not t) and p for t, p in pairs)
    fn = sum(t and (not p) for t, p in pairs)

    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else np.nan
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn)
        else np.nan
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (
            np.isfinite(precision)
            and np.isfinite(recall)
            and (precision + recall) > 0
        )
        else np.nan
    )

    return {
        "n": len(pairs),
        "accuracy": (tp + tn) / len(pairs),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def multiclass_accuracy(
    truth: pd.Series,
    pred: pd.Series,
):
    pairs = [
        (clean(t), clean(p))
        for t, p in zip(truth, pred)
        if clean(t) and clean(p)
    ]

    if not pairs:
        return {
            "n": 0,
            "accuracy": np.nan,
        }

    return {
        "n": len(pairs),
        "accuracy": sum(
            t == p
            for t, p in pairs
        ) / len(pairs),
    }


def multiclass_macro_f1(
    truth: pd.Series,
    pred: pd.Series,
):
    pairs = [
        (clean(t), clean(p))
        for t, p in zip(truth, pred)
        if clean(t) and clean(p)
    ]

    if not pairs:
        return np.nan

    labels = sorted(
        set(t for t, _ in pairs)
        | set(p for _, p in pairs)
    )

    f1s = []

    for label in labels:
        tp = sum(
            t == label and p == label
            for t, p in pairs
        )

        fp = sum(
            t != label and p == label
            for t, p in pairs
        )

        fn = sum(
            t == label and p != label
            for t, p in pairs
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp)
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn)
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        f1s.append(f1)

    return float(
        np.mean(f1s)
    )



def derive_hybrid_extended_identity(row: pd.Series) -> bool:
    """
    최종 Hybrid V4:
    extended identity 필요 여부는 LLM에게 맡기지 않는다.

    deterministic facts:
    - direct CAS가 없으면 True
    - direct CAS가 있더라도 AI가 BROAD_SALT_SCOPE로 해석한 경우 True
    - 그 외 direct-CAS rule은 False

    이유:
    identity requirement는 이미 direct-CAS 존재 여부라는 명확한
    구조화 정보와 scope class로 결정할 수 있으므로 LLM이 별도로
    yes/no를 예측하는 것은 중복이며 오히려 오류를 증가시킨다.
    """
    has_direct = bool(row.get("has_direct_cas"))
    ai_scope = clean(row.get("ai_scope_type"))

    if not has_direct:
        return True

    if ai_scope == "BROAD_SALT_SCOPE":
        return True

    return False


def semantic_row_scoring(
    ai: pd.DataFrame,
):
    if not len(ai):
        return pd.DataFrame()

    x = ai.copy()
    ok = x["ai_status"].eq("OK")

    x["match_scope_type"] = (
        ok
        & (
            x["ref_scope_type"].astype(str)
            == x["ai_scope_type"].astype(str)
        )
    )

    x["match_has_exclusion"] = (
        ok
        & (
            x["ref_has_exclusion"].astype(bool)
            == x["ai_has_exclusion"].astype(bool)
        )
    )

    # AI raw extended-identity output is retained only as audit information.
    # Final hybrid identity requirement is deterministic.
    x["hybrid_requires_extended_identity"] = x.apply(
        derive_hybrid_extended_identity,
        axis=1,
    )

    x["match_hybrid_extended_identity"] = (
        ok
        & (
            x[
                "ref_requires_extended_identity"
            ].astype(bool)
            == x[
                "hybrid_requires_extended_identity"
            ].astype(bool)
        )
    )

    # AI의 핵심 역할은 semantic scope classification.
    # exclusion은 보조 semantic field.
    x["hybrid_semantic_core_exact"] = (
        x[
            [
                "match_scope_type",
                "match_has_exclusion",
                "match_hybrid_extended_identity",
            ]
        ]
        .all(axis=1)
    )

    return x

def ai_vs_rule_metrics(
    scored: pd.DataFrame,
):
    x = scored[
        scored["ai_status"].eq("OK")
    ].copy()

    if not len(x):
        return pd.DataFrame()

    scope = multiclass_accuracy(
        x["ref_scope_type"],
        x["ai_scope_type"],
    )

    exclusion = binary_metrics(
        x["ref_has_exclusion"],
        x["ai_has_exclusion"],
    )

    hybrid_ext = binary_metrics(
        x["ref_requires_extended_identity"],
        x["hybrid_requires_extended_identity"],
    )

    rows = [
        {
            "reference": "RULE_BASED_REFERENCE",
            "metric": "scope_accuracy",
            "n": scope["n"],
            "value": scope["accuracy"],
        },
        {
            "reference": "RULE_BASED_REFERENCE",
            "metric": "scope_macro_f1",
            "n": scope["n"],
            "value": multiclass_macro_f1(
                x["ref_scope_type"],
                x["ai_scope_type"],
            ),
        },
        {
            "reference": "RULE_BASED_REFERENCE",
            "metric": "exclusion_accuracy",
            "n": exclusion["n"],
            "value": exclusion["accuracy"],
        },
        {
            "reference": "RULE_BASED_REFERENCE",
            "metric": "exclusion_f1",
            "n": exclusion["n"],
            "value": exclusion["f1"],
        },
        {
            "reference": "RULE_BASED_REFERENCE",
            "metric": "hybrid_extended_identity_accuracy",
            "n": hybrid_ext["n"],
            "value": hybrid_ext["accuracy"],
        },
        {
            "reference": "RULE_BASED_REFERENCE",
            "metric": "hybrid_extended_identity_precision",
            "n": hybrid_ext["n"],
            "value": hybrid_ext["precision"],
        },
        {
            "reference": "RULE_BASED_REFERENCE",
            "metric": "hybrid_extended_identity_recall",
            "n": hybrid_ext["n"],
            "value": hybrid_ext["recall"],
        },
        {
            "reference": "RULE_BASED_REFERENCE",
            "metric": "hybrid_extended_identity_f1",
            "n": hybrid_ext["n"],
            "value": hybrid_ext["f1"],
        },
        {
            "reference": "RULE_BASED_REFERENCE",
            "metric": "hybrid_semantic_core_exact",
            "n": len(x),
            "value": float(
                x["hybrid_semantic_core_exact"].mean()
            ),
        },
    ]

    return pd.DataFrame(rows)

def ai_by_scope(
    scored: pd.DataFrame,
):
    x = scored[
        scored["ai_status"].eq("OK")
    ].copy()

    if not len(x):
        return pd.DataFrame()

    rows = []

    for scope, g in x.groupby(
        "ref_scope_type"
    ):
        ext = binary_metrics(
            g[
                "ref_requires_extended_identity"
            ],
            g[
                "hybrid_requires_extended_identity"
            ],
        )

        rows.append({
            "ref_scope_type": scope,
            "n": len(g),
            "scope_accuracy": float(
                g["match_scope_type"].mean()
            ),
            "exclusion_accuracy": float(
                g["match_has_exclusion"].mean()
            ),
            "hybrid_extended_identity_accuracy": ext[
                "accuracy"
            ],
            "hybrid_semantic_core_exact": float(
                g[
                    "hybrid_semantic_core_exact"
                ].mean()
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# 7-6. Expert validation
# ============================================================
EXPERT_COLS = [
    "expert1_scope_type",
    "expert1_has_exclusion",
    "expert1_requires_extended_identity",
    "expert1_name",

    "expert2_scope_type",
    "expert2_has_exclusion",
    "expert2_requires_extended_identity",
    "expert2_name",

    "adjudicated_scope_type",
    "adjudicated_has_exclusion",
    "adjudicated_requires_extended_identity",
    "adjudication_status",
    "adjudicator",
    "expert_notes",
]


def make_expert_validation_sheet(
    scored: pd.DataFrame,
    prior: pd.DataFrame,
):
    cols = [
        "rule_id",
        "designation_id",
        "substance_name_ko",
        "source_text",
        "direct_cas_final",
        "has_direct_cas",

        "ref_scope_type",
        "ai_scope_type",

        "ref_has_exclusion",
        "ai_has_exclusion",

        "ref_requires_extended_identity",
        "hybrid_requires_extended_identity",

        "hybrid_semantic_core_exact",
        "sample_reason",
    ]

    x = scored[
        [c for c in cols if c in scored.columns]
    ].copy()

    for c in EXPERT_COLS:
        x[c] = ""

    if (
        len(prior)
        and "rule_id" in prior.columns
    ):
        keep_cols = [
            "rule_id"
        ] + [
            c for c in EXPERT_COLS
            if c in prior.columns
        ]

        keep = prior[
            keep_cols
        ].drop_duplicates("rule_id")

        x = x.drop(
            columns=EXPERT_COLS,
            errors="ignore",
        ).merge(
            keep,
            on="rule_id",
            how="left",
        )

        for c in EXPERT_COLS:
            if c not in x.columns:
                x[c] = ""
            else:
                x[c] = x[c].fillna("")

    return x


def valid_adjudicated_rows(
    expert: pd.DataFrame,
):
    if not len(expert):
        return expert.head(0)

    status = (
        expert["adjudication_status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    good_status = status.isin(
        {
            "VALIDATED",
            "FINAL",
            "COMPLETE",
            "COMPLETED",
            "1",
            "TRUE",
        }
    )

    scope_ok = (
        expert[
            "adjudicated_scope_type"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    return expert[
        good_status & scope_ok
    ].copy()


def join_expert_gold(
    scored: pd.DataFrame,
    expert: pd.DataFrame,
):
    gold = valid_adjudicated_rows(
        expert
    )

    if not len(gold):
        return pd.DataFrame()

    keep = [
        "rule_id",
        "adjudicated_scope_type",
        "adjudicated_has_exclusion",
        "adjudicated_requires_extended_identity",
    ]

    return scored.merge(
        gold[keep],
        on="rule_id",
        how="inner",
    )


def compare_source_to_expert(
    x: pd.DataFrame,
    source: str,
):
    """
    source = AI or RULE
    """
    if not len(x):
        return pd.DataFrame()

    if source == "AI":
        scope_pred = x["ai_scope_type"]
        exc_pred = x["ai_has_exclusion"]
        ext_pred = x[
            "hybrid_requires_extended_identity"
        ]
    elif source == "RULE":
        scope_pred = x["ref_scope_type"]
        exc_pred = x["ref_has_exclusion"]
        ext_pred = x[
            "ref_requires_extended_identity"
        ]
    else:
        raise ValueError(source)

    scope = multiclass_accuracy(
        x["adjudicated_scope_type"],
        scope_pred,
    )

    exc = binary_metrics(
        x["adjudicated_has_exclusion"],
        exc_pred,
    )

    ext = binary_metrics(
        x[
            "adjudicated_requires_extended_identity"
        ],
        ext_pred,
    )

    # row-wise semantic exact
    row_exact = []

    for i in range(len(x)):
        gold_scope = clean(
            x.iloc[i][
                "adjudicated_scope_type"
            ]
        )

        pred_scope = clean(
            scope_pred.iloc[i]
        )

        ge = bool_value(
            x.iloc[i][
                "adjudicated_has_exclusion"
            ]
        )
        pe = bool_value(
            exc_pred.iloc[i]
        )

        gx = bool_value(
            x.iloc[i][
                "adjudicated_requires_extended_identity"
            ]
        )
        px = bool_value(
            ext_pred.iloc[i]
        )

        ok = (
            gold_scope
            and pred_scope
            and gold_scope == pred_scope
            and ge is not None
            and pe is not None
            and ge == pe
            and gx is not None
            and px is not None
            and gx == px
        )

        row_exact.append(
            bool(ok)
        )

    rows = [
        {
            "system": source,
            "reference": "ADJUDICATED_EXPERT_GOLD",
            "metric": "scope_accuracy",
            "n": scope["n"],
            "value": scope["accuracy"],
        },
        {
            "system": source,
            "reference": "ADJUDICATED_EXPERT_GOLD",
            "metric": "scope_macro_f1",
            "n": scope["n"],
            "value": multiclass_macro_f1(
                x[
                    "adjudicated_scope_type"
                ],
                scope_pred,
            ),
        },
        {
            "system": source,
            "reference": "ADJUDICATED_EXPERT_GOLD",
            "metric": "exclusion_accuracy",
            "n": exc["n"],
            "value": exc["accuracy"],
        },
        {
            "system": source,
            "reference": "ADJUDICATED_EXPERT_GOLD",
            "metric": "exclusion_f1",
            "n": exc["n"],
            "value": exc["f1"],
        },
        {
            "system": source,
            "reference": "ADJUDICATED_EXPERT_GOLD",
            "metric": "extended_identity_accuracy",
            "n": ext["n"],
            "value": ext["accuracy"],
        },
        {
            "system": source,
            "reference": "ADJUDICATED_EXPERT_GOLD",
            "metric": "extended_identity_precision",
            "n": ext["n"],
            "value": ext["precision"],
        },
        {
            "system": source,
            "reference": "ADJUDICATED_EXPERT_GOLD",
            "metric": "extended_identity_recall",
            "n": ext["n"],
            "value": ext["recall"],
        },
        {
            "system": source,
            "reference": "ADJUDICATED_EXPERT_GOLD",
            "metric": "extended_identity_f1",
            "n": ext["n"],
            "value": ext["f1"],
        },
        {
            "system": source,
            "reference": "ADJUDICATED_EXPERT_GOLD",
            "metric": "semantic_core_exact",
            "n": len(row_exact),
            "value": (
                float(np.mean(row_exact))
                if row_exact
                else np.nan
            ),
        },
    ]

    return pd.DataFrame(rows)


# ============================================================
# 7-7. Inter-reviewer agreement
# ============================================================
def cohen_kappa(
    a: pd.Series,
    b: pd.Series,
):
    pairs = [
        (clean(x), clean(y))
        for x, y in zip(a, b)
        if clean(x) and clean(y)
    ]

    if not pairs:
        return {
            "n": 0,
            "agreement": np.nan,
            "kappa": np.nan,
        }

    labels = sorted(
        set(x for x, _ in pairs)
        | set(y for _, y in pairs)
    )

    n = len(pairs)

    po = sum(
        x == y
        for x, y in pairs
    ) / n

    pa = {
        lab: sum(
            x == lab
            for x, _ in pairs
        ) / n
        for lab in labels
    }

    pb = {
        lab: sum(
            y == lab
            for _, y in pairs
        ) / n
        for lab in labels
    }

    pe = sum(
        pa.get(lab, 0)
        * pb.get(lab, 0)
        for lab in labels
    )

    kappa = (
        (po - pe) / (1 - pe)
        if (1 - pe) > 1e-12
        else np.nan
    )

    return {
        "n": n,
        "agreement": po,
        "kappa": kappa,
    }


def binary_series_as_label(s):
    vals = []

    for x in s:
        b = bool_value(x)

        if b is None:
            vals.append("")
        else:
            vals.append(
                "TRUE" if b else "FALSE"
            )

    return pd.Series(vals)


def reviewer_agreement(
    expert: pd.DataFrame,
):
    if not len(expert):
        return pd.DataFrame()

    scope = cohen_kappa(
        expert["expert1_scope_type"],
        expert["expert2_scope_type"],
    )

    excl = cohen_kappa(
        binary_series_as_label(
            expert[
                "expert1_has_exclusion"
            ]
        ),
        binary_series_as_label(
            expert[
                "expert2_has_exclusion"
            ]
        ),
    )

    ext = cohen_kappa(
        binary_series_as_label(
            expert[
                "expert1_requires_extended_identity"
            ]
        ),
        binary_series_as_label(
            expert[
                "expert2_requires_extended_identity"
            ]
        ),
    )

    return pd.DataFrame([
        {
            "field": "scope_type",
            **scope,
        },
        {
            "field": "has_exclusion",
            **excl,
        },
        {
            "field": "requires_extended_identity",
            **ext,
        },
    ])


# ============================================================
# 7-8. Excel
# ============================================================
def append_hybrid_ai_sheets(
    path: Path,
    sample: pd.DataFrame,
    scored: pd.DataFrame,
    vs_rule: pd.DataFrame,
    by_scope: pd.DataFrame,
    disagreements: pd.DataFrame,
    expert: pd.DataFrame,
    expert_gold_rows: pd.DataFrame,
    ai_vs_expert: pd.DataFrame,
    rule_vs_expert: pd.DataFrame,
    reviewer_kappa: pd.DataFrame,
    metadata: pd.DataFrame,
):
    with pd.ExcelWriter(
        path,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:

        sample.drop(
            columns=[
                c for c in sample.columns
                if c.startswith("_")
            ],
            errors="ignore",
        ).to_excel(
            writer,
            sheet_name="AI_semantic_sample",
            index=False,
        )

        scored.to_excel(
            writer,
            sheet_name="AI_semantic_results",
            index=False,
        )

        vs_rule.to_excel(
            writer,
            sheet_name="AI_vs_rule_metrics",
            index=False,
        )

        by_scope.to_excel(
            writer,
            sheet_name="AI_by_scope",
            index=False,
        )

        disagreements.to_excel(
            writer,
            sheet_name="AI_disagreements",
            index=False,
        )

        expert.to_excel(
            writer,
            sheet_name="Expert_validation",
            index=False,
        )

        expert_gold_rows.to_excel(
            writer,
            sheet_name="Expert_gold_rows",
            index=False,
        )

        ai_vs_expert.to_excel(
            writer,
            sheet_name="AI_vs_expert_metrics",
            index=False,
        )

        rule_vs_expert.to_excel(
            writer,
            sheet_name="Rule_vs_expert_metrics",
            index=False,
        )

        reviewer_kappa.to_excel(
            writer,
            sheet_name="Reviewer_agreement",
            index=False,
        )

        metadata.to_excel(
            writer,
            sheet_name="AI_run_metadata",
            index=False,
        )

        for ws in writer.book.worksheets:
            try:
                autosize_sheet(ws)
            except Exception:
                pass




def load_prior_ai_cache():
    """
    AI 재실행 없이 기존 semantic 결과를 재사용할 수 있도록 한다.

    검색 순서
    --------
    1) 현재 새 output workbook
    2) 이전 연구 결과 workbook 후보

    반환값은 기존 AI_semantic_results sheet이며,
    없으면 빈 DataFrame을 반환한다.
    """
    candidates = [OUT_XLSX] + AI_CACHE_CANDIDATES

    for p in candidates:
        try:
            if p.exists():
                x = load_sheet_if_exists(
                    p,
                    "AI_semantic_results",
                )
                if x is not None and len(x):
                    print(
                        f"[AI cache] reused: {p.name} "
                        f"({len(x):,} rows)"
                    )
                    return x
        except Exception:
            continue

    return pd.DataFrame()


# ============================================================
# 7-9. Embedded Stage 0 factorial evidence module
# ============================================================
# The original V2 factorial pipeline is embedded so that the final framework
# remains self-contained.  If an external updated factorial pipeline is found,
# that external file is preferred; otherwise this embedded version is used.
def preliminary_summary_table(tables: dict):
    rows = []

    perf = tables.get(
        "T4a_model_performance",
        pd.DataFrame(),
    )

    if len(perf):
        for _, r in perf.iterrows():
            rows.append({
                "evidence_block": "CONTROLLED_SCREENING",
                "metric_or_model": clean(
                    r.get("model")
                ),
                "value": r.get(
                    "benchmark_accuracy",
                    np.nan,
                ),
                "false_safe_rate": r.get(
                    "false_safe_rate",
                    np.nan,
                ),
                "over_screening_rate": r.get(
                    "over_screening_rate",
                    np.nan,
                ),
                "interpretation": (
                    "Controlled stress-test only; "
                    "not real-company prevalence."
                ),
            })

    identity = tables.get(
        "T3",
        pd.DataFrame(),
    )

    if len(identity):
        total_rule_rows = np.nan
        total_designations = np.nan

        # Current V2 T3 schema:
        # summary_type / identity_class / n_rule_rows / n_unique_designations
        if {
            "summary_type",
            "n_rule_rows",
        }.issubset(identity.columns):
            total_rows = identity[
                identity["summary_type"]
                .astype(str)
                .eq("TOTAL")
            ]

            if len(total_rows):
                total_rule_rows = pd.to_numeric(
                    total_rows.iloc[0].get(
                        "n_rule_rows"
                    ),
                    errors="coerce",
                )
                total_designations = pd.to_numeric(
                    total_rows.iloc[0].get(
                        "n_unique_designations"
                    ),
                    errors="coerce",
                )
            else:
                primary = identity[
                    identity["summary_type"]
                    .astype(str)
                    .eq("PRIMARY_CLASS")
                ]
                total_rule_rows = pd.to_numeric(
                    primary.get(
                        "n_rule_rows",
                        pd.Series(dtype=float),
                    ),
                    errors="coerce",
                ).fillna(0).sum()

        # Legacy fallback
        elif "count" in identity.columns:
            total_rule_rows = pd.to_numeric(
                identity["count"],
                errors="coerce",
            ).fillna(0).sum()

        if np.isfinite(total_rule_rows):
            total_rule_rows = int(total_rule_rows)

        if np.isfinite(total_designations):
            total_designations = int(total_designations)

        rows.append({
            "evidence_block": "IDENTITY_HETEROGENEITY",
            "metric_or_model": "non_direct_CAS_rule_rows",
            "value": total_rule_rows,
            "false_safe_rate": np.nan,
            "over_screening_rate": np.nan,
            "interpretation": (
                "Regulatory rule rows not fully represented "
                "by one direct CAS identifier."
            ),
        })

        rows.append({
            "evidence_block": "IDENTITY_HETEROGENEITY",
            "metric_or_model": "non_direct_CAS_unique_designations",
            "value": total_designations,
            "false_safe_rate": np.nan,
            "over_screening_rate": np.nan,
            "interpretation": (
                "Unique designation IDs represented by one or more "
                "non-direct-CAS regulatory rules."
            ),
        })

    return pd.DataFrame(rows)


def append_research_integration_sheets(
    path: Path,
    preliminary: dict,
    alerts: pd.DataFrame,
):
    """
    STAGE 0 및 연구설계/alert 결과를 기업형 결과 workbook에 추가.
    """
    if not path.exists():
        return

    framework = pd.DataFrame([
        {
            "stage": 0,
            "module": "Preliminary failure-mode benchmark",
            "question": (
                "Why is simple CAS-based regulatory screening insufficient?"
            ),
            "output": (
                "Over-screening from omitted concentration rules and "
                "false-safe misses from non-direct-CAS scope."
            ),
        },
        {
            "stage": 1,
            "module": "Regulatory update detector",
            "question": (
                "Has the official regulation changed since the approved local version?"
            ),
            "output": "New/version-changed regulation candidate",
        },
        {
            "stage": 2,
            "module": "Local LLM semantic parser",
            "question": (
                "What semantic scope and explicit exclusions are expressed in the rule?"
            ),
            "output": (
                "Scope class + exclusion flag + evidence text"
            ),
        },
        {
            "stage": 3,
            "module": "Deterministic applicability engine",
            "question": (
                "Does an inventory item satisfy CAS/validated relation "
                "and concentration conditions?"
            ),
            "output": (
                "Reproducible triggered / below-threshold / review status"
            ),
        },
        {
            "stage": 4,
            "module": "Update-impact triage",
            "question": (
                "Which company materials changed regulatory status after the amendment?"
            ),
            "output": (
                "Newly triggered / no longer triggered / still triggered / review"
            ),
        },
        {
            "stage": 5,
            "module": "Explainable enterprise alert",
            "question": (
                "What should a company regulatory manager review first, and why?"
            ),
            "output": (
                "Priority alert + changed field + rule text + inventory evidence"
            ),
        },
    ])

    rqs = pd.DataFrame([
        {
            "RQ": "RQ1",
            "question": (
                "단순화된 규제정보 표현은 어떤 조건에서 "
                "chemical screening 실패를 만드는가?"
            ),
            "method": (
                "Factorial controlled benchmark / failure-mode analysis"
            ),
        },
        {
            "RQ": "RQ2",
            "question": (
                "Local LLM은 비정형 regulatory scope를 "
                "구조화된 scope class로 얼마나 일관되게 변환하는가?"
            ),
            "method": (
                "LLM semantic parsing + agreement with rule-based reference"
            ),
        },
        {
            "RQ": "RQ3",
            "question": (
                "Hybrid framework는 어떤 규제유형을 안전하게 자동판정하고, "
                "어떤 사례를 REVIEW_REQUIRED로 abstain해야 하는가?"
            ),
            "method": (
                "Controlled automation-boundary benchmark + selective abstention evaluation"
            ),
        },
        {
            "RQ": "RQ4",
            "question": (
                "Selective abstention을 적용했을 때 automation coverage와 "
                "human review burden 사이의 trade-off는 어떻게 나타나는가?"
            ),
            "method": (
                "Automation coverage / review burden / unsafe-auto analysis"
            ),
        },
    ])

    summary = preliminary_summary_table(
        preliminary
    )

    selected = [
        ("Research_framework", framework),
        ("Research_questions", rqs),
        ("Preliminary_evidence_summary", summary),
        ("Prelim_T1d_aggregation", preliminary.get("T1d")),
        ("Prelim_T3_identity", preliminary.get("T3")),
        ("Prelim_T4a_performance", preliminary.get("T4a_model_performance")),
        ("Prelim_T4e_failures", preliminary.get("T4e_failure_summary")),
        ("Prelim_T6_traceability", preliminary.get("T6b_source_audit_summary")),
        ("Enterprise_alerts", alerts),
    ]

    with pd.ExcelWriter(
        path,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        for name, df in selected:
            if df is None:
                df = pd.DataFrame()

            df.to_excel(
                writer,
                sheet_name=name[:31],
                index=False,
            )

        for ws in writer.book.worksheets:
            try:
                autosize_sheet(ws)
            except Exception:
                pass


def build_explainable_enterprise_alerts(
    company_update_impact: pd.DataFrame,
    extended_review_queue: pd.DataFrame,
):
    """
    법 위반 확정 알림이 아니라 regulatory impact triage 알림.

    PRIORITY 1:
      새 규제버전에서 새로 trigger된 회사 inventory item

    PRIORITY 2:
      변경 규칙이 non-direct-CAS / extended scope여서 자동판정이 불완전한 경우

    PRIORITY 3:
      기존에도/현재도 trigger되지만 규칙 자체가 변경된 경우

    INFORMATION:
      규제 적용이 해소되었거나 direct/validated rule에서 비trigger인 경우
    """
    rows = []

    if company_update_impact is not None and len(company_update_impact):
        for _, r in company_update_impact.iterrows():
            impact = clean(
                r.get("company_impact")
            )

            review = bool(
                r.get(
                    "human_review_required",
                    False,
                )
            )

            if impact == "NEWLY_TRIGGERED_BY_UPDATE":
                level = "PRIORITY_1_NEW_POTENTIAL_IMPACT"
                action = (
                    "규제담당자 우선 검토: 신규 적용 가능성 확인"
                )
            elif review:
                level = "PRIORITY_2_REVIEW_REQUIRED"
                action = (
                    "규제범위/identity 관계 수동 확인"
                )
            elif impact == "STILL_TRIGGERED":
                level = "PRIORITY_3_CHANGED_RULE_STILL_TRIGGERED"
                action = (
                    "변경된 규칙 내용과 기존 이행조치 재확인"
                )
            else:
                level = "INFORMATION"
                action = (
                    "기록 유지; 필요 시 변경근거 확인"
                )

            rows.append({
                "alert_level": level,
                "inventory_id": r.get(
                    "inventory_id",
                    "",
                ),
                "product_name": r.get(
                    "product_name",
                    "",
                ),
                "chemical_name": r.get(
                    "chemical_name",
                    "",
                ),
                "cas": r.get("cas", ""),
                "concentration_pct": r.get(
                    "concentration_pct",
                    np.nan,
                ),
                "designation_id": r.get(
                    "designation_id",
                    "",
                ),
                "regulatory_update_type": r.get(
                    "regulatory_update_type",
                    "",
                ),
                "company_impact": impact,
                "review_required": review,
                "changed_fields": r.get(
                    "changed_fields",
                    "",
                ),
                "recommended_action": action,
                "old_rule_text": r.get(
                    "old_rule_text",
                    "",
                ),
                "new_rule_text": r.get(
                    "new_rule_text",
                    "",
                ),
                "legal_interpretation_boundary": (
                    "Potential compliance impact triage; "
                    "not a final legal violation determination."
                ),
            })

    # company CAS로 자동 연결할 수 없더라도,
    # changed extended-scope rule 자체를 alert queue에 별도 추가
    if extended_review_queue is not None and len(extended_review_queue):
        for _, r in extended_review_queue.iterrows():
            rows.append({
                "alert_level": "PRIORITY_2_EXTENDED_SCOPE_RULE",
                "inventory_id": "",
                "product_name": "",
                "chemical_name": r.get(
                    "new_substance_names",
                    r.get(
                        "substance_name_ko",
                        "",
                    ),
                ),
                "cas": "",
                "concentration_pct": np.nan,
                "designation_id": r.get(
                    "designation_id",
                    "",
                ),
                "regulatory_update_type": r.get(
                    "update_change_type",
                    "",
                ),
                "company_impact": (
                    "POTENTIAL_IDENTITY_SCOPE_IMPACT"
                ),
                "review_required": True,
                "changed_fields": r.get(
                    "changed_fields",
                    "",
                ),
                "recommended_action": (
                    "Local LLM scope 해석 후 company identity relation 확인"
                ),
                "old_rule_text": r.get(
                    "old_rule_text",
                    "",
                ),
                "new_rule_text": r.get(
                    "new_rule_text",
                    "",
                ),
                "legal_interpretation_boundary": (
                    "Potential scope impact; requires identity validation."
                ),
            })

    out = pd.DataFrame(rows)

    if len(out):
        order = {
            "PRIORITY_1_NEW_POTENTIAL_IMPACT": 1,
            "PRIORITY_2_REVIEW_REQUIRED": 2,
            "PRIORITY_2_EXTENDED_SCOPE_RULE": 2,
            "PRIORITY_3_CHANGED_RULE_STILL_TRIGGERED": 3,
            "INFORMATION": 4,
        }

        out["_priority"] = (
            out["alert_level"]
            .map(order)
            .fillna(9)
        )

        out = (
            out.sort_values(
                ["_priority", "inventory_id", "designation_id"]
            )
            .drop(
                columns=["_priority"],
                errors="ignore",
            )
            .reset_index(drop=True)
        )

    return out



# ============================================================
# 8-1. Controlled automation-boundary challenge benchmark
# ============================================================
def minimum_active_threshold_from_row(r):
    vals = []

    for c in [
        "acute_threshold_pct",
        "chronic_threshold_pct",
        "eco_threshold_pct",
    ]:
        v = pd.to_numeric(
            pd.Series([r.get(c)]),
            errors="coerce",
        ).iloc[0]

        if np.isfinite(v):
            vals.append(float(v))

    return min(vals) if vals else np.nan


def embedded_cas_list(text):
    return list(
        dict.fromkeys(
            CAS_RE.findall(
                clean(text)
            )
        )
    )


def build_automation_challenge_inventory(
    scored: pd.DataFrame,
):
    """
    Local LLM의 semantic evaluation sample을 기업형 challenge inventory로 변환한다.

    각 regulatory rule에 대해:
      - threshold 80% (below)
      - threshold 120% (above)
    두 조건을 생성한다.

    중요한 설계
    ------------
    * direct-CAS rule:
        실제 direct CAS를 company CAS로 사용.
    * BROAD_SALT_SCOPE:
        source text에 명시된 parent CAS를 company CAS로 사용.
        규칙 본문의 direct_cas field는 비어 있으므로 단순 CAS DB 방식은 놓칠 수 있다.
        그러나 "A와 그 염류"에서 A 자체는 명시적으로 범위에 포함되므로
        parent CAS exact match는 controlled benchmark에서 안전한 identity evidence이다.
    * 기타 extended scope:
        CAS를 임의로 만들지 않는다.
        해당 regulatory entity와 inventory item의 관계는 known-by-design으로만 둔다.
        Proposed system은 자동 판정하지 않고 REVIEW_REQUIRED로 abstain해야 한다.

    이 benchmark는 실제 기업 prevalence 추정용이 아니다.
    """
    if scored is None or not len(scored):
        return pd.DataFrame()

    rows = []
    case_no = 0

    for _, r in scored.iterrows():
        thr = minimum_active_threshold_from_row(r)

        if not np.isfinite(thr):
            continue

        direct = splitsemi(
            r.get("direct_cas_final")
        )

        embedded = embedded_cas_list(
            r.get("source_text")
        )

        ref_scope = clean(
            r.get("ref_scope_type")
        )

        if direct:
            inventory_cas = direct[0]
            identity_truth_type = "DIRECT_CAS_MEMBER"
            identity_input_mode = "DIRECT_CAS"
        elif (
            ref_scope == "BROAD_SALT_SCOPE"
            and embedded
        ):
            # first embedded CAS is normally the named parent substance.
            inventory_cas = embedded[0]
            identity_truth_type = "EXPLICIT_PARENT_MEMBER_OF_BROAD_SALT_SCOPE"
            identity_input_mode = "PARENT_CAS_EMBEDDED_IN_RULE_TEXT"
        else:
            inventory_cas = ""
            identity_truth_type = "KNOWN_BY_DESIGN_EXTENDED_SCOPE_MEMBER"
            identity_input_mode = "NO_SINGLE_CAS_IDENTITY"

        for factor in CHALLENGE_FACTORS:
            case_no += 1

            conc = max(
                0.000001,
                min(
                    100.0,
                    float(thr) * float(factor),
                ),
            )

            gold = bool(
                float(conc) >= float(thr)
            )

            rows.append({
                "challenge_id": f"AUTOBOUND_{case_no:04d}",
                "rule_id": clean(r.get("rule_id")),
                "designation_id": clean(
                    r.get("designation_id")
                ),
                "substance_name_ko": clean(
                    r.get("substance_name_ko")
                ),
                "inventory_chemical_name": clean(
                    r.get("substance_name_ko")
                ),
                "inventory_cas": inventory_cas,
                "concentration_pct": conc,
                "threshold_pct": float(thr),
                "threshold_factor": float(factor),
                "scenario": (
                    "BELOW_THRESHOLD"
                    if factor < 1
                    else "ABOVE_THRESHOLD"
                ),

                "ref_scope_type": ref_scope,
                "ref_has_exclusion": bool(
                    r.get("ref_has_exclusion", False)
                ),
                "ref_requires_extended_identity": bool(
                    r.get(
                        "ref_requires_extended_identity",
                        False,
                    )
                ),

                "ai_status": clean(
                    r.get("ai_status")
                ),
                "ai_scope_type": clean(
                    r.get("ai_scope_type")
                ),
                "ai_has_exclusion": r.get(
                    "ai_has_exclusion",
                    np.nan,
                ),
                "ai_note": clean(
                    r.get("ai_note")
                ),

                "direct_cas_final": clean(
                    r.get("direct_cas_final")
                ),
                "embedded_cas_candidates": ";".join(
                    embedded
                ),
                "identity_truth_type": identity_truth_type,
                "identity_input_mode": identity_input_mode,

                # Candidate rule pairing is known by design only for the benchmark.
                "candidate_rule_pairing": (
                    "CONTROLLED_KNOWN_BY_DESIGN"
                ),
                "reference_identity_membership": True,
                "reference_regulated": gold,
                "source_text": clean(
                    r.get("source_text")
                ),
            })

    return pd.DataFrame(rows)


def automation_binary_metrics(
    pred: pd.Series,
    gold: pd.Series,
):
    p = pred.astype(bool)
    g = gold.astype(bool)

    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    tn = int((~p & ~g).sum())
    n = tp + fp + fn + tn

    return {
        "n": n,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "accuracy": (
            (tp + tn) / n
            if n else np.nan
        ),
        "precision": (
            tp / (tp + fp)
            if (tp + fp) else np.nan
        ),
        "recall": (
            tp / (tp + fn)
            if (tp + fn) else np.nan
        ),
        "false_safe_rate": (
            fn / (tp + fn)
            if (tp + fn) else np.nan
        ),
        "over_screening_rate": (
            fp / (fp + tn)
            if (fp + tn) else np.nan
        ),
    }


def evaluate_automation_boundary(
    challenge: pd.DataFrame,
):
    """
    세 screening strategy를 동일 challenge cases에서 비교한다.

    SYSTEM A: CAS_ONLY
      - direct CAS match만 보고 자동 YES/NO
      - concentration 무시
      - review 없음

    SYSTEM B: DETERMINISTIC_DIRECT_RULE
      - direct CAS + concentration
      - non-direct scope는 비대상으로 처리
      - review 없음

    SYSTEM C: HYBRID_SELECTIVE
      - direct CAS exact match: deterministic auto
      - AI가 BROAD_SALT_SCOPE라고 판정하고,
        rule text에 명시된 parent CAS가 inventory CAS와 exact match:
        AI-assisted + deterministic threshold auto
      - 그 외 non-direct/ambiguous/AI error:
        REVIEW_REQUIRED (abstention)

    핵심:
    Proposed system의 목표는 모든 것을 자동 YES/NO로 만드는 것이 아니라
    unsafe automatic decision을 줄이면서 사람이 볼 사례를 선택적으로 남기는 것이다.
    """
    if challenge is None or not len(challenge):
        return pd.DataFrame()

    x = challenge.copy()

    # ------------------------------
    # A. CAS only
    # ------------------------------
    def direct_cas_match(r):
        direct = set(
            splitsemi(
                r.get("direct_cas_final")
            )
        )
        cas = normalize_cas_value(
            r.get("inventory_cas")
        )
        return bool(
            cas and cas in direct
        )

    x["cas_only_identity_match"] = x.apply(
        direct_cas_match,
        axis=1,
    )

    x["CAS_ONLY_prediction"] = (
        x["cas_only_identity_match"]
        .astype(bool)
    )
    x["CAS_ONLY_disposition"] = "AUTO"

    # ------------------------------
    # B. deterministic direct rule
    # ------------------------------
    x["DETERMINISTIC_prediction"] = (
        x["cas_only_identity_match"].astype(bool)
        & (
            pd.to_numeric(
                x["concentration_pct"],
                errors="coerce",
            )
            >= pd.to_numeric(
                x["threshold_pct"],
                errors="coerce",
            )
        )
    )
    x["DETERMINISTIC_disposition"] = "AUTO"

    # ------------------------------
    # C. hybrid selective
    # ------------------------------
    hybrid_pred = []
    hybrid_disp = []
    hybrid_reason = []

    for _, r in x.iterrows():
        conc = float(r["concentration_pct"])
        thr = float(r["threshold_pct"])

        if bool(r["cas_only_identity_match"]):
            hybrid_pred.append(
                conc >= thr
            )
            hybrid_disp.append(
                "AUTO_DETERMINISTIC_DIRECT_CAS"
            )
            hybrid_reason.append(
                "Direct CAS exact match + deterministic concentration threshold"
            )
            continue

        if clean(r.get("ai_status")) != "OK":
            hybrid_pred.append(
                None
            )
            hybrid_disp.append(
                "REVIEW_REQUIRED"
            )
            hybrid_reason.append(
                "Local LLM output unavailable or invalid"
            )
            continue

        ai_scope = clean(
            r.get("ai_scope_type")
        )

        inv_cas = normalize_cas_value(
            r.get("inventory_cas")
        )

        embedded = set(
            splitsemi(
                r.get("embedded_cas_candidates")
            )
        )

        # Safety gate:
        # AI classification alone is insufficient.
        # Parent CAS must also be explicitly present in the regulatory text.
        deterministic_broad_salt_text = any(
            re.search(
                p,
                clean(r.get("source_text")),
                flags=re.I,
            )
            for p in BROAD_SALT_PATTERNS
        )

        if (
            ai_scope == "BROAD_SALT_SCOPE"
            and deterministic_broad_salt_text
            and inv_cas
            and inv_cas in embedded
        ):
            hybrid_pred.append(
                conc >= thr
            )
            hybrid_disp.append(
                "AUTO_AI_ASSISTED_EXPLICIT_PARENT"
            )
            hybrid_reason.append(
                "LLM broad-salt scope + deterministic broad-salt text gate "
                "+ explicit parent CAS match + threshold"
            )
        else:
            hybrid_pred.append(
                None
            )
            hybrid_disp.append(
                "REVIEW_REQUIRED"
            )

            if ai_scope in {
                "MIXTURE_OR_REACTION",
                "STRUCTURAL_RANGE",
                "GENERIC_GROUP_OR_DERIVATIVE",
                "SALT_FAMILY",
                "OTHER",
            }:
                reason = (
                    f"Complex scope ({ai_scope}) requires identity/context validation"
                )
            else:
                reason = (
                    "Non-direct identity not supported by safe automatic evidence"
                )

            hybrid_reason.append(reason)

    x["HYBRID_prediction"] = hybrid_pred
    x["HYBRID_disposition"] = hybrid_disp
    x["HYBRID_reason"] = hybrid_reason

    # outcome labels
    gold = x["reference_regulated"].astype(bool)

    for system, col in [
        ("CAS_ONLY", "CAS_ONLY_prediction"),
        ("DETERMINISTIC", "DETERMINISTIC_prediction"),
    ]:
        p = x[col].astype(bool)

        x[f"{system}_outcome"] = np.where(
            p == gold,
            "CORRECT_AUTO",
            np.where(
                (~p) & gold,
                "FALSE_SAFE_AUTO",
                "OVER_SCREEN_AUTO",
            ),
        )

    def hybrid_outcome(r):
        if r["HYBRID_disposition"] == "REVIEW_REQUIRED":
            return "SAFE_ESCALATION_REVIEW"

        pred = bool(
            r["HYBRID_prediction"]
        )
        ref = bool(
            r["reference_regulated"]
        )

        if pred == ref:
            return "CORRECT_AUTO"
        if (not pred) and ref:
            return "FALSE_SAFE_AUTO"
        return "OVER_SCREEN_AUTO"

    x["HYBRID_outcome"] = x.apply(
        hybrid_outcome,
        axis=1,
    )

    return x


def summarize_automation_boundary(
    cases: pd.DataFrame,
):
    """
    자동화율과 안전성의 trade-off를 요약한다.

    review case는 정답으로 간주하지 않는다.
    대신 자동결정에서 제외되어 사람이 확인해야 하는 workload로 계산한다.
    """
    if cases is None or not len(cases):
        return pd.DataFrame()

    rows = []
    gold = cases["reference_regulated"].astype(bool)

    # fully automatic baselines
    for label, pred_col in [
        ("CAS_ONLY", "CAS_ONLY_prediction"),
        (
            "DETERMINISTIC_DIRECT_RULE",
            "DETERMINISTIC_prediction",
        ),
    ]:
        m = automation_binary_metrics(
            cases[pred_col],
            gold,
        )

        rows.append({
            "system": label,
            "n_cases": len(cases),
            "n_automated": len(cases),
            "n_review": 0,
            "automation_coverage": 1.0,
            "review_burden": 0.0,
            "selective_accuracy_on_automated": m["accuracy"],
            "false_safe_rate_among_gold_positive": m[
                "false_safe_rate"
            ],
            "over_screening_rate_among_gold_negative": m[
                "over_screening_rate"
            ],
            "unsafe_auto_n": m["FN"] + m["FP"],
            "unsafe_auto_rate_all_cases": (
                (m["FN"] + m["FP"]) / len(cases)
            ),
            "interpretation": (
                "All cases forced to automatic YES/NO."
            ),
        })

    # selective hybrid
    auto = cases[
        cases["HYBRID_disposition"]
        .ne("REVIEW_REQUIRED")
    ].copy()

    review = cases[
        cases["HYBRID_disposition"]
        .eq("REVIEW_REQUIRED")
    ].copy()

    if len(auto):
        m = automation_binary_metrics(
            auto["HYBRID_prediction"],
            auto["reference_regulated"],
        )
        selective_acc = m["accuracy"]
        auto_fn = m["FN"]
        auto_fp = m["FP"]
    else:
        m = {
            "false_safe_rate": np.nan,
            "over_screening_rate": np.nan,
        }
        selective_acc = np.nan
        auto_fn = 0
        auto_fp = 0

    # false-safe denominator remains all gold-positive challenge cases.
    total_gold_pos = int(
        gold.sum()
    )
    total_gold_neg = int(
        (~gold).sum()
    )

    unsafe_false_safe = int(
        (
            cases["HYBRID_outcome"]
            == "FALSE_SAFE_AUTO"
        ).sum()
    )

    unsafe_over = int(
        (
            cases["HYBRID_outcome"]
            == "OVER_SCREEN_AUTO"
        ).sum()
    )

    rows.append({
        "system": "HYBRID_SELECTIVE_LOCAL_LLM_RULE",
        "n_cases": len(cases),
        "n_automated": len(auto),
        "n_review": len(review),
        "automation_coverage": (
            len(auto) / len(cases)
        ),
        "review_burden": (
            len(review) / len(cases)
        ),
        "selective_accuracy_on_automated": selective_acc,
        "false_safe_rate_among_gold_positive": (
            unsafe_false_safe / total_gold_pos
            if total_gold_pos
            else np.nan
        ),
        "over_screening_rate_among_gold_negative": (
            unsafe_over / total_gold_neg
            if total_gold_neg
            else np.nan
        ),
        "unsafe_auto_n": (
            unsafe_false_safe
            + unsafe_over
        ),
        "unsafe_auto_rate_all_cases": (
            (
                unsafe_false_safe
                + unsafe_over
            )
            / len(cases)
        ),
        "interpretation": (
            "Automates only when deterministic identity evidence is sufficient; "
            "otherwise abstains to targeted review."
        ),
    })

    return pd.DataFrame(rows)


def automation_boundary_by_scope(
    cases: pd.DataFrame,
):
    if cases is None or not len(cases):
        return pd.DataFrame()

    rows = []

    for scope, g in cases.groupby(
        "ref_scope_type"
    ):
        auto = g[
            g["HYBRID_disposition"]
            .ne("REVIEW_REQUIRED")
        ]
        review = g[
            g["HYBRID_disposition"]
            .eq("REVIEW_REQUIRED")
        ]

        if len(auto):
            auto_correct = int(
                (
                    auto["HYBRID_outcome"]
                    == "CORRECT_AUTO"
                ).sum()
            )
            selective_acc = (
                auto_correct / len(auto)
            )
        else:
            selective_acc = np.nan

        rows.append({
            "ref_scope_type": scope,
            "n_cases": len(g),
            "n_automated": len(auto),
            "n_review": len(review),
            "automation_coverage": (
                len(auto) / len(g)
            ),
            "review_burden": (
                len(review) / len(g)
            ),
            "selective_accuracy_on_automated": selective_acc,
            "unsafe_auto_n": int(
                g["HYBRID_outcome"]
                .isin([
                    "FALSE_SAFE_AUTO",
                    "OVER_SCREEN_AUTO",
                ])
                .sum()
            ),
            "ai_scope_agreement_in_source_sample": (
                float(
                    g[
                        "ai_scope_type"
                    ].astype(str)
                    .eq(
                        g[
                            "ref_scope_type"
                        ].astype(str)
                    )
                    .mean()
                )
            ),
        })

    return pd.DataFrame(rows)


def build_rule_universe_routing(
    semantic_ref: pd.DataFrame,
    scored: pd.DataFrame,
):
    """
    최신 active regulatory rule universe를 세 routing class로 나눈다.

    DETERMINISTIC_AUTO_CANDIDATE:
      direct CAS가 있어 numeric applicability를 rule engine으로 처리 가능

    AI_ASSISTED_PARENT_SCOPE_CANDIDATE:
      non-direct broad-salt rule이면서 LLM도 broad salt로 판정하고,
      본문에 explicit parent CAS가 존재

    REVIEW_REQUIRED:
      나머지 extended/ambiguous scope

    이 표는 실제 기업 workload가 아니라 rule-level automation potential이다.
    """
    if semantic_ref is None or not len(semantic_ref):
        return pd.DataFrame()

    ai_cols = [
        "rule_id",
        "ai_status",
        "ai_scope_type",
    ]

    ai = (
        scored[ai_cols]
        .drop_duplicates("rule_id")
        if scored is not None
        and len(scored)
        else pd.DataFrame(columns=ai_cols)
    )

    x = semantic_ref.merge(
        ai,
        on="rule_id",
        how="left",
    )

    rows = []

    for _, r in x.iterrows():
        has_direct = bool(
            r.get("has_direct_cas")
        )

        source = clean(
            r.get("source_text")
        )

        embedded = embedded_cas_list(
            source
        )

        if has_direct:
            route = (
                "DETERMINISTIC_AUTO_CANDIDATE"
            )
            reason = (
                "Direct CAS available; threshold remains deterministic."
            )
        else:
            ai_ok = (
                clean(r.get("ai_status"))
                == "OK"
            )
            ai_scope = clean(
                r.get("ai_scope_type")
            )

            broad_text = any(
                re.search(
                    p,
                    source,
                    flags=re.I,
                )
                for p in BROAD_SALT_PATTERNS
            )

            if (
                ai_ok
                and ai_scope
                == "BROAD_SALT_SCOPE"
                and broad_text
                and embedded
            ):
                route = (
                    "AI_ASSISTED_PARENT_SCOPE_CANDIDATE"
                )
                reason = (
                    "LLM broad-salt classification + deterministic phrase gate "
                    "+ explicit parent CAS."
                )
            else:
                route = "REVIEW_REQUIRED"
                reason = (
                    "No direct CAS or safe automatic identity evidence."
                )

        rows.append({
            "rule_id": r.get("rule_id", ""),
            "designation_id": r.get(
                "designation_id",
                "",
            ),
            "substance_name_ko": r.get(
                "substance_name_ko",
                "",
            ),
            "ref_scope_type": r.get(
                "ref_scope_type",
                "",
            ),
            "has_direct_cas": has_direct,
            "ai_status": r.get(
                "ai_status",
                "",
            ),
            "ai_scope_type": r.get(
                "ai_scope_type",
                "",
            ),
            "routing_class": route,
            "routing_reason": reason,
            "source_text": source,
        })

    return pd.DataFrame(rows)


def summarize_rule_universe_routing(
    routing: pd.DataFrame,
):
    if routing is None or not len(routing):
        return pd.DataFrame()

    vc = (
        routing["routing_class"]
        .value_counts()
    )

    rows = []

    for route, n in vc.items():
        rows.append({
            "routing_class": route,
            "n_rules": int(n),
            "pct_rules": float(
                n / len(routing)
            ),
        })

    rows.append({
        "routing_class": "TOTAL",
        "n_rules": int(len(routing)),
        "pct_rules": 1.0,
    })

    return pd.DataFrame(rows)


def build_update_rule_routing(
    update_diff: pd.DataFrame,
    rule_routing: pd.DataFrame,
):
    """
    실제 최신 개정에서 바뀐 designation이 어느 정도 자동 처리 후보인지 계산.
    """
    if (
        update_diff is None
        or not len(update_diff)
    ):
        return pd.DataFrame()

    changed = update_diff[
        update_diff[
            "update_change_type"
        ].ne("UNCHANGED")
    ].copy()

    if (
        rule_routing is None
        or not len(rule_routing)
    ):
        changed["automation_route"] = (
            "REVIEW_REQUIRED"
        )
        return changed

    route_by_did = []

    rank = {
        "DETERMINISTIC_AUTO_CANDIDATE": 1,
        "AI_ASSISTED_PARENT_SCOPE_CANDIDATE": 2,
        "REVIEW_REQUIRED": 3,
    }

    for did, g in rule_routing.groupby(
        "designation_id"
    ):
        routes = g[
            "routing_class"
        ].tolist()

        # designation 내부 sub-rule 중 하나라도 review가 필요하면
        # 해당 designation 전체를 review 쪽으로 보수적으로 라우팅.
        selected = max(
            routes,
            key=lambda z: rank.get(z, 9),
        )

        route_by_did.append({
            "designation_id": str(did),
            "automation_route": selected,
            "rule_scope_types": ";".join(
                sorted(
                    set(
                        g[
                            "ref_scope_type"
                        ].astype(str)
                    )
                )
            ),
        })

    route_df = pd.DataFrame(
        route_by_did
    )

    out = changed.merge(
        route_df,
        on="designation_id",
        how="left",
    )

    out["automation_route"] = (
        out["automation_route"]
        .fillna(
            "DETERMINISTIC_AUTO_CANDIDATE"
        )
    )

    return out


def summarize_update_rule_routing(
    update_routing: pd.DataFrame,
):
    if (
        update_routing is None
        or not len(update_routing)
    ):
        return pd.DataFrame()

    x = (
        update_routing[
            "automation_route"
        ]
        .value_counts()
        .rename_axis(
            "automation_route"
        )
        .reset_index(
            name="n_changed_designations"
        )
    )

    x["pct_changed_designations"] = (
        x["n_changed_designations"]
        / len(update_routing)
    )

    return x


def append_automation_boundary_sheets(
    path: Path,
    challenge: pd.DataFrame,
    cases: pd.DataFrame,
    summary: pd.DataFrame,
    by_scope: pd.DataFrame,
    routing: pd.DataFrame,
    routing_summary: pd.DataFrame,
    update_routing: pd.DataFrame,
    update_routing_summary: pd.DataFrame,
):
    """
    Automation boundary 연구 결과를 workbook에 추가한다.
    """
    policy = pd.DataFrame([
        {
            "system": "CAS_ONLY",
            "automatic_decision": "All cases",
            "review_policy": "No abstention",
            "identity_evidence": "Parsed direct CAS only",
            "concentration": "Ignored",
        },
        {
            "system": "DETERMINISTIC_DIRECT_RULE",
            "automatic_decision": "All cases",
            "review_policy": "No abstention",
            "identity_evidence": "Parsed direct CAS only",
            "concentration": "Applied",
        },
        {
            "system": "HYBRID_SELECTIVE_LOCAL_LLM_RULE",
            "automatic_decision": (
                "Direct-CAS cases; explicit-parent broad-salt cases "
                "passing LLM + deterministic safety gate"
            ),
            "review_policy": (
                "Abstain on mixture/reaction, generic group, salt family, "
                "structural range, unsupported non-direct identity, or AI error"
            ),
            "identity_evidence": (
                "Direct CAS OR LLM broad-salt + deterministic phrase gate "
                "+ explicit parent CAS"
            ),
            "concentration": "Applied deterministically",
        },
    ])

    sheets = [
        ("Automation_policy", policy),
        ("Challenge_inventory", challenge),
        ("Automation_boundary_cases", cases),
        ("Automation_boundary_summary", summary),
        ("Automation_by_scope", by_scope),
        ("Rule_universe_routing", routing),
        ("Rule_routing_summary", routing_summary),
        ("Update_rule_routing", update_routing),
        ("Update_routing_summary", update_routing_summary),
    ]

    with pd.ExcelWriter(
        path,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:

        for name, df in sheets:
            if df is None:
                df = pd.DataFrame()

            df.to_excel(
                writer,
                sheet_name=name[:31],
                index=False,
            )

        for ws in writer.book.worksheets:
            try:
                autosize_sheet(ws)
            except Exception:
                pass




# ============================================================
# 8-2. Cross-regime generalizability
# ============================================================

# ============================================================
# 8-2A. Official API ingestion for all major chemical regimes
# ============================================================
def _normalize_notice_title(text: str) -> str:
    return re.sub(r"[\s·ㆍ\-–—]", "", clean(text)).lower()


def _title_matches(candidate: str, target: str, query_terms) -> bool:
    c = _normalize_notice_title(candidate)
    t = _normalize_notice_title(target)
    if c == t:
        return True
    return all(_normalize_notice_title(term) in c for term in query_terms)


def query_current_admrul_by_title(spec: dict):
    """
    국가법령정보 공동활용 API에서 각 규제범주의 현행 행정규칙을 조회한다.
    인증값 자체는 출력/저장하지 않는다.
    """
    if not AUTO_FETCH_CROSS_REGIME_API:
        return {}, pd.DataFrame([{
            "regime": spec["regime"],
            "status": "API_FETCH_DISABLED",
            "title": spec["title"],
            "message": "AUTO_FETCH_CROSS_REGIME_API=0",
        }])

    oc = get_law_oc()
    if not oc:
        return {}, pd.DataFrame([{
            "regime": spec["regime"],
            "status": "SKIPPED_NO_LAW_OC",
            "title": spec["title"],
            "message": "국가법령정보 공동활용 API 인증정보를 찾지 못함",
        }])

    try:
        import requests

        r = requests.get(
            "https://www.law.go.kr/DRF/lawSearch.do",
            params={
                "OC": oc,
                "target": "admrul",
                "type": "JSON",
                "nw": 1,
                "search": 1,
                "query": spec["title"],
                "display": 100,
            },
            timeout=45,
        )
        r.raise_for_status()
        data = r.json()

        candidates = []
        for d in _recursive_dicts(data):
            title = clean(d.get("행정규칙명") or d.get("행정규칙제목") or "")
            serial = clean(d.get("행정규칙일련번호") or d.get("행정규칙 일련번호") or "")
            if not title or not serial:
                continue
            if not _title_matches(title, spec["title"], spec.get("query_terms", [])):
                continue
            label = _extract_notice_label(d)
            candidates.append({
                "label": label,
                "serial": serial,
                "title": title,
                "issue_date": clean(d.get("발령일자")),
                "effective_date": clean(d.get("시행일자")),
                "revision_type": clean(d.get("제개정구분명")),
                "ministry": clean(d.get("소관부처명")),
            })

        if not candidates:
            return {}, pd.DataFrame([{
                "regime": spec["regime"],
                "status": "OFFICIAL_QUERY_NO_MATCH",
                "title": spec["title"],
                "message": "현행 행정규칙 검색 결과에서 대상 고시를 찾지 못함",
            }])

        candidates = sorted(candidates, key=lambda x: notice_key(x.get("label", "")))
        best = candidates[-1]

        return best, pd.DataFrame([{
            "regime": spec["regime"],
            "status": "OFFICIAL_NOTICE_FOUND",
            **best,
            "message": "현행 행정규칙 API 검색 성공",
        }])

    except Exception as e:
        return {}, pd.DataFrame([{
            "regime": spec["regime"],
            "status": "OFFICIAL_CHECK_FAILED",
            "title": spec["title"],
            "message": f"{type(e).__name__}: {e}",
        }])


def download_cross_regime_notice(info: dict, spec: dict):
    """현행 규제 원문 JSON과 첨부파일을 regime별 폴더에 저장."""
    if not info or not clean(info.get("serial")):
        return [], pd.DataFrame()

    oc = get_law_oc()
    if not oc:
        return [], pd.DataFrame([{
            "regime": spec["regime"],
            "notice": clean(info.get("label")),
            "status": "SKIPPED_NO_LAW_OC",
        }])

    try:
        import requests
        import json as _json
    except Exception as e:
        return [], pd.DataFrame([{
            "regime": spec["regime"],
            "notice": clean(info.get("label")),
            "status": f"IMPORT_ERROR: {type(e).__name__}: {e}",
        }])

    regime_dir = CROSS_REGIME_API_RAW_DIR / spec["regime"]
    regime_dir.mkdir(parents=True, exist_ok=True)

    label = clean(info.get("label")) or "CURRENT"
    serial = clean(info.get("serial"))
    saved = []
    manifest = []

    try:
        r = requests.get(
            "https://www.law.go.kr/DRF/lawService.do",
            params={
                "OC": oc,
                "target": "admrul",
                "ID": serial,
                "type": "JSON",
            },
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()

        json_path = regime_dir / f"{spec['regime']}_{label}_raw.json"
        json_path.write_text(
            _json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        saved.append(json_path)
        manifest.append({
            "regime": spec["regime"],
            "notice": label,
            "serial": serial,
            "file_type": "JSON",
            "local_file": str(json_path),
            "status": "DOWNLOADED",
        })

        svc = data.get("AdmRulService", {}) or {}
        attach = svc.get("첨부파일", {}) or {}
        links = attach.get("첨부파일링크", []) or []
        names = attach.get("첨부파일명", []) or []
        if isinstance(links, str):
            links = [links]
        if isinstance(names, str):
            names = [names]

        for i, link in enumerate(links):
            raw_name = names[i] if i < len(names) else f"attachment_{i+1}"
            safe_name = re.sub(r'[\\/:*?"<>|]', "_", str(raw_name)).strip()
            fpath = regime_dir / f"{spec['regime']}_{label}_{safe_name}"
            url = str(link)
            if not url.startswith("http"):
                url = "https://www.law.go.kr" + url

            try:
                fr = requests.get(url, timeout=120)
                fr.raise_for_status()
                fpath.write_bytes(fr.content)
                saved.append(fpath)
                manifest.append({
                    "regime": spec["regime"],
                    "notice": label,
                    "serial": serial,
                    "file_type": fpath.suffix.lower().lstrip(".").upper() or "ATTACHMENT",
                    "local_file": str(fpath),
                    "status": "DOWNLOADED",
                })
            except Exception as e:
                manifest.append({
                    "regime": spec["regime"],
                    "notice": label,
                    "serial": serial,
                    "file_type": "ATTACHMENT",
                    "local_file": str(fpath),
                    "status": f"DOWNLOAD_ERROR: {type(e).__name__}: {e}",
                })

    except Exception as e:
        manifest.append({
            "regime": spec["regime"],
            "notice": label,
            "serial": serial,
            "file_type": "",
            "local_file": "",
            "status": f"SERVICE_ERROR: {type(e).__name__}: {e}",
        })

    return saved, pd.DataFrame(manifest)


def _iter_recursive_strings(obj, path="root"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_recursive_strings(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _iter_recursive_strings(v, f"{path}[{i}]")
    elif isinstance(obj, (str, int, float)):
        t = clean(obj)
        if t:
            yield path, t


def _infer_restricted_or_prohibited(text: str, current: str = "") -> str:
    t = re.sub(r"\s+", "", clean(text))
    # More specific category indicators first.
    has_prohibited = "금지물질" in t or "금지대상" in t
    has_restricted = "제한물질" in t or "제한대상" in t
    if has_prohibited and not has_restricted:
        return "PROHIBITED"
    if has_restricted and not has_prohibited:
        return "RESTRICTED"
    return current


_PERCENT_COND_RE = re.compile(
    r"(?P<op>이상|초과|이하|미만|>=|>|<=|<)?\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?:%|퍼센트)"
)

_QUANTITY_COND_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>kg|킬로그램|ton|tons|톤|t)\b",
    flags=re.I,
)


def _extract_percent_condition(text: str):
    vals = []
    raw = []
    for m in _PERCENT_COND_RE.finditer(clean(text)):
        try:
            vals.append(float(m.group("value")))
            raw.append((m.group("op") or "") + m.group("value") + "%")
        except Exception:
            pass
    if not vals:
        return np.nan, ""
    # A single common applicability field needs a conservative screening
    # threshold; all original expressions remain in *_raw and source_text.
    return min(vals), ";".join(dict.fromkeys(raw))


def _extract_quantity_condition(text: str):
    m = _QUANTITY_COND_RE.search(clean(text))
    if not m:
        return "", ""
    return m.group("value"), m.group("unit")


def _extract_use_condition(text: str) -> str:
    t = clean(text)
    if re.search(r"(?:용도|사용용도|용도로|용으로|사용에|사용을|제조·수입·사용|제조ㆍ수입ㆍ사용)", t):
        return t
    return ""


def _extract_temporal_condition(text: str) -> str:
    t = clean(text)
    if re.search(r"(?:시행일|시행한다|유예|경과조치|부터\s*시행|까지)", t):
        return t
    if re.search(r"\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일", t):
        return t
    return ""


def _extract_authorization_condition(text: str, regime: str) -> str:
    t = clean(text)
    if regime == "AUTHORIZED" and re.search(r"(?:허가|허가면제|허가를\s*받)", t):
        return t
    if regime == "RESTRICTED" and re.search(r"(?:제한|제조|수입|사용|판매)", t):
        return t
    if regime == "PROHIBITED" and re.search(r"(?:금지|제조|수입|사용|판매)", t):
        return t
    return ""


def _guess_substance_name(text: str, cas_list) -> str:
    t = clean(text)
    if not t:
        return ""
    first_cas = cas_list[0] if cas_list else ""
    before = t.split(first_cas, 1)[0] if first_cas and first_cas in t else t
    parts = [p.strip() for p in re.split(r"\||\t|\n", before) if p.strip()]
    # walk backwards to the nearest descriptive token before the CAS
    for p in reversed(parts):
        q = re.sub(r"^\s*(?:제?\d+(?:[-–.]\d+)*[.)]?|[가-하][.)])\s*", "", p).strip()
        q = re.sub(r"CAS\s*(?:No\.?|RN)?\s*[:：]?\s*$", "", q, flags=re.I).strip()
        if not q:
            continue
        if len(q) > 180:
            continue
        if re.fullmatch(r"[\d\W_]+", q):
            continue
        if any(x in q for x in ["제한물질의 지정", "금지물질의 지정", "사고대비물질의 지정", "허가물질 지정 등에 관한 규정"]):
            continue
        return q
    return ""


def _stable_cross_rule_id(regime: str, cas_text: str, source_text: str) -> str:
    import hashlib
    normalized_source = re.sub(r"\s+", " ", clean(source_text))
    token = f"{regime}|{cas_text}|{normalized_source}"
    return "API_" + hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _make_api_rule_record(
    regime: str,
    text: str,
    locator: str,
    info: dict,
    source_file: str,
):
    cas_list = list(dict.fromkeys(CAS_RE.findall(clean(text))))
    if not cas_list:
        return None

    threshold, threshold_raw = _extract_percent_condition(text)
    qty, qty_unit = _extract_quantity_condition(text)
    direct_cas = ";".join(cas_list)
    name = _guess_substance_name(text, cas_list)
    rid = _stable_cross_rule_id(regime, direct_cas, text)

    return {
        "regime": regime,
        "rule_id": rid,
        "designation_id": rid,
        "substance_name": name,
        "direct_cas": direct_cas,
        "source_text": re.sub(r"\s+", " ", clean(text)).strip(),
        "concentration_threshold_pct": threshold,
        "concentration_threshold_raw": threshold_raw,
        "use_condition": _extract_use_condition(text),
        "quantity_threshold": qty,
        "quantity_unit": qty_unit,
        "temporal_condition": _extract_temporal_condition(text),
        "authorization_condition": _extract_authorization_condition(text, regime),
        "source_name": clean(info.get("title")),
        "source_reference": clean(info.get("label")),
        "source_locator": locator,
        "official_serial": clean(info.get("serial")),
        "official_issue_date": clean(info.get("issue_date")),
        "official_effective_date": clean(info.get("effective_date")),
        "extraction_status": "AUTO_EXTRACTED_FROM_OFFICIAL_API_NEEDS_TRACEABILITY_AUDIT",
        "input_source_file": source_file,
    }


def _extract_rules_from_tabular(path: Path, spec: dict, info: dict):
    records = []
    issues = []
    suffix = path.suffix.lower()
    try:
        if suffix in {".xlsx", ".xls"}:
            sheets = pd.read_excel(path, sheet_name=None, dtype=str)
        elif suffix == ".csv":
            sheets = {"CSV": pd.read_csv(path, dtype=str, encoding_errors="ignore")}
        else:
            return records, issues
    except Exception as e:
        return records, [{
            "regime": spec["regime"],
            "file": str(path),
            "status": f"TABULAR_READ_ERROR: {type(e).__name__}: {e}",
        }]

    for sheet_name, df in sheets.items():
        current = ""
        if spec.get("split_mode") == "RESTRICTED_PROHIBITED":
            current = _infer_restricted_or_prohibited(sheet_name, "")
        else:
            current = spec["regime"]

        # Include column names because some official tables place category
        # wording in merged/header columns rather than ordinary data cells.
        header_text = " | ".join(map(str, df.columns))
        if spec.get("split_mode") == "RESTRICTED_PROHIBITED":
            current = _infer_restricted_or_prohibited(header_text, current)

        for idx, row in df.iterrows():
            vals = [
                str(v).strip()
                for v in row.values
                if pd.notna(v) and str(v).strip()
            ]
            if not vals:
                continue
            text = " | ".join(vals)

            if spec.get("split_mode") == "RESTRICTED_PROHIBITED":
                current = _infer_restricted_or_prohibited(text, current)
                if current not in {"RESTRICTED", "PROHIBITED"} and CAS_RE.search(text):
                    # Ambiguous combined-notice row: retain for manual routing
                    regime = "RESTRICTED_PROHIBITED_REVIEW"
                else:
                    regime = current
            else:
                regime = spec["regime"]

            if CAS_RE.search(text):
                rec = _make_api_rule_record(
                    regime=regime,
                    text=text,
                    locator=f"{path.name}:{sheet_name}:row_{idx+2}",
                    info=info,
                    source_file=str(path),
                )
                if rec:
                    records.append(rec)

    return records, issues


def _extract_rules_from_json(path: Path, spec: dict, info: dict):
    import json as _json
    try:
        data = _json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return [], [{
            "regime": spec["regime"],
            "file": str(path),
            "status": f"JSON_READ_ERROR: {type(e).__name__}: {e}",
        }]

    records = []
    current = "" if spec.get("split_mode") == "RESTRICTED_PROHIBITED" else spec["regime"]

    for loc, text in _iter_recursive_strings(data):
        if spec.get("split_mode") == "RESTRICTED_PROHIBITED":
            current = _infer_restricted_or_prohibited(text, current)
            regime = current if current in {"RESTRICTED", "PROHIBITED"} else "RESTRICTED_PROHIBITED_REVIEW"
        else:
            regime = spec["regime"]

        if not CAS_RE.search(text):
            continue
        rec = _make_api_rule_record(
            regime=regime,
            text=text,
            locator=loc,
            info=info,
            source_file=str(path),
        )
        if rec:
            records.append(rec)

    return records, []


def parse_cross_regime_downloads(files, spec: dict, info: dict):
    """
    Official table attachments are preferred over JSON strings because they
    preserve row-level regulatory context more faithfully. JSON is used as a
    fallback when no parseable XLS/XLSX/CSV rule rows are available.
    """
    tabular_records = []
    json_records = []
    issues = []
    unsupported = []

    for p0 in files:
        p = Path(p0)
        suffix = p.suffix.lower()
        if suffix in {".xlsx", ".xls", ".csv"}:
            recs, iss = _extract_rules_from_tabular(p, spec, info)
            tabular_records.extend(recs)
            issues.extend(iss)
        elif suffix == ".json":
            recs, iss = _extract_rules_from_json(p, spec, info)
            json_records.extend(recs)
            issues.extend(iss)
        elif suffix in {".hwp", ".hwpx", ".pdf", ".doc", ".docx"}:
            unsupported.append(str(p))

    records = tabular_records if tabular_records else json_records
    df = pd.DataFrame(records)

    if len(df):
        for c in CROSS_REGIME_RULE_COLUMNS:
            if c not in df.columns:
                df[c] = ""
        # Avoid duplicate copies from repeated API/attachment contexts.
        df["_norm_text"] = df["source_text"].fillna("").astype(str).map(
            lambda x: re.sub(r"\s+", " ", x).strip().lower()
        )
        df = df.drop_duplicates(
            subset=["regime", "direct_cas", "_norm_text"],
            keep="first",
        ).drop(columns=["_norm_text"]).reset_index(drop=True)

    status_rows = list(issues)
    status_rows.append({
        "regime": spec["regime"],
        "file": "",
        "status": (
            "API_RULES_PARSED_TABULAR"
            if tabular_records
            else "API_RULES_PARSED_JSON_FALLBACK"
            if json_records
            else "NO_CAS_RULES_EXTRACTED"
        ),
        "n_rule_rows": len(df),
        "unsupported_attachment_n": len(unsupported),
        "message": (
            "Tabular official attachment used"
            if tabular_records
            else "Official API JSON used because no parseable tabular rules were found"
            if json_records
            else "CAS-bearing regulatory rows were not automatically extracted; inspect official attachments"
        ),
    })

    for f in unsupported:
        status_rows.append({
            "regime": spec["regime"],
            "file": f,
            "status": "ORIGINAL_ATTACHMENT_RETAINED_FOR_TRACEABILITY",
            "n_rule_rows": np.nan,
            "unsupported_attachment_n": np.nan,
            "message": "HWP/HWPX/PDF/DOC is retained as source evidence; parser relies on API JSON/tabular attachment",
        })

    return df, pd.DataFrame(status_rows)


def load_official_cross_regime_rules_from_api():
    all_rules = []
    status_parts = []
    manifest_parts = []

    for spec in CROSS_REGIME_NOTICE_SPECS:
        info, query_status = query_current_admrul_by_title(spec)
        status_parts.append(query_status)
        if not info:
            continue

        files, manifest = download_cross_regime_notice(info, spec)
        if len(manifest):
            manifest_parts.append(manifest)

        parsed, parse_status = parse_cross_regime_downloads(files, spec, info)
        status_parts.append(parse_status)
        if len(parsed):
            all_rules.append(parsed)

    rules = pd.concat(all_rules, ignore_index=True, sort=False) if all_rules else pd.DataFrame()
    status = pd.concat(status_parts, ignore_index=True, sort=False) if status_parts else pd.DataFrame()
    manifest = pd.concat(manifest_parts, ignore_index=True, sort=False) if manifest_parts else pd.DataFrame()

    # Split-mode safety: unresolved combined rows are never treated as
    # automatically applicable restricted/prohibited rules.
    if len(rules):
        rules["regime"] = rules["regime"].map(normalize_cross_regime)

    return rules, status, manifest


def cross_regime_input_template():
    """
    다른 regulated-chemical categories의 공식 rule data를
    동일 schema로 입력하기 위한 빈 template.

    주의:
    이 template의 행은 연구결과가 아니며, 공식 고시/법령 원문을
    구조화하여 채우기 위한 입력 형식이다.
    """
    rows = []
    for regime, label, role in CROSS_REGIME_EXPECTED:
        if regime == "HUMAN_HAZARD":
            continue
        rows.append({
            "regime": regime,
            "rule_id": "",
            "designation_id": "",
            "substance_name": "",
            "direct_cas": "",
            "source_text": "",
            "concentration_threshold_pct": "",
            "use_condition": "",
            "quantity_threshold": "",
            "quantity_unit": "",
            "temporal_condition": "",
            "authorization_condition": "",
            "source_name": label + " 관련 공식 고시/법령",
            "source_reference": "",
            "input_note": (
                "공식 규제원문을 구조화하여 입력. "
                "빈 template 행은 분석결과로 사용하지 않음."
            ),
        })
    return pd.DataFrame(rows)


def _read_cross_regime_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        sheets = pd.read_excel(path, sheet_name=None)
        parts = []
        for sheet_name, df in sheets.items():
            if df is None or not len(df):
                continue
            x = df.copy()
            x["source_sheet"] = sheet_name
            parts.append(x)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return pd.read_csv(path)


def load_external_cross_regime_rules():
    """
    Optional official cross-regime structured rule file.
    Missing inputs are not imputed or fabricated.
    """
    candidates = []

    if CROSS_REGIME_RULES_FILE:
        candidates.append(Path(CROSS_REGIME_RULES_FILE))

    candidates += [
        CROSS_REGIME_DIR / "cross_regime_rules.xlsx",
        CROSS_REGIME_DIR / "cross_regime_rules.csv",
        SCRIPT_DIR / "cross_regime_rules.xlsx",
        SCRIPT_DIR / "cross_regime_rules.csv",
    ]

    for p in candidates:
        if not p.exists():
            continue

        try:
            df = _read_cross_regime_file(p)
        except Exception as e:
            return pd.DataFrame(), pd.DataFrame([{
                "status": "CROSS_REGIME_INPUT_READ_ERROR",
                "input_file": str(p),
                "message": f"{type(e).__name__}: {e}",
            }])

        for c in CROSS_REGIME_RULE_COLUMNS:
            if c not in df.columns:
                df[c] = ""

        # Empty template rows must never become empirical results.
        substantive = (
            df["source_text"].fillna("").astype(str).str.strip().ne("")
            | df["direct_cas"].fillna("").astype(str).str.strip().ne("")
            | df["substance_name"].fillna("").astype(str).str.strip().ne("")
        )
        df = df[substantive].copy()

        if not len(df):
            return pd.DataFrame(), pd.DataFrame([{
                "status": "CROSS_REGIME_INPUT_EMPTY",
                "input_file": str(p),
                "message": "파일은 있으나 substantive official rule row가 없음",
            }])

        df["input_source_file"] = str(p)
        return df, pd.DataFrame([{
            "status": "CROSS_REGIME_INPUT_LOADED",
            "input_file": str(p),
            "n_rows": len(df),
            "message": "외부 regulated-category rule data loaded",
        }])

    return pd.DataFrame(), pd.DataFrame([{
        "status": "CROSS_REGIME_INPUT_REQUIRED",
        "input_file": "",
        "message": (
            "사고대비/제한/금지/허가물질의 공식 구조화 rule file이 없음. "
            "해당 regime 결과는 생성하지 않으며 template만 제공."
        ),
    }])


def build_core_human_hazard_cross_regime(
    master: pd.DataFrame,
    latest_notice: str,
):
    """
    현재 core master를 cross-regime 공통 schema로 변환.
    HUMAN_HAZARD는 실제 core benchmark data에서 자동 생성한다.
    """
    g = substantive_rule_rows_for_notice(master, latest_notice)
    rows = []

    for _, r in g.iterrows():
        thr = min_threshold(r)
        direct = direct_cas_list_from_row(r)
        text = clean(r.get("source_text"))

        rows.append({
            "regime": "HUMAN_HAZARD",
            "rule_id": clean(r.get("rule_id")),
            "designation_id": clean(r.get("designation_id")),
            "substance_name": clean(r.get("substance_name_ko")),
            "direct_cas": ";".join(direct),
            "source_text": text,
            "concentration_threshold_pct": thr,
            "use_condition": "",
            "quantity_threshold": "",
            "quantity_unit": "",
            "temporal_condition": "",
            "authorization_condition": "",
            "source_name": REGULATION_TITLE,
            "source_reference": latest_notice,
            "input_source_file": "CORE_REGULATORY_MASTER",
        })

    return pd.DataFrame(rows)


def normalize_cross_regime(regime: str) -> str:
    t = clean(regime).upper().replace(" ", "_")
    aliases = {
        "인체등유해성물질": "HUMAN_HAZARD",
        "HUMAN_HAZARD": "HUMAN_HAZARD",
        "사고대비물질": "ACCIDENT_PREPAREDNESS",
        "ACCIDENT_PREPAREDNESS": "ACCIDENT_PREPAREDNESS",
        "제한물질": "RESTRICTED",
        "RESTRICTED": "RESTRICTED",
        "금지물질": "PROHIBITED",
        "PROHIBITED": "PROHIBITED",
        "허가물질": "AUTHORIZED",
        "AUTHORIZED": "AUTHORIZED",
        "RESTRICTED_PROHIBITED": "RESTRICTED_PROHIBITED_REVIEW",
        "RESTRICTED_PROHIBITED_REVIEW": "RESTRICTED_PROHIBITED_REVIEW",
    }
    return aliases.get(t, t)


def _nonempty(x) -> bool:
    return bool(clean(x))


def _numeric_value(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def cross_regime_condition_flags(row: pd.Series):
    """
    Applicability dimensions를 rule-level로 표시한다.

    Dedicated structured fields를 우선 사용하고, source_text heuristic은
    'condition presence candidate'를 보조적으로 찾는 데만 사용한다.
    이는 법적 applicability 판정이 아니다.
    """
    text = clean(row.get("source_text"))

    conc_val = _numeric_value(row.get("concentration_threshold_pct"))
    concentration = bool(
        np.isfinite(conc_val)
        or re.search(r"\d+(?:\.\d+)?\s*%", text)
    )

    use = bool(
        _nonempty(row.get("use_condition"))
        or re.search(r"(?:용도|사용용도|특정\s*용도|용도로)", text)
    )

    quantity = bool(
        _nonempty(row.get("quantity_threshold"))
        or re.search(
            r"\d+(?:\.\d+)?\s*(?:kg|킬로그램|톤|ton|t)\b",
            text,
            flags=re.I,
        )
    )

    temporal = bool(
        _nonempty(row.get("temporal_condition"))
        or re.search(
            r"(?:시행일|유예|경과조치|\d{4}\s*년\s*\d{1,2}\s*월)",
            text,
        )
    )

    authorization = bool(
        _nonempty(row.get("authorization_condition"))
        or re.search(r"(?:허가|승인|신고|금지|제한)", text)
    )

    return {
        "requires_identity": True,
        "requires_concentration": concentration,
        "requires_use": use,
        "requires_quantity": quantity,
        "requires_time": temporal,
        "requires_authorization": authorization,
    }


def cross_regime_identity_scope(row: pd.Series):
    direct = splitsemi(row.get("direct_cas"))
    has_direct = bool(direct)
    text = clean(row.get("source_text"))
    return semantic_reference_scope(text, has_direct), has_direct


def build_cross_regime_rule_analysis(all_rules: pd.DataFrame):
    if all_rules is None or not len(all_rules):
        return pd.DataFrame()

    rows = []

    for i, (_, r) in enumerate(all_rules.iterrows(), start=1):
        regime = normalize_cross_regime(r.get("regime"))
        scope, has_direct = cross_regime_identity_scope(r)
        flags = cross_regime_condition_flags(r)

        embedded = list(dict.fromkeys(CAS_RE.findall(clean(r.get("source_text")))))
        text = clean(r.get("source_text"))

        broad_text = broad_salt_scope(text)

        non_identity_dims = [
            name
            for name, flag in [
                ("concentration", flags["requires_concentration"]),
                ("use", flags["requires_use"]),
                ("quantity", flags["requires_quantity"]),
                ("time", flags["requires_time"]),
                ("authorization", flags["requires_authorization"]),
            ]
            if flag
        ]

        # Rule-level automation potential. This does NOT assert that the
        # enterprise has all required data.
        if has_direct:
            if any([
                flags["requires_use"],
                flags["requires_time"],
                flags["requires_authorization"],
            ]):
                route = "AI_ASSISTED_CONDITION_CANDIDATE"
                reason = (
                    "Direct identity available, but semantic/non-numeric "
                    "applicability condition requires interpretation."
                )
            else:
                route = "DETERMINISTIC_AUTO_CANDIDATE"
                reason = (
                    "Direct identity and structured/numeric applicability "
                    "conditions can be evaluated deterministically if enterprise data exist."
                )
        elif broad_text and embedded:
            route = "AI_ASSISTED_IDENTITY_CANDIDATE"
            reason = (
                "Broad-salt scope with explicit CAS in source text; "
                "semantic scope + deterministic evidence gate may support limited automation."
            )
        else:
            route = "REVIEW_REQUIRED"
            reason = (
                "No safe single-identifier evidence for complete regulatory membership."
            )

        required_fields = ["cas"]
        if flags["requires_concentration"]:
            required_fields.append("concentration_pct")
        if flags["requires_use"]:
            required_fields.append("use_description")
        if flags["requires_quantity"]:
            required_fields.append("annual_quantity_kg")
        if flags["requires_time"]:
            required_fields.append("as_of_date")
        if flags["requires_authorization"]:
            required_fields.append("authorization_status")

        rows.append({
            "regime": regime,
            "rule_id": clean(r.get("rule_id")) or f"CROSS_RULE_{i:05d}",
            "designation_id": clean(r.get("designation_id")),
            "substance_name": clean(r.get("substance_name")),
            "direct_cas": clean(r.get("direct_cas")),
            "has_direct_cas": has_direct,
            "identity_scope": scope,
            "requires_identity": flags["requires_identity"],
            "requires_concentration": flags["requires_concentration"],
            "requires_use": flags["requires_use"],
            "requires_quantity": flags["requires_quantity"],
            "requires_time": flags["requires_time"],
            "requires_authorization": flags["requires_authorization"],
            "applicability_dimensions": ";".join(["identity"] + non_identity_dims),
            "required_enterprise_fields": ";".join(required_fields),
            "rule_level_route": route,
            "routing_reason": reason,
            "source_name": clean(r.get("source_name")),
            "source_reference": clean(r.get("source_reference")),
            "source_text": text,
            "input_source_file": clean(r.get("input_source_file")),
            "interpretation_boundary": (
                "Generalizability/routing analysis only; not final legal applicability."
            ),
        })

    return pd.DataFrame(rows)


def build_cross_regime_status(
    rule_analysis: pd.DataFrame,
    external_status: pd.DataFrame,
):
    rows = []

    for regime, label, role in CROSS_REGIME_EXPECTED:
        g = (
            rule_analysis[
                rule_analysis["regime"].astype(str).eq(regime)
            ]
            if rule_analysis is not None and len(rule_analysis)
            else pd.DataFrame()
        )

        rows.append({
            "regime": regime,
            "regime_label_ko": label,
            "study_role": role,
            "n_rules_loaded": len(g),
            "data_status": (
                "CORE_MASTER_AVAILABLE"
                if regime == "HUMAN_HAZARD" and len(g)
                else "OFFICIAL_EXTENSION_DATA_LOADED"
                if len(g)
                else "INPUT_REQUIRED"
            ),
            "performance_benchmark_status": (
                "CORE_CONTROLLED_BENCHMARK"
                if regime == "HUMAN_HAZARD"
                else "GENERALIZABILITY_ONLY_NO_PERFORMANCE_BENCHMARK"
            ),
        })

    status = pd.DataFrame(rows)
    if external_status is not None and len(external_status):
        status.attrs["external_input_status"] = external_status.to_dict("records")
    return status


def summarize_cross_regime_conditions(rule_analysis: pd.DataFrame):
    if rule_analysis is None or not len(rule_analysis):
        return pd.DataFrame()

    rows = []
    for regime, g in rule_analysis.groupby("regime"):
        rows.append({
            "regime": regime,
            "n_rules": len(g),
            "identity_condition_n": int(g["requires_identity"].sum()),
            "concentration_condition_n": int(g["requires_concentration"].sum()),
            "use_condition_n": int(g["requires_use"].sum()),
            "quantity_condition_n": int(g["requires_quantity"].sum()),
            "time_condition_n": int(g["requires_time"].sum()),
            "authorization_condition_n": int(g["requires_authorization"].sum()),
            "deterministic_candidate_n": int(
                g["rule_level_route"].eq("DETERMINISTIC_AUTO_CANDIDATE").sum()
            ),
            "ai_identity_candidate_n": int(
                g["rule_level_route"].eq("AI_ASSISTED_IDENTITY_CANDIDATE").sum()
            ),
            "ai_condition_candidate_n": int(
                g["rule_level_route"].eq("AI_ASSISTED_CONDITION_CANDIDATE").sum()
            ),
            "review_required_n": int(
                g["rule_level_route"].eq("REVIEW_REQUIRED").sum()
            ),
        })
    return pd.DataFrame(rows)


def evaluate_cross_regime_enterprise_data_requirements(
    rule_analysis: pd.DataFrame,
    inventory: pd.DataFrame,
):
    """
    Direct/explicit-parent identity로 candidate pairing 가능한 경우에 한해,
    enterprise-side input completeness를 확인한다.

    missing use/quantity/time/authorization information은 AI failure가 아니라
    ADDITIONAL_ENTERPRISE_DATA_REQUIRED로 분리한다.
    """
    if rule_analysis is None or not len(rule_analysis):
        return pd.DataFrame()

    inv = inventory.copy() if inventory is not None else pd.DataFrame()
    for c in INVENTORY_COLUMNS:
        if c not in inv.columns:
            inv[c] = ""

    if not len(inv):
        return pd.DataFrame()

    rows = []

    for _, rule in rule_analysis.iterrows():
        direct = splitsemi(rule.get("direct_cas"))
        embedded = list(dict.fromkeys(CAS_RE.findall(clean(rule.get("source_text")))))

        candidate_cas = set(direct)
        if (
            rule.get("rule_level_route") == "AI_ASSISTED_IDENTITY_CANDIDATE"
            and broad_salt_scope(rule.get("source_text"))
        ):
            candidate_cas.update(embedded)

        if not candidate_cas:
            continue

        hits = inv[
            inv["cas"].map(normalize_cas_value).isin(candidate_cas)
        ].copy()

        for _, item in hits.iterrows():
            required = splitsemi(rule.get("required_enterprise_fields"))
            missing = []

            for field in required:
                val = item.get(field, "")
                if field in {"concentration_pct", "annual_quantity_kg"}:
                    if not np.isfinite(pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]):
                        missing.append(field)
                else:
                    if not clean(val):
                        missing.append(field)

            if missing:
                final_route = "ADDITIONAL_ENTERPRISE_DATA_REQUIRED"
                reason = "Missing enterprise fields: " + ";".join(missing)
            else:
                final_route = clean(rule.get("rule_level_route"))
                reason = clean(rule.get("routing_reason"))

            rows.append({
                "inventory_id": clean(item.get("inventory_id")),
                "product_name": clean(item.get("product_name")),
                "chemical_name": clean(item.get("chemical_name")),
                "inventory_cas": normalize_cas_value(item.get("cas")),
                "regime": clean(rule.get("regime")),
                "rule_id": clean(rule.get("rule_id")),
                "designation_id": clean(rule.get("designation_id")),
                "rule_level_route": clean(rule.get("rule_level_route")),
                "required_enterprise_fields": clean(rule.get("required_enterprise_fields")),
                "missing_enterprise_fields": ";".join(missing),
                "enterprise_data_route": final_route,
                "routing_reason": reason,
                "interpretation_boundary": (
                    "Data-sufficiency triage only; not final legal applicability."
                ),
            })

    return pd.DataFrame(rows)


def summarize_cross_regime_enterprise_requirements(detail: pd.DataFrame):
    if detail is None or not len(detail):
        return pd.DataFrame()

    return (
        detail.groupby(["regime", "enterprise_data_route"])
        .size()
        .rename("n_candidate_pairs")
        .reset_index()
    )


def run_cross_regime_generalizability(
    master: pd.DataFrame,
    latest_notice: str,
    inventory: pd.DataFrame,
):
    """
    Unified regulatory universe:
      1) HUMAN_HAZARD from the validated core master used in the benchmark
      2) ACCIDENT_PREPAREDNESS / RESTRICTED / PROHIBITED / AUTHORIZED
         automatically fetched from the current official administrative-rule API
      3) optional external structured file remains a supplemental/manual fallback

    No API-extracted row is treated as independent legal gold; raw source text,
    source locator and official metadata are retained for traceability.
    """
    core = build_core_human_hazard_cross_regime(master, latest_notice)

    api_ext, api_status, api_manifest = load_official_cross_regime_rules_from_api()
    file_ext, file_status = load_external_cross_regime_rules()

    parts = [core]
    if len(api_ext):
        parts.append(api_ext)
    if len(file_ext):
        x = file_ext.copy()
        x["regime"] = x["regime"].map(normalize_cross_regime)
        x["extraction_status"] = x.get(
            "extraction_status",
            pd.Series("USER_SUPPLIED_STRUCTURED_OFFICIAL_RULE", index=x.index),
        )
        parts.append(x)

    all_rules = pd.concat(parts, ignore_index=True, sort=False)
    for c in CROSS_REGIME_RULE_COLUMNS:
        if c not in all_rules.columns:
            all_rules[c] = ""

    # API takes priority over supplemental duplicate rows, but the core human
    # hazard master always remains authoritative for HUMAN_HAZARD.
    all_rules["_source_priority"] = np.where(
        all_rules["regime"].eq("HUMAN_HAZARD"), 0,
        np.where(
            all_rules["extraction_status"].astype(str).str.contains("OFFICIAL_API", na=False),
            1,
            2,
        ),
    )
    all_rules["_norm_text"] = all_rules["source_text"].fillna("").astype(str).map(
        lambda x: re.sub(r"\s+", " ", x).strip().lower()
    )
    all_rules = (
        all_rules.sort_values("_source_priority")
        .drop_duplicates(
            subset=["regime", "direct_cas", "_norm_text"],
            keep="first",
        )
        .drop(columns=["_source_priority", "_norm_text"])
        .reset_index(drop=True)
    )

    analysis = build_cross_regime_rule_analysis(all_rules)

    input_status = pd.concat(
        [
            api_status if api_status is not None else pd.DataFrame(),
            file_status if file_status is not None else pd.DataFrame(),
        ],
        ignore_index=True,
        sort=False,
    )

    status = build_cross_regime_status(analysis, input_status)
    summary = summarize_cross_regime_conditions(analysis)
    enterprise = evaluate_cross_regime_enterprise_data_requirements(
        analysis,
        inventory,
    )
    enterprise_summary = summarize_cross_regime_enterprise_requirements(enterprise)

    regime_counts = (
        all_rules.groupby("regime")
        .size()
        .rename("n_unified_rule_rows")
        .reset_index()
    ) if len(all_rules) else pd.DataFrame()

    return {
        "status": status,
        "external_input_status": input_status,
        "api_status": api_status,
        "api_download_manifest": api_manifest,
        "unified_master": all_rules,
        "unified_regime_counts": regime_counts,
        "rule_analysis": analysis,
        "condition_summary": summary,
        "enterprise_requirements": enterprise,
        "enterprise_requirements_summary": enterprise_summary,
        "input_template": cross_regime_input_template(),
    }

def append_cross_regime_sheets(path: Path, tables: dict):
    sheets = [
        ("Cross_regime_status", tables.get("status")),
        ("Cross_regime_API_status", tables.get("api_status")),
        ("Cross_regime_API_manifest", tables.get("api_download_manifest")),
        ("Cross_regime_input_status", tables.get("external_input_status")),
        ("Unified_regulatory_master", tables.get("unified_master")),
        ("Unified_regime_counts", tables.get("unified_regime_counts")),
        ("Cross_regime_rules", tables.get("rule_analysis")),
        ("Cross_regime_condition_summary", tables.get("condition_summary")),
        ("Enterprise_data_requirements", tables.get("enterprise_requirements")),
        ("Enterprise_req_summary", tables.get("enterprise_requirements_summary")),
        ("Cross_regime_input_template", tables.get("input_template")),
    ]

    with pd.ExcelWriter(
        path,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        for name, df in sheets:
            if df is None:
                df = pd.DataFrame()
            df.to_excel(
                writer,
                sheet_name=name[:31],
                index=False,
            )
        for ws in writer.book.worksheets:
            try:
                autosize_sheet(ws)
            except Exception:
                pass


# ============================================================
# 8-2. Paper figure generation
# ============================================================
def _save_paper_figure(
    fig,
    stem: str,
):
    """
    논문 제출용으로 동일 figure를 PNG(600 dpi)와 vector PDF로 저장.
    """
    png = FIGURE_DIR / f"{stem}.png"
    pdf = FIGURE_DIR / f"{stem}.pdf"

    fig.savefig(
        png,
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    return png, pdf


def figure1_framework_workflow():
    """
    Figure 1.
    Local LLM + deterministic rule + selective review framework의
    전체 연구/업무 흐름을 보여주는 schematic.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    except Exception as e:
        print(
            "[FIGURE] matplotlib unavailable:",
            type(e).__name__,
            str(e),
        )
        return None

    fig, ax = plt.subplots(
        figsize=(8.4, 9.2)
    )
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis("off")

    boxes = [
        (
            5.0,
            12.0,
            "Official regulatory versions\n(previous vs latest)",
        ),
        (
            5.0,
            10.2,
            "Regulatory change extraction\nand machine-readable rules",
        ),
        (
            2.6,
            8.1,
            "Deterministic layer\nCAS · concentration · thresholds",
        ),
        (
            7.4,
            8.1,
            "Local LLM layer\nsemantic scope · exclusions",
        ),
        (
            5.0,
            5.8,
            "Selective hybrid routing\nsafe auto-decision vs abstention",
        ),
        (
            2.6,
            3.4,
            "AUTO\nDirect / supported cases",
        ),
        (
            7.4,
            3.4,
            "REVIEW REQUIRED\ncomplex identity / ambiguity",
        ),
        (
            5.0,
            1.0,
            "Enterprise impact triage\nwith regulatory evidence",
        ),
    ]

    width = 3.9
    height = 1.0

    for x, y, text in boxes:
        patch = FancyBboxPatch(
            (
                x - width / 2,
                y - height / 2,
            ),
            width,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            fill=False,
            linewidth=1.2,
        )
        ax.add_patch(patch)
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=10,
        )

    arrows = [
        ((5.0, 11.5), (5.0, 10.7)),
        ((5.0, 9.7), (2.9, 8.6)),
        ((5.0, 9.7), (7.1, 8.6)),
        ((2.6, 7.6), (4.4, 6.3)),
        ((7.4, 7.6), (5.6, 6.3)),
        ((5.0, 5.3), (2.9, 3.9)),
        ((5.0, 5.3), (7.1, 3.9)),
        ((2.6, 2.9), (4.4, 1.5)),
        ((7.4, 2.9), (5.6, 1.5)),
    ]

    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="->",
                mutation_scale=12,
                linewidth=1.0,
            )
        )

    ax.set_title(
        "Local AI–Rule Hybrid Framework for Chemical Regulatory Triage",
        fontsize=12,
        pad=16,
    )

    paths = _save_paper_figure(
        fig,
        "Fig1_framework_workflow",
    )
    plt.close(fig)
    return paths


def figure2_automation_boundary(
    automation_summary: pd.DataFrame,
):
    """
    Figure 2.
    CAS-only, deterministic direct-rule, hybrid-selective 전략의
    automation coverage / review burden / unsafe automatic decision 비교.
    """
    if automation_summary is None or not len(
        automation_summary
    ):
        return None

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    xdf = automation_summary.copy()

    name_map = {
        "CAS_ONLY": "CAS only",
        "DETERMINISTIC_DIRECT_RULE": "CAS + rules",
        "HYBRID_SELECTIVE_LOCAL_LLM_RULE": "Hybrid selective",
    }

    xdf["display"] = (
        xdf["system"]
        .map(name_map)
        .fillna(xdf["system"])
    )

    metrics = [
        (
            "automation_coverage",
            "Automation coverage",
        ),
        (
            "review_burden",
            "Review burden",
        ),
        (
            "unsafe_auto_rate_all_cases",
            "Unsafe auto decision",
        ),
    ]

    x = np.arange(len(xdf))
    width = 0.24

    fig, ax = plt.subplots(
        figsize=(8.6, 5.6)
    )

    for i, (col, label) in enumerate(metrics):
        vals = (
            pd.to_numeric(
                xdf[col],
                errors="coerce",
            )
            .fillna(0)
            .to_numpy()
            * 100.0
        )

        bars = ax.bar(
            x + (i - 1) * width,
            vals,
            width,
            label=label,
        )

        ax.bar_label(
            bars,
            labels=[
                f"{v:.1f}%"
                for v in vals
            ],
            padding=2,
            fontsize=8,
        )

    ax.set_xticks(
        x,
        xdf["display"].tolist(),
    )
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 112)
    ax.set_title(
        "Automation Boundary Across Screening Strategies"
    )
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    paths = _save_paper_figure(
        fig,
        "Fig2_automation_boundary",
    )
    plt.close(fig)
    return paths


def figure3_scope_automation(
    automation_scope: pd.DataFrame,
):
    """
    Figure 3.
    Regulatory scope별 hybrid automation coverage와
    Local LLM scope agreement를 병렬 비교.
    """
    if automation_scope is None or not len(
        automation_scope
    ):
        return None

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    xdf = automation_scope.copy()

    xdf = xdf.sort_values(
        "automation_coverage",
        ascending=True,
    )

    y = np.arange(len(xdf))
    height = 0.36

    auto = (
        pd.to_numeric(
            xdf["automation_coverage"],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy()
        * 100
    )

    agree = (
        pd.to_numeric(
            xdf[
                "ai_scope_agreement_in_source_sample"
            ],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy()
        * 100
    )

    fig, ax = plt.subplots(
        figsize=(9.0, 6.8)
    )

    bars1 = ax.barh(
        y - height / 2,
        auto,
        height,
        label="Hybrid automation coverage",
    )

    bars2 = ax.barh(
        y + height / 2,
        agree,
        height,
        label="Local LLM scope agreement",
    )

    ax.set_yticks(
        y,
        xdf["ref_scope_type"].tolist(),
    )
    ax.set_xlabel("Rate (%)")
    ax.set_xlim(0, 108)
    ax.set_title(
        "Automation Boundary by Regulatory Scope"
    )

    ax.bar_label(
        bars1,
        labels=[
            f"{v:.1f}%"
            for v in auto
        ],
        padding=2,
        fontsize=8,
    )

    ax.bar_label(
        bars2,
        labels=[
            f"{v:.1f}%"
            for v in agree
        ],
        padding=2,
        fontsize=8,
    )

    ax.legend(
        frameon=False,
        loc="lower right",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    paths = _save_paper_figure(
        fig,
        "Fig3_scope_automation_boundary",
    )
    plt.close(fig)
    return paths


def figure4_changed_rule_routing(
    update_routing_summary: pd.DataFrame,
):
    """
    Figure 4.
    실제 규제버전 변경에서 changed designations가
    deterministic / AI-assisted / review로 어떻게 routing되는지 표시.
    """
    if update_routing_summary is None or not len(
        update_routing_summary
    ):
        return None

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    xdf = update_routing_summary.copy()

    label_map = {
        "DETERMINISTIC_AUTO_CANDIDATE": "Deterministic",
        "AI_ASSISTED_PARENT_SCOPE_CANDIDATE": "AI-assisted",
        "REVIEW_REQUIRED": "Review required",
    }

    xdf["display"] = (
        xdf["automation_route"]
        .map(label_map)
        .fillna(
            xdf["automation_route"]
        )
    )

    order = [
        "Deterministic",
        "AI-assisted",
        "Review required",
    ]

    xdf["_order"] = (
        xdf["display"]
        .map(
            {
                k: i
                for i, k in enumerate(order)
            }
        )
        .fillna(99)
    )

    xdf = xdf.sort_values(
        "_order"
    )

    counts = pd.to_numeric(
        xdf["n_changed_designations"],
        errors="coerce",
    ).fillna(0)

    pct = pd.to_numeric(
        xdf["pct_changed_designations"],
        errors="coerce",
    ).fillna(0) * 100

    fig, ax = plt.subplots(
        figsize=(7.4, 5.2)
    )

    bars = ax.bar(
        xdf["display"],
        counts,
    )

    ax.bar_label(
        bars,
        labels=[
            f"{int(n)}\n({p:.1f}%)"
            for n, p in zip(
                counts,
                pct,
            )
        ],
        padding=3,
        fontsize=9,
    )

    ax.set_ylabel(
        "Number of changed designations"
    )
    ax.set_title(
        "Routing of Changed Regulatory Designations"
    )
    ax.set_ylim(
        0,
        max(counts.max() * 1.18, 5),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    paths = _save_paper_figure(
        fig,
        "Fig4_changed_rule_routing",
    )
    plt.close(fig)
    return paths



def figure5_generalized_applicability_architecture():
    """
    Figure 5. Cross-regime generalized applicability architecture.
    Conceptual schematic; not an empirical performance figure.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    dimensions = [
        (1.2, 5.5, "Identity\nCAS · salts · groups"),
        (3.1, 5.5, "Concentration\nthreshold"),
        (5.0, 5.5, "Use\ncondition"),
        (6.9, 5.5, "Quantity\ncondition"),
        (8.8, 5.5, "Temporal\ncondition"),
        (10.7, 5.5, "Authorization\nstatus"),
    ]

    for x, y, text in dimensions:
        patch = FancyBboxPatch(
            (x - 0.8, y - 0.55),
            1.6,
            1.1,
            boxstyle="round,pad=0.03,rounding_size=0.07",
            fill=False,
            linewidth=1.1,
        )
        ax.add_patch(patch)
        ax.text(x, y, text, ha="center", va="center", fontsize=9)
        ax.add_patch(FancyArrowPatch(
            (x, 4.9),
            (6.0, 3.55),
            arrowstyle="->",
            mutation_scale=10,
            linewidth=0.9,
        ))

    center = FancyBboxPatch(
        (4.4, 2.55),
        3.2,
        1.2,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        fill=False,
        linewidth=1.3,
    )
    ax.add_patch(center)
    ax.text(
        6.0,
        3.15,
        "Regulatory applicability\ninformation requirements",
        ha="center",
        va="center",
        fontsize=10,
    )

    routes = [
        (2.0, 1.0, "Deterministic\nautomation"),
        (4.7, 1.0, "AI-assisted\ninterpretation"),
        (7.4, 1.0, "Additional enterprise\ndata required"),
        (10.1, 1.0, "Human\nreview"),
    ]

    for x, y, text in routes:
        patch = FancyBboxPatch(
            (x - 1.0, y - 0.5),
            2.0,
            1.0,
            boxstyle="round,pad=0.03,rounding_size=0.07",
            fill=False,
            linewidth=1.1,
        )
        ax.add_patch(patch)
        ax.text(x, y, text, ha="center", va="center", fontsize=9)
        ax.add_patch(FancyArrowPatch(
            (6.0, 2.5),
            (x, 1.55),
            arrowstyle="->",
            mutation_scale=10,
            linewidth=0.9,
        ))

    ax.set_title(
        "Generalized Applicability Architecture Across Chemical Regulatory Regimes",
        fontsize=12,
        pad=12,
    )

    paths = _save_paper_figure(fig, "Fig5_cross_regime_applicability_architecture")
    plt.close(fig)
    return paths


def generate_paper_figures(
    automation_summary: pd.DataFrame,
    automation_scope: pd.DataFrame,
    update_routing_summary: pd.DataFrame,
):
    """
    논문용 핵심 figure 5개 생성 및 figure manifest 반환.
    """
    records = []

    figure_specs = [
        (
            "Figure 1",
            figure1_framework_workflow,
            (),
            (
                "Overview of the proposed Local LLM–deterministic rule "
                "hybrid framework. Explicit numeric conditions are processed "
                "deterministically, semantic regulatory scope is parsed by a "
                "Local LLM, and unsupported or ambiguous identity cases are "
                "selectively escalated for review."
            ),
        ),
        (
            "Figure 2",
            figure2_automation_boundary,
            (automation_summary,),
            (
                "Comparison of automation coverage, review burden, and unsafe "
                "automatic decisions across CAS-only, deterministic direct-rule, "
                "and selective hybrid screening strategies in the controlled "
                "automation-boundary challenge."
            ),
        ),
        (
            "Figure 3",
            figure3_scope_automation,
            (automation_scope,),
            (
                "Hybrid automation coverage and Local LLM scope agreement by "
                "regulatory scope type. Automation coverage reflects the full "
                "hybrid routing logic and should not be interpreted as pure "
                "LLM accuracy."
            ),
        ),
        (
            "Figure 4",
            figure4_changed_rule_routing,
            (update_routing_summary,),
            (
                "Routing of designations changed between the analyzed regulatory "
                "versions into deterministic automation candidates, AI-assisted "
                "parent-scope candidates, and review-required cases."
            ),
        ),
        (
            "Figure 5",
            figure5_generalized_applicability_architecture,
            (),
            (
                "Generalized applicability architecture for extending the framework "
                "beyond the core human-hazard test bed. Regulatory applicability may "
                "depend on identity, concentration, use, quantity, temporal, and "
                "authorization information; missing enterprise-side information is "
                "separated from model uncertainty."
            ),
        ),
    ]

    for fig_no, fn, args, caption in figure_specs:
        try:
            out = fn(*args)

            if out is None:
                records.append({
                    "figure": fig_no,
                    "png_path": "",
                    "pdf_path": "",
                    "status": "SKIPPED_NO_DATA_OR_MATPLOTLIB",
                    "suggested_caption": caption,
                })
            else:
                png, pdf = out

                records.append({
                    "figure": fig_no,
                    "png_path": str(png),
                    "pdf_path": str(pdf),
                    "status": "CREATED",
                    "suggested_caption": caption,
                })

        except Exception as e:
            records.append({
                "figure": fig_no,
                "png_path": "",
                "pdf_path": "",
                "status": (
                    "FAILED: "
                    + type(e).__name__
                    + ": "
                    + str(e)
                ),
                "suggested_caption": caption,
            })

    manifest = pd.DataFrame(
        records
    )

    manifest.to_csv(
        FIGURE_DIR / "Figure_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return manifest


def append_figure_manifest_sheet(
    path: Path,
    manifest: pd.DataFrame,
):
    if manifest is None:
        manifest = pd.DataFrame()

    with pd.ExcelWriter(
        path,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        manifest.to_excel(
            writer,
            sheet_name="Figure_manifest",
            index=False,
        )

        for ws in writer.book.worksheets:
            try:
                autosize_sheet(ws)
            except Exception:
                pass


def append_ai_research_sheets(
    path: Path,
    sample: pd.DataFrame,
    scored: pd.DataFrame,
    vs_rule: pd.DataFrame,
    by_scope: pd.DataFrame,
    disagreements: pd.DataFrame,
    metadata: pd.DataFrame,
):
    """
    전문가 panel을 필수 전제로 하지 않는 논문용 AI 결과 저장.
    AI_vs_rule은 agreement임을 명시한다.
    """
    with pd.ExcelWriter(
        path,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:

        sheets = [
            ("AI_semantic_sample", sample),
            ("AI_semantic_results", scored),
            ("AI_vs_rule_agreement", vs_rule),
            ("AI_by_scope", by_scope),
            ("AI_disagreements", disagreements),
            ("AI_run_metadata", metadata),
        ]

        for name, df in sheets:
            if df is None:
                df = pd.DataFrame()

            df.to_excel(
                writer,
                sheet_name=name[:31],
                index=False,
            )

        for ws in writer.book.worksheets:
            try:
                autosize_sheet(ws)
            except Exception:
                pass


# ============================================================
# 8. Integrated main
# ============================================================


# Split-pipeline core module: no integrated main.
