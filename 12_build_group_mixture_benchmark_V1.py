# -*- coding: utf-8 -*-
"""Study 2 extension V1: build source-anchored chemical-group and mixture benchmarks.

Regulatory basis
----------------
Korea, Designation of Restricted/Prohibited Substances, effective 2026-01-01,
Annex 3: specific list of restricted substances designated by generic scope.

This builder intentionally creates a conservative validation set:
- group cases use CAS entries explicitly enumerated in Annex 3;
- negative cases are cross-group chemicals explicitly enumerated under a different scope;
- mixture cases vary concentration around the exact legal threshold/operator.

It is therefore a source-anchored *closed-registry* validation benchmark, not an
open-world test of every possible compound that could fall under a catch-all
phrase such as "other lead compounds".
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
INTER = ROOT / "intermediate_gm_v1"
DATA.mkdir(parents=True, exist_ok=True)
INTER.mkdir(parents=True, exist_ok=True)

SOURCE_VERSION = "기후에너지환경부고시 제2025-28호, 시행 2026-01-01"
SOURCE_URL = (
    "https://www.law.go.kr/LSW/flDownload.do?bylClsCd=200201&"
    "flNm=%5B%EB%B3%84%ED%91%9C+3%5D+%EC%B4%9D%EC%B9%AD%EC%9C%BC%EB%A1%9C+"
    "%EC%A7%80%EC%A0%95%EB%90%9C+%EC%A0%9C%ED%95%9C%EB%AC%BC%EC%A7%88%EC%9D%98+"
    "%EA%B5%AC%EC%B2%B4%EC%A0%81+%EB%AA%A9%EB%A1%9D%28%EC%A0%9C3%EC%A1%B0%EC%A0%9C2%ED%95%AD+%EA%B4%80%EB%A0%A8%29&"
    "flSeq=158305589"
)

RULES = {
    "MG_SALTS": {
        "regulatory_id": "06-5-1",
        "group_name": "Malachite green salts",
        "scope_text": "말라카이트 그린[Malachite green]의 염류 및 그 중 하나를 0.1% 이상 함유한 혼합물",
        "threshold_pct": 0.1,
        "threshold_operator": ">=",
        "members": [
            ("Malachite green", "10309-95-2"),
            ("Malachite green phosphotungstomolybdate", "61725-50-6"),
            ("Malachite green phosphomolybdate", "68083-41-0"),
            ("Malachite green chloride", "569-64-2"),
            ("Malachite green oxalate", "2437-29-8"),
            ("Malachite green oxalate", "18015-76-4"),
        ],
    },
    "TBT_TRIALKYLTIN": {
        "regulatory_id": "06-5-4",
        "group_name": "Trialkyl tin hydroxide salts / tributyltin compounds",
        "scope_text": "수산화 트리알킬주석[Trialkyl tin hydroxide]과 그 염류(산화 트리알킬주석을 포함한다) 및 트리부틸주석화합물[Tributyl tin compound], 또는 그 중 하나를 0.1% 이상 함유한 혼합물",
        "threshold_pct": 0.1,
        "threshold_operator": ">=",
        "members": [
            ("Bis(tributyltin) phthalate", "4782-29-0"),
            ("Tributyltin hydroxide", "1067-97-6"),
            ("Tributyltin salicylate", "4342-30-7"),
            ("Tributyltin oxide; bis(tributyltin)oxide; TBTO", "56-35-9"),
            ("Tributyltin chloride; TBTCL", "1461-22-9"),
            ("Tributyltin fluoride; TBTF", "1983-10-4"),
            ("Tributyltin hydride", "688-73-3"),
            ("Tributyltin laurate", "3090-36-6"),
            ("Tributyltin maleate", "4027-18-3"),
            ("Tributyltin methacrylate", "2155-70-6"),
            ("Octyl acrylate-methyl methacrylate-tributyltin methacrylate copolymer", "67772-01-4"),
            ("Bis(tributyltin) meso-2,3-dibromosuccinate", "31732-71-5"),
        ],
    },
    "NONYLPHENOL_NPE": {
        "regulatory_id": "06-5-6",
        "group_name": "Nonylphenols / nonylphenol ethoxylates",
        "scope_text": "노닐페놀[Nonylphenols, Nonylphenol ethoxylates] 및 이를 0.1% 이상 함유한 혼합물",
        "threshold_pct": 0.1,
        "threshold_operator": ">=",
        "members": [
            ("Nonylphenol", "25154-52-3"),
            ("4-Nonylphenol", "104-40-5"),
            ("4-Nonylphenol, branched", "84852-15-3"),
            ("3-Nonylphenol", "139-84-4"),
            ("2-Nonylphenol", "136-83-4"),
            ("Nonylphenol polyethylene glycol ether; nonylphenol ethoxylate", "9016-45-9"),
            ("Branched nonylphenol", "90481-04-2"),
            ("Isononylphenol", "11066-49-2"),
            ("Nonylphenol ethoxylates", "27177-05-5"),
            ("alpha-(Nonylphenyl)-omega-hydroxy poly(oxy-1,2-ethanediyl), branched", "68412-54-4"),
            ("alpha-(Nonylphenyl)-omega-hydroxy poly(oxy-1,2-ethanediyl), branched phosphates", "68412-53-3"),
            ("alpha-(Isononylphenyl)-omega-hydroxy poly(oxy-1,2-ethanediyl)", "37205-87-1"),
            ("alpha-(4-Nonylphenyl)-omega-hydroxy poly(oxy-1,2-ethanediyl)", "26027-38-3"),
            ("alpha-(4-Nonylphenyl)-omega-hydroxy poly(oxy-1,2-ethanediyl), branched", "127087-87-0"),
        ],
    },
    "LEAD_COMPOUNDS": {
        "regulatory_id": "06-5-8",
        "group_name": "Lead and lead compounds",
        "scope_text": "납[Lead; 7439-92-1]과 그 화합물[Lead compounds] 및 이를 0.009% 초과 함유한 혼합물",
        "threshold_pct": 0.009,
        "threshold_operator": ">",
        "members": [
            ("Lead", "7439-92-1"),
            ("C.I. Pigment red 104; Lead chromate molybdate sulfate red", "12656-85-8"),
            ("C.I. Pigment yellow 34; Lead sulfochromate yellow", "1344-37-2"),
            ("Dioxobis(stearato)trilead; Bis(octadecanoato)dioxotrilead", "12578-12-0"),
            ("Fatty acids, C16-18, lead salts", "91031-62-8"),
            ("Lead 2,4,6-trinitroresorcinoxide", "15245-44-0"),
            ("Lead 2-ethylhexanoate", "301-08-6"),
            ("Lead acetate", "301-04-2"),
            ("Lead acetate", "1335-32-6"),
            ("Lead azide", "13424-46-9"),
            ("Lead bis(tetrafluoroborate)", "13814-96-5"),
            ("Lead chromate (PbCrO4)", "7758-97-6"),
            ("Lead dinitrate", "10099-74-8"),
            ("Lead monoxide", "1317-36-8"),
            ("Lead oxide (Orange lead)", "1314-41-6"),
            ("Lead oxide phosphonate", "12141-20-7"),
            ("Lead oxide sulfate", "12202-17-4"),
            ("Lead stearate", "1072-35-1"),
            ("Lead sulfate, tetrabasic", "52732-72-6"),
            ("Pentalead tetraoxide sulfate", "12065-90-6"),
        ],
    },
    "CRVI_COMPOUNDS": {
        "regulatory_id": "06-5-10",
        "group_name": "Chromium(VI) compounds",
        "scope_text": "크로뮴(6+)화합물[Chromium(6+) compounds] 및 이를 0.1% 이상 함유한 혼합물",
        "threshold_pct": 0.1,
        "threshold_operator": ">=",
        "members": [
            ("Ammonium chromate", "7788-98-9"),
            ("Chromium(III) chromate", "24613-89-6"),
            ("Chromium(VI) oxide; chromium trioxide", "1333-82-0"),
            ("Chromyl dichloride", "14977-61-8"),
            ("Copper chromate", "13548-42-0"),
            ("Dichromic acid", "13530-68-2"),
            ("Lead(II) chromate", "7758-97-6"),
            ("Lithium chromate", "14307-35-8"),
            ("Magnesium chromate", "13423-61-5"),
            ("Potassium chlorochromate", "16037-50-6"),
            ("Ammonium dichromate", "7789-09-5"),
            ("Potassium chromate", "7789-00-6"),
            ("Potassium dichromate", "7778-50-9"),
            ("Sodium chromate", "7775-11-3"),
            ("Sodium dichromate", "10588-01-9"),
            ("Strontium chromate", "7789-06-2"),
            ("Zinc chromate", "13530-65-9"),
            ("Zinc dichromate", "14018-95-2"),
            ("Barium chromate", "10294-40-3"),
            ("Calcium chromate", "13765-19-0"),
            ("Calcium dichromate", "14307-33-6"),
            ("Lead chromate molybdate sulfate red", "12656-85-8"),
            ("C.I. Pigment Yellow 36", "37300-23-5"),
            ("Bis(triphenylsilyl) chromate", "1624-02-8"),
            ("Chromic acid (H2Cr2O7) compd. with pyridine (1:2)", "20039-37-6"),
            ("Chromic acid", "7738-94-5"),
            ("Chromic acid (H2CrO4), cobalt(2+) salt (1:1)", "13455-25-9"),
            ("Chromium(6+) compounds", "18540-29-9"),
            ("Dicesium chromate", "13454-78-9"),
            ("Dichromic acid sodium salt", "34493-01-1"),
            ("Diiron tris(chromate)", "10294-52-7"),
            ("Dipotassium heptadecaoxotetrazincate tetrachromate(2-)", "12433-50-0"),
            ("Dirubidium dichromate", "13446-73-6"),
            ("Dithallium chromate", "13473-75-1"),
            ("Dithallium dichromate", "13453-35-5"),
            ("Lead chromate", "11119-70-3"),
            ("Lead chromate oxide (Pb2(CrO4)O)", "18454-12-1"),
            ("Lithium chromate dihydrate", "7789-01-7"),
            ("Mercury dichromate", "7789-10-8"),
            ("Nickel chromate", "14721-18-7"),
            ("Nickel dichromate", "15586-38-6"),
            ("Lead sulfochromate yellow", "1344-37-2"),
            ("Potassium hydroxyoctaoxodizincatedichromate(1-)", "11103-86-9"),
            ("Chromic acid, potassium zinc salt; Potassium zinc chromate", "41189-36-0"),
            ("Silver chromate", "7784-01-2"),
            ("Silver dichromate", "7784-02-3"),
            ("Sodium chromate tetrahydrate", "10034-82-9"),
            ("Chromic acid (H2Cr2O7), sodium salt, hydrate (1:2:2)", "7789-12-0"),
            ("Pentazinc chromate octahydroxide", "49663-84-5"),
        ],
    },
}

POSITIVE_SELECTION = {
    "MG_SALTS": ["10309-95-2", "569-64-2", "2437-29-8", "61725-50-6", "68083-41-0"],
    "TBT_TRIALKYLTIN": ["1067-97-6", "56-35-9", "1461-22-9", "1983-10-4", "688-73-3"],
    "NONYLPHENOL_NPE": ["25154-52-3", "104-40-5", "84852-15-3", "139-84-4", "136-83-4"],
    "LEAD_COMPOUNDS": ["7439-92-1", "301-04-2", "7758-97-6", "10099-74-8", "1317-36-8"],
    "CRVI_COMPOUNDS": ["7788-98-9", "1333-82-0", "7789-09-5", "7778-50-9", "7775-11-3"],
}

NEGATIVE_SELECTION = {
    "MG_SALTS": ["56-35-9", "104-40-5", "10099-74-8", "7778-50-9", "1317-36-8"],
    "TBT_TRIALKYLTIN": ["569-64-2", "104-40-5", "301-04-2", "7789-09-5", "136-83-4"],
    "NONYLPHENOL_NPE": ["2437-29-8", "1461-22-9", "10099-74-8", "7775-11-3", "56-35-9"],
    "LEAD_COMPOUNDS": ["569-64-2", "56-35-9", "104-40-5", "7778-50-9", "1461-22-9"],
    "CRVI_COMPOUNDS": ["569-64-2", "56-35-9", "104-40-5", "10099-74-8", "1317-36-8"],
}

MIXTURE_POSITIVE_MEMBERS = {
    "MG_SALTS": ["569-64-2", "2437-29-8"],
    "TBT_TRIALKYLTIN": ["56-35-9", "1461-22-9"],
    "NONYLPHENOL_NPE": ["104-40-5", "25154-52-3"],
    "LEAD_COMPOUNDS": ["10099-74-8", "1317-36-8"],
    "CRVI_COMPOUNDS": ["7778-50-9", "7775-11-3"],
}


def cas_map() -> dict[str, dict]:
    out = {}
    for rid, r in RULES.items():
        for name, cas in r["members"]:
            out.setdefault(cas, {"name": name, "groups": []})["groups"].append(rid)
    return out


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def truth_from_threshold(conc: float, threshold: float, op: str) -> str:
    if op == ">=":
        return "MATCH" if conc >= threshold - 1e-12 else "NO_MATCH"
    if op == ">":
        return "MATCH" if conc > threshold + 1e-12 else "NO_MATCH"
    raise ValueError(op)


def main() -> None:
    cmap = cas_map()

    rule_rows = []
    for rid, r in RULES.items():
        for idx, (name, cas) in enumerate(r["members"], start=1):
            rule_rows.append({
                "rule_id": rid,
                "regulatory_id": r["regulatory_id"],
                "group_name": r["group_name"],
                "scope_text": r["scope_text"],
                "threshold_pct": r["threshold_pct"],
                "threshold_operator": r["threshold_operator"],
                "official_member_index": idx,
                "member_name": name,
                "member_cas": cas,
                "source_version": SOURCE_VERSION,
                "source_url": SOURCE_URL,
                "provenance": "OFFICIAL_ANNEX3_ENUMERATION",
            })
    assert len(rule_rows) == 101, len(rule_rows)
    write_csv(DATA / "regulatory_group_rules_v1.csv", rule_rows, list(rule_rows[0]))

    group_rows = []
    for rid, r in RULES.items():
        for label, selection in [("MATCH", POSITIVE_SELECTION[rid]), ("NO_MATCH", NEGATIVE_SELECTION[rid])]:
            for cas in selection:
                rec = cmap[cas]
                group_rows.append({
                    "case_id": f"G{len(group_rows)+1:03d}",
                    "task_type": "CHEMICAL_GROUP",
                    "rule_id": rid,
                    "regulatory_id": r["regulatory_id"],
                    "regulatory_scope_text": r["scope_text"],
                    "candidate_name": rec["name"],
                    "candidate_cas": cas,
                    "reference_membership": label,
                    "difficulty": "OFFICIAL_MEMBER" if label == "MATCH" else "CROSS_GROUP_HARD_NEGATIVE",
                    "candidate_official_groups": ";".join(sorted(rec["groups"])),
                    "truth_basis": "Annex 3 target-group enumeration" if label == "MATCH" else "Annex 3 enumeration under other target group(s)",
                    "source_version": SOURCE_VERSION,
                    "source_url": SOURCE_URL,
                })
    assert len(group_rows) == 50
    assert sum(r["reference_membership"] == "MATCH" for r in group_rows) == 25
    assert sum(r["reference_membership"] == "NO_MATCH" for r in group_rows) == 25
    write_csv(DATA / "group_membership_v1_50.csv", group_rows, list(group_rows[0]))

    mix_rows = []
    group_keys = list(RULES)
    for gi, rid in enumerate(group_keys):
        r = RULES[rid]
        t = float(r["threshold_pct"]); op = r["threshold_operator"]
        a, b = MIXTURE_POSITIVE_MEMBERS[rid]
        other_rules = [x for x in group_keys if x != rid]
        neg1 = POSITIVE_SELECTION[other_rules[0]][0]
        neg2 = POSITIVE_SELECTION[other_rules[1]][1]
        if op == ">":
            plan = [
                (a, 0.50*t), (a, 0.99*t), (a, 1.00*t), (a, 1.01*t), (a, 2.00*t),
                (b, 1.00*t), (b, 1.01*t), (b, 2.00*t), (b, 4.00*t),
                (neg1, 10.00*t),
            ]
        else:
            plan = [
                (a, 0.50*t), (a, 0.99*t), (a, 1.00*t), (a, 1.01*t), (a, 2.00*t),
                (b, 0.99*t), (b, 1.00*t), (b, 1.01*t),
                (neg1, 10.00*t), (neg2, 20.00*t),
            ]
        for cas, conc in plan:
            rec = cmap[cas]
            in_scope = rid in rec["groups"]
            truth = truth_from_threshold(conc, t, op) if in_scope else "NO_MATCH"
            relation = "IN_SCOPE_COMPONENT" if in_scope else "OUT_OF_SCOPE_COMPONENT"
            boundary = "AT_THRESHOLD" if abs(conc - t) <= 1e-12 else ("BELOW_THRESHOLD" if conc < t else "ABOVE_THRESHOLD")
            mix_rows.append({
                "case_id": f"M{len(mix_rows)+1:03d}",
                "task_type": "MIXTURE_THRESHOLD",
                "rule_id": rid,
                "regulatory_id": r["regulatory_id"],
                "regulatory_scope_text": r["scope_text"],
                "component_name": rec["name"],
                "component_cas": cas,
                "component_concentration_pct": round(conc, 6),
                "threshold_pct": t,
                "threshold_operator": op,
                "reference_membership": truth,
                "component_relation": relation,
                "boundary_class": boundary,
                "truth_basis": "Official Annex 3 group membership plus exact threshold/operator",
                "source_version": SOURCE_VERSION,
                "source_url": SOURCE_URL,
            })
    assert len(mix_rows) == 50
    assert sum(r["reference_membership"] == "MATCH" for r in mix_rows) == 25
    assert sum(r["reference_membership"] == "NO_MATCH" for r in mix_rows) == 25
    write_csv(DATA / "mixture_threshold_v1_50.csv", mix_rows, list(mix_rows[0]))

    manifest_rows = [{
        "dataset": "regulatory_group_rules_v1.csv",
        "n_rows": len(rule_rows),
        "scope": "101 explicit CAS entries across five generic restricted-substance scopes",
        "source_version": SOURCE_VERSION,
        "source_url": SOURCE_URL,
    }, {
        "dataset": "group_membership_v1_50.csv",
        "n_rows": len(group_rows),
        "scope": "25 MATCH + 25 cross-group NO_MATCH",
        "source_version": SOURCE_VERSION,
        "source_url": SOURCE_URL,
    }, {
        "dataset": "mixture_threshold_v1_50.csv",
        "n_rows": len(mix_rows),
        "scope": "25 MATCH + 25 NO_MATCH; includes >= versus > boundary cases",
        "source_version": SOURCE_VERSION,
        "source_url": SOURCE_URL,
    }]
    write_csv(DATA / "group_mixture_source_manifest_v1.csv", manifest_rows, list(manifest_rows[0]))

    qc = {
        "source_version": SOURCE_VERSION,
        "source_url": SOURCE_URL,
        "n_rules": len(RULES),
        "n_official_member_rows": len(rule_rows),
        "n_group_cases": len(group_rows),
        "group_match": sum(r["reference_membership"] == "MATCH" for r in group_rows),
        "group_no_match": sum(r["reference_membership"] == "NO_MATCH" for r in group_rows),
        "n_mixture_cases": len(mix_rows),
        "mixture_match": sum(r["reference_membership"] == "MATCH" for r in mix_rows),
        "mixture_no_match": sum(r["reference_membership"] == "NO_MATCH" for r in mix_rows),
        "lead_threshold_operator": RULES["LEAD_COMPOUNDS"]["threshold_operator"],
        "lead_threshold_pct": RULES["LEAD_COMPOUNDS"]["threshold_pct"],
        "other_threshold_operator": ">=",
        "other_threshold_pct": 0.1,
        "benchmark_design": "SOURCE_ANCHORED_CLOSED_REGISTRY_V1",
        "limitation": "Does not claim open-world coverage of catch-all 'other compounds' beyond explicit Annex 3 CAS entries.",
    }
    for p in [DATA / "regulatory_group_rules_v1.csv", DATA / "group_membership_v1_50.csv", DATA / "mixture_threshold_v1_50.csv"]:
        qc[p.name + "_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    (INTER / "12_group_mixture_benchmark_qc.json").write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(qc, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
