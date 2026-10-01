STUDY 2 EXTENSION — CHEMICAL GROUP + MIXTURE V1
================================================

Purpose
-------
Extend the parent/salt identity experiment to two additional regulatory tasks:
1) chemical-group membership;
2) mixture threshold determination.

Regulatory basis
----------------
Designation of Restricted/Prohibited Substances
(기후에너지환경부고시 제2025-28호; effective 2026-01-01), Annex 3.
The benchmark uses five generic restricted-substance scopes:
- 06-5-1 Malachite green salts
- 06-5-4 Trialkyl tin hydroxide salts / tributyltin compounds
- 06-5-6 Nonylphenols / nonylphenol ethoxylates
- 06-5-8 Lead and lead compounds
- 06-5-10 Chromium(VI) compounds

Dataset design
--------------
Step 12 creates:

data/regulatory_group_rules_v1.csv
- 101 explicitly enumerated CAS rows from Annex 3 across the five scopes.

data/group_membership_v1_50.csv
- 50 cases: 25 MATCH + 25 NO_MATCH.
- MATCH cases are explicit official members.
- NO_MATCH cases are cross-group chemicals explicitly listed under a different scope.

data/mixture_threshold_v1_50.csv
- 50 cases: 25 MATCH + 25 NO_MATCH.
- Tests group identity plus concentration threshold.
- Explicitly tests boundary semantics:
  * Lead: > 0.009% (exactly 0.009% is NO_MATCH)
  * Other four groups: >= 0.1% (exactly 0.1% is MATCH)

data/group_mixture_source_manifest_v1.csv
- source/provenance summary for the generated benchmark files.

Important limitation
--------------------
V1 is a source-anchored closed-registry benchmark. It does NOT claim open-world
coverage of every possible material falling under catch-all phrases such as
"other lead compounds" or "other chromium(VI) compounds". Open-set chemical-
group generalization should be evaluated separately as a V2 stress test.

Systems
-------
1. LLM + PubChem
   - receives the written scope, authoritative Annex-3 member registry,
     candidate/component CAS/name, PubChem record, and concentration if relevant.
2. Hybrid + PubChem
   - first uses the structured official registry and exact threshold operator;
   - only unresolved cases are routed to the LLM.

Execution order
---------------
1) Build/rebuild benchmark
   python 12_build_group_mixture_benchmark_V1.py

2) Freeze protocol
   python 13_freeze_group_mixture_V1.py

3) Dry run (NO Claude charge)
   Remove-Item Env:GROUP_MIX_V1_EXECUTE_CLAUDE -ErrorAction SilentlyContinue
   python 14_run_group_mixture_LLM_HYBRID_V1.py

   Inspect:
   intermediate_gm_v1/14_gm_hybrid_gate_preflight.csv
   intermediate_gm_v1/14_gm_prompt_manifest.csv
   intermediate_gm_v1/14_gm_run_metadata.json

4) Paid run
   $env:GROUP_MIX_V1_EXECUTE_CLAUDE="1"
   python 14_run_group_mixture_LLM_HYBRID_V1.py
   Remove-Item Env:GROUP_MIX_V1_EXECUTE_CLAUDE -ErrorAction SilentlyContinue

5) Aggregate results
   python 15_make_group_mixture_results_V1.py

Primary outputs
---------------
intermediate_gm_v1/14_gm_system_predictions_caselevel.csv
intermediate_gm_v1/14_gm_efficiency.csv
intermediate_gm_v1/15_gm_performance.csv
intermediate_gm_v1/15_gm_performance_by_rule.csv
intermediate_gm_v1/15_gm_mixture_boundary_performance.csv
intermediate_gm_v1/15_gm_failure_or_review_cases.csv

Validation already performed before upload
------------------------------------------
The deterministic gate was checked against all generated V1 truth labels:
- chemical-group cases: 50/50 correct
- mixture-threshold cases: 50/50 correct

This confirms benchmark generation and threshold semantics. It does not replace
the paid LLM-vs-Hybrid evaluation in Step 14.
