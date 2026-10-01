STUDY 2 E2E V6 — LLM + PubChem vs Hybrid + PubChem
====================================================

Research question
-----------------
Can an LLM system connected to an external chemical DB correctly identify whether a candidate chemical is included in a scope such as "parent chemical and its salts" when the operational input is the regulatory scope text plus chemical name/CAS information?

A second operational question is whether a Hybrid architecture can retain comparable decision performance while reducing LLM calls, token use, API cost, and processing time.

Systems compared
----------------
1. LLM + PubChem
   - PubChem exact-CAS lookup is performed first.
   - The LLM receives the written regulatory scope, parent/candidate name and CAS, and PubChem-resolved structures.
   - All DB-resolved cases are sent to the LLM.

2. Hybrid + PubChem
   - The same PubChem lookup is used.
   - A deterministic low-cost chemical-structure gate first decides clear MATCH/NO_MATCH cases.
   - Only gate=REVIEW cases are sent to the LLM.
   - RDKit is therefore NOT presented as a third comparison model; it is an internal low-cost component of Hybrid.

Outputs
-------
MATCH / NO_MATCH / REVIEW

Primary performance endpoint
----------------------------
Overall correct resolution: REVIEW remains in the denominator as unresolved.

Efficiency endpoints
--------------------
- LLM case rate
- Number of LLM API calls
- Input tokens
- Output tokens
- API cost (USD)
- LLM API elapsed time
- Operational elapsed-time proxy including shared PubChem lookup

Benchmark
---------
data/identity_e2e_v6_72.csv
- 72 cases
- 9 parent chemicals
- 36 in-scope / 36 out-of-scope
- NO pre-supplied SMILES in the benchmark
- PubChem structures are retrieved at run time from CAS numbers.

Important interpretation note
-----------------------------
The 72 chemicals are intentionally reused from the earlier V5 controlled-structure study. Therefore V6 is a newly framed end-to-end evaluation using the same chemical cases; it should not be described as a new independent confirmatory holdout.

Execution order (PowerShell)
----------------------------
1) Pull latest code
   git pull origin main

2) Freeze the V6 protocol
   python 08_freeze_identity_e2e_V6.py

3) Dry-run: PubChem lookup + Hybrid gate only; NO Claude charge
   Remove-Item Env:IDENTITY_V6_EXECUTE_CLAUDE -ErrorAction SilentlyContinue
   python 09_run_identity_e2e_LLM_HYBRID_V6.py

   Inspect:
   intermediate_v6/09_v6_hybrid_gate_preflight.csv
   intermediate_v6/09_v6_prompt_manifest.csv
   intermediate_v6/09_v6_run_metadata.json

4) Paid evaluation
   $env:IDENTITY_V6_EXECUTE_CLAUDE="1"
   python 09_run_identity_e2e_LLM_HYBRID_V6.py
   Remove-Item Env:IDENTITY_V6_EXECUTE_CLAUDE -ErrorAction SilentlyContinue

5) Aggregate results
   python 10_make_identity_e2e_results_V6.py

6) Make figures
   python 11_visualize_identity_e2e_results_V6.py

Main outputs
------------
intermediate_v6/09_v6_system_predictions_caselevel.csv
intermediate_v6/09_v6_efficiency.csv
intermediate_v6/10_v6_performance.csv
intermediate_v6/10_v6_efficiency_summary.csv
intermediate_v6/10_v6_mcnemar.csv
intermediate_v6/10_v6_failure_or_review_cases.csv

figures_study2_v6/Fig01_e2e_performance.png/.pdf
figures_study2_v6/Fig02_case_outcomes.png/.pdf
figures_study2_v6/Fig03_operational_efficiency.png/.pdf
figures_study2_v6/Fig04_performance_by_difficulty.png/.pdf
figures_study2_v6/FigS01_parent_heatmap.png/.pdf

Pricing assumption
------------------
Default code values for Claude Sonnet 5 (checked 2026-10-01):
- input: USD 2 per 1 million tokens
- output: USD 10 per 1 million tokens

Override if pricing changes:
$env:IDENTITY_V6_INPUT_USD_PER_MTOK="..."
$env:IDENTITY_V6_OUTPUT_USD_PER_MTOK="..."

Existing V5 files are preserved and are not modified by V6.
