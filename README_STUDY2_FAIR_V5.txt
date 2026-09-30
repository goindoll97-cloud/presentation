STUDY 2 FAIR V5
================

Why V5 exists
-------------
V4 showed a methodological asymmetry: the LLM prompt explicitly received a generic
parent-and-salts decision policy, while the primary frozen RDKit comparator still
relied mainly on FragmentParent normalization. Complex salt representations could
therefore be harder for RDKit for reasons unrelated to the intended model comparison.
V5 removes that asymmetry.

Primary FAIR V5 systems
-----------------------
1. LLM + DB
2. DB + salt-aware RDKit V5
3. Hybrid: salt-aware RDKit V5 first; reuse the same LLM result only on REVIEW

Equal-information rules
-----------------------
- exact-CAS PubChem lookup is performed once and shared;
- candidate name and CAS are withheld after the lookup;
- benchmark source SMILES are not used as operational candidate structures;
- all systems receive the same PubChem candidate SMILES;
- all systems receive the same frozen reference-parent SMILES;
- all systems receive the same isomer_scope;
- the same generic salt policy is written in the LLM prompt and implemented in RDKit;
- no compound-specific RDKit exception is allowed;
- unresolved PubChem cases remain REVIEW for all systems.
- a failed Claude API call is never scored as REVIEW: Step 05 makes one preflight
  call before the batch and stops without scoring if any call fails (rerun retries
  only the failed calls; successful calls are cached).

Salt-aware V5 deterministic policy
----------------------------------
- inspect all disconnected fragments;
- standardize charge/protonation and canonical tautomers;
- support stoichiometric duplicate parent fragments;
- recognize generic acid/base counterion relations, including phenol/phenolate;
- recognize chemically compatible cations/anions and neutral-drawn acid/base pairs;
- recognize PubChem-style neutral drawings of counterions: hydrohalic, nitric and
  other inorganic oxoacids drawn neutral (e.g. HCl as "Cl") for basic parents, and
  lone neutral metal atoms (e.g. "[Na]") for acidic parents (added in V5.1);
- allow common solvate fragments;
- covalent derivatives/analogs without the same parent fragment are NO_MATCH;
- unexplained components are REVIEW;
- stereo-only differences use the same frozen isomer_scope supplied to the LLM.

Critical inference rule
-----------------------
The existing 60-case and 134-case benchmarks were already visible during method
refinement. Therefore V5 results on those sets are DEVELOPMENT / EXPLORATORY only.
They can diagnose whether the fairness correction behaves as intended, but they
must not be presented as a new confirmatory superiority test.

For confirmatory evaluation, create a NEW independently curated holdout AFTER the
V5 protocol is frozen. Candidate CAS numbers must not overlap the development set.
The file must include at least:
  case_id
  rule_id
  candidate_cas
  reference_parent_smiles
  reference_membership
  reference_source
  holdout_frozen_before_v5_evaluation = true

Recommended run order
---------------------
A. Development fairness reanalysis
   python 05_compare_identity_SHARED_DB_FAIR_V5.py
   # inspect dry-run files
   $env:IDENTITY_EXECUTE_CLAUDE="1"
   python 05_compare_identity_SHARED_DB_FAIR_V5.py
   Remove-Item Env:IDENTITY_EXECUTE_CLAUDE
   python 06_make_identity_results_FAIR_V5.py
   python 07_visualize_identity_results_FAIR_V5.py

B. Freeze protocol BEFORE final holdout evaluation
   python 04D_freeze_fair_v5_protocol.py

C. Independent final holdout
   $env:IDENTITY_V5_FINAL_HOLDOUT_FILE="data/identity_final_holdout_v5.csv"
   python 05_compare_identity_SHARED_DB_FAIR_V5.py   # dry run first
   $env:IDENTITY_EXECUTE_CLAUDE="1"
   python 05_compare_identity_SHARED_DB_FAIR_V5.py
   Remove-Item Env:IDENTITY_EXECUTE_CLAUDE
   python 06_make_identity_results_FAIR_V5.py
   python 07_visualize_identity_results_FAIR_V5.py

Important statistical change
----------------------------
V4 replicated deterministic predictions across LLM repeats for paired summaries.
V5 instead collapses repeated LLM calls to ONE case-level consensus and performs
primary pairwise McNemar comparisons once per chemical identity case. Repeat-level
LLM consistency is retained as a robustness result, avoiding pseudo-replication.

Files added in FAIR V5
----------------------
- cheminformatics_identity_V5_FAIR.py
- 04D_freeze_fair_v5_protocol.py
- 05_compare_identity_SHARED_DB_FAIR_V5.py
- 06_make_identity_results_FAIR_V5.py
- 07_visualize_identity_results_FAIR_V5.py

V4 files are retained unchanged as an audit trail of the earlier experiment.
