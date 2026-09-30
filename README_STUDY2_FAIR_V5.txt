STUDY 2 FAIR V5.3
=================

Why V5 exists
-------------
The earlier V4 comparison had a methodological asymmetry: the LLM prompt was given
an explicit generic parent-and-salts decision policy while the primary deterministic
comparator relied mainly on FragmentParent normalization. Complex salt representations
could therefore be harder for RDKit for reasons unrelated to the intended system
comparison. FAIR V5 removes that asymmetry.

Primary FAIR V5 systems
-----------------------
1. LLM + shared PubChem structure
2. DB + salt-aware RDKit V5
3. Hybrid: salt-aware RDKit V5 first; reuse the same case-level LLM consensus only on REVIEW

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
- unresolved PubChem cases remain REVIEW for all systems;
- a failed Claude API call is never scored as REVIEW: the run stops without scoring;
- truth labels and reference-evidence text are excluded from the model-input view.

Salt-aware deterministic policy
-------------------------------
- inspect all disconnected fragments;
- standardize charge/protonation and canonical tautomers;
- preserve sp3 stereocenters during tautomer canonicalization;
- support stoichiometric duplicate parent fragments;
- recognize generic acid/base counterion relations, including phenol/phenolate;
- recognize chemically compatible cations/anions and neutral-drawn acid/base pairs;
- recognize PubChem-style neutral counterion drawings such as HCl written as "Cl";
- recognize neutral metal atoms such as "[Na]" for acidic parents;
- allow the same explicit common-solvate list in both LLM and RDKit policy;
- covalent derivatives/analogs without the same parent fragment are NO_MATCH;
- unexplained or chemically incompatible components are REVIEW;
- a reference parent drawn without stereochemistry covers all stereoisomers;
- otherwise stereo-only differences use the same frozen isomer_scope supplied to both systems.

Inference status of the existing benchmarks
-------------------------------------------
The existing 60-case and 134-case benchmarks were already visible during method
refinement. FAIR V5 performance on those sets is therefore DEVELOPMENT / EXPLORATORY
only. It is useful for debugging and fairness checks, but it must not be presented
as an independent confirmatory superiority test.

A confirmatory analysis requires a NEW independently curated holdout constructed
after the V5 protocol is frozen. Candidate CAS numbers must not overlap the
development set. The exact holdout file is then frozen by SHA256 before Step 05.

Minimum final-holdout columns
-----------------------------
case_id
rule_id
candidate_cas
reference_parent_smiles
reference_membership
reference_source

Active FAIR V5 files
--------------------
- 04C_prepare_dual_benchmark_METHODSAFE.py
- 04D_freeze_fair_v5_protocol.py
- 04E_freeze_final_holdout_FAIR_V5.py
- identity_shared_runtime_V5.py
- cheminformatics_identity_V5_FAIR.py
- 05_compare_identity_SHARED_DB_FAIR_V5.py
- 06_make_identity_results_FAIR_V5.py
- 07_visualize_identity_results_FAIR_V5.py

The older METHODSAFE V4 evaluator/result/visualization files are not part of the
active pipeline. Their history remains recoverable from Git, so they do not need
to remain in the current tree after FAIR V5 became standalone.

04A/04B note
------------
04A and 04B are benchmark-construction/provenance utilities. Their filenames retain
historical V4.x naming, but they are not the obsolete V4 system comparator. Normal
FAIR V5 reanalysis starts at 04C because the curated input CSVs already exist.
Keep 04A/04B only when benchmark regeneration/audit provenance is needed.

Recommended run order
---------------------
Study 2 does not need Study-1 Steps 01-03.

A. Development fairness reanalysis (exploratory only)
   python 04C_prepare_dual_benchmark_METHODSAFE.py
   python 05_compare_identity_SHARED_DB_FAIR_V5.py
   # inspect 05_v5_rdkit_preflight_predictions.csv and 05_v5_claude_prompt_manifest.csv
   $env:IDENTITY_EXECUTE_CLAUDE="1"
   python 05_compare_identity_SHARED_DB_FAIR_V5.py
   Remove-Item Env:IDENTITY_EXECUTE_CLAUDE
   python 06_make_identity_results_FAIR_V5.py
   python 07_visualize_identity_results_FAIR_V5.py

B. Freeze the final protocol BEFORE creating/opening the final holdout
   # Set IDENTITY_ANTHROPIC_* exactly as intended for the final experiment.
   python 04D_freeze_fair_v5_protocol.py

C. Independently construct/curate the final holdout without inspecting V5 predictions
   # Then set the path and freeze the exact file bytes BEFORE any Step-05 run.
   $env:IDENTITY_V5_FINAL_HOLDOUT_FILE="data/identity_final_holdout_v5.csv"
   python 04E_freeze_final_holdout_FAIR_V5.py

D. Independent final holdout evaluation
   python 05_compare_identity_SHARED_DB_FAIR_V5.py   # dry run first
   # inspect RDKit preflight + exact Claude prompt manifest; do not change protocol/holdout
   $env:IDENTITY_EXECUTE_CLAUDE="1"
   python 05_compare_identity_SHARED_DB_FAIR_V5.py
   Remove-Item Env:IDENTITY_EXECUTE_CLAUDE
   python 06_make_identity_results_FAIR_V5.py
   python 07_visualize_identity_results_FAIR_V5.py

Protocol/holdout integrity gates
--------------------------------
04D stores SHA256 for:
- salt-aware engine;
- shared PubChem/Anthropic runtime;
- Step 05 evaluator;
- LLM model, effort, token limit, repeats, system prompt, response schema;
- structural prompt policy.

04E stores SHA256 for:
- the exact final holdout CSV;
- the 04D protocol freeze file;
- Step 05.

Step 05 rechecks these hashes before any confirmatory evaluation. A changed engine,
prompt, runtime, evaluator, or holdout invalidates the frozen protocol and stops the run.

Statistical design
------------------
Repeated LLM calls are collapsed to ONE case-level consensus. RDKit is evaluated once
per chemical-identity case. Primary paired comparison uses exact McNemar on the
case-level overall-correct-resolution endpoint, where REVIEW is unresolved rather
than silently discarded. Selective accuracy is still reported separately among
MATCH/NO_MATCH decisions, together with coverage and overall correct resolution.
