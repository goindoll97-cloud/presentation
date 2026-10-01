Study 2 Retrieval Evaluation V1
===============================

Research task
-------------
Input is candidate CAS identifier(s) only. PubChem is used only to obtain SMILES.
The system must search the entire frozen regulatory catalog and return the correct
regulatory target_id, NOT_FOUND, or REVIEW.

This is NOT a pairwise question such as "does CAS X belong to target Y?".
The original Validation V2 132-row pairwise benchmark remains frozen and unchanged.
A retrieval-style dataset is derived from it by deduplicating real candidate CAS
signatures and assigning each query to its actual MATCH target when one exists.

Candidate-side information allowed
----------------------------------
- CAS identifier(s)
- PubChem-derived SMILES for exactly those CAS identifier(s)

Candidate-side information forbidden
-------------------------------------
- candidate chemical name
- PubChem title or synonym
- PubChem classification/description
- source category
- row-specific regulatory scope
- GOLD label or GOLD target

The regulatory catalog is the common search database and is provided equally to
LLM-only and Hybrid.

Mixtures
--------
If a named mixture has a single mixture CAS, that CAS is used.
If no mixture CAS exists but every component has a CAS, the unordered component-CAS
set is used as the CAS-only query.
If neither is possible, the mixture is excluded from the primary CAS-only retrieval
analysis and written to data/retrieval_EXCLUDED_NO_CAS.csv.

Systems
-------
1. LLM_ONLY
   Every retrieval query is sent to the LLM with the full frozen regulatory catalog.

2. HYBRID
   High-confidence deterministic matches are resolved first:
   - exact CAS in regulatory catalog
   - exact named-mixture component CAS set
   - unique RDKit parent/salt structure match
   All unresolved cases are sent to the SAME LLM with the SAME full-catalog prompt
   used by LLM_ONLY. The gate reason is not injected into the LLM prompt.

Execution order
---------------
From repository root after git pull:

  cd study2_eval_v2

  python 00_build_retrieval_benchmark.py
  python 01_freeze_retrieval_dataset.py
  python 02_prepare_shared_pubchem.py
  python 03_freeze_eval_protocol.py
  python 04_run_retrieval_llm_hybrid.py

The first run of Step 04 is a dry run and makes no LLM calls.
Inspect EVAL_PROTOCOL_FREEZE.json and intermediate/hybrid_gate_preflight.csv.

To execute LLM calls in PowerShell:

  $env:RETRIEVAL_EXECUTE_LLM="1"
  python 04_run_retrieval_llm_hybrid.py

Then score predictions:

  python 05_score_retrieval_predictions.py

Important leakage rule
----------------------
Steps 02, 03 and 04 never read retrieval_GOLD.csv.
Step 05 is the first stage allowed to join predictions to GOLD.

Primary outputs
---------------
- data/regulatory_catalog.csv
- data/retrieval_INPUT.csv
- data/retrieval_GOLD.csv
- data/retrieval_EXCLUDED_NO_CAS.csv
- RETRIEVAL_DATASET_FREEZE.json
- SHARED_PUBCHEM_FREEZE.json
- EVAL_PROTOCOL_FREEZE.json
- intermediate/hybrid_gate_preflight.csv
- intermediate/retrieval_predictions_caselevel.csv
- intermediate/retrieval_metrics_summary.csv
- intermediate/retrieval_metrics_by_source_category.csv
- intermediate/retrieval_metrics.json
