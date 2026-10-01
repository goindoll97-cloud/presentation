Study 2 Retrieval Evaluation V2
===============================

Research task
-------------
Input is candidate CAS identifier(s) only. PubChem is used only to obtain SMILES.
The system must search the entire frozen regulatory catalog and return the correct
regulatory target_id, NOT_FOUND, or REVIEW.

This is NOT a pairwise question such as "does CAS X belong to target Y?".
The original Validation V2 132-row pairwise benchmark remains frozen and unchanged.
A retrieval-style dataset is derived from it by deduplicating real candidate CAS
signatures and assigning each query to its verified MATCH target when one exists.

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

Catalog provenance safeguards
-----------------------------
- Parent/salt targets come from frozen target reference fields.
- Chemical-group membership lists come from the frozen regulatory group reference.
- Named-mixture targets use frozen target scope text and target component CAS sets.
- A mixture CAS is NOT back-filled from a GOLD MATCH candidate.
- Target-level scope fields must be unique; row-order fallback is prohibited.

Retrieval GOLD review
---------------------
A pairwise NO_MATCH means only "not this target" and cannot automatically prove
"not in any target". Therefore every derived NOT_FOUND (and any MULTI_TARGET)
retrieval GOLD row requires a second researcher review against all 29 frozen
catalog targets before retrieval freeze.

01_freeze_retrieval_dataset.py creates/refreshes:
  review/retrieval_gold_manual_review.csv

The human-only review file may show source candidate names and exact-hit prechecks.
These fields are NEVER supplied to LLM-only or Hybrid.

After manual inspection only:
  python 01b_retrieval_gold_signoff.py --approve

Then rerun the freeze.

Exact-hit conflicts (CAS listed under a catalog target other than its derived GOLD)
are never bulk-approved. Resolve each one either by
- a GOLD correction in review/retrieval_gold_overrides.csv
  (cas_inputs, from_gold_target_id, to_gold_target_id, evidence, researcher_note),
  applied by 00b_apply_gold_overrides.py during 01 and requiring fresh approval, or
- an explicit per-row approval with justification:
    python 01b_retrieval_gold_signoff.py --approve-conflict RQ-0001 --note "..."
01 fails while any conflict row is unresolved.

Mixtures
--------
If a named mixture has a single mixture CAS, that CAS may appear as the query.
If no mixture CAS exists but every component has a CAS, the unordered component-CAS
set is used as the CAS-only query. The catalog does not learn a mixture CAS from
GOLD; it uses target scope/component information only.

Systems
-------
1. LLM_ONLY
   Every retrieval query is sent to the LLM with the full frozen regulatory catalog.

2. HYBRID
   High-confidence deterministic matches are resolved first:
   - exact CAS in target reference/group registry
   - exact named-mixture component CAS set
   - unique RDKit parent/salt structure match using the frozen target isomer_scope
   All unresolved cases are sent to the SAME LLM with the SAME full-catalog prompt.
   The catalog in that prompt includes each parent/salt target's normalized
   isomer_scope, so the LLM sees the same stereo rule the gate applies.

Protocol hardening
------------------
03_freeze_eval_protocol.py requires:
- parent/salt engine present
- RDKit available
- generic parent/salt engine self-tests all PASS
- parent/salt engine SHA256 recorded
- RDKit version recorded
- model, prompt version, system prompt hash, repeat count, sampling mode,
  max_tokens and effort recorded

04_run_retrieval_llm_hybrid.py re-checks all of those runtime settings before any
API call. A changed PowerShell environment cannot silently change the experiment.
04 also requires that RETRIEVAL_DATASET_FREEZE.json, SHARED_PUBCHEM_FREEZE.json and
prompt_manifest.csv are unchanged since 03, and recomputes every prompt hash and
every Hybrid gate decision; any difference blocks execution (rerun 03 instead).

LLM repetition policy
---------------------
The frozen repeat count must be completed for every LLM-evaluated query.
For n_repeats=3, every query must have exactly 3 successful calls before consensus.
If any query has fewer than 3 successful calls, prediction generation and scoring
are blocked. Re-running the same command reuses successful cached calls.

LLM-only and Hybrid-review calls are interleaved by repeat/query to reduce temporal
API drift between conditions; which condition is called first alternates by repeat.
Every fresh attempt (including failures) is appended to intermediate/llm_attempt_log.csv,
and per-condition failed attempts / unparseable responses are summarized in
retrieval_run_metadata.json so retry-induced selection effects can be reported.

Execution order
---------------
From repository root after git pull:

  cd study2_eval_v2

1) Build and attempt retrieval freeze:
  python 01_freeze_retrieval_dataset.py

The first run is expected to stop if retrieval GOLD review rows are PENDING.
Open:
  review/retrieval_gold_manual_review.csv
  data/regulatory_catalog.csv

After manual review:
  python 01b_retrieval_gold_signoff.py --approve
  python 01_freeze_retrieval_dataset.py

2) Rebuild shared PubChem SMILES after the new retrieval freeze:
  python 02_prepare_shared_pubchem.py

3) Freeze evaluation protocol:
  python 03_freeze_eval_protocol.py

3b) Researcher QC before any LLM call (reads GOLD; not a model input):
  python 03b_gate_gold_conflict_check.py
  If GOLD is corrected via the override file, rerun 01 -> 02 -> 03.
  Report the number of conflicts and overrides: correcting GOLD toward the gate
  favours Hybrid, so each override needs a regulatory justification.

4) Dry run:
  python 04_run_retrieval_llm_hybrid.py

5) Real execution in PowerShell:
  $env:RETRIEVAL_EXECUTE_LLM="1"
  python 04_run_retrieval_llm_hybrid.py

6) Score only after full repeat completion:
  python 05_score_retrieval_predictions.py

Important leakage rule
----------------------
Steps 02, 03 and 04 never read retrieval_GOLD.csv.
Step 05 is the first model-evaluation stage allowed to join predictions to GOLD.
The separate human review step occurs before model execution and is not model input.

Primary outputs
---------------
- data/regulatory_catalog.csv
- data/retrieval_INPUT.csv
- data/retrieval_GOLD.csv
- data/retrieval_EXCLUDED_NO_CAS.csv
- data/retrieval_CAS_CORRECTIONS.csv
- review/retrieval_gold_manual_review.csv
- review/retrieval_gold_overrides.csv
- review/gate_gold_conflicts_pre_execution.csv
- RETRIEVAL_DATASET_FREEZE.json
- SHARED_PUBCHEM_FREEZE.json
- EVAL_PROTOCOL_FREEZE.json
- intermediate/hybrid_gate_preflight.csv
- intermediate/retrieval_predictions_caselevel.csv
- intermediate/llm_attempt_log.csv
- intermediate/retrieval_metrics_summary.csv
- intermediate/retrieval_metrics_by_gold_category.csv
- intermediate/retrieval_not_found_gate_conflicts.csv
- intermediate/retrieval_metrics.json
