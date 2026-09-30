STUDY 2 METHOD-SAFE V4
======================

Why this revision exists
------------------------
The expanded 134-case benchmark was built with structure-consistency QC.
Therefore it must not be used as an unbiased estimate of generic RDKit performance.
This package separates two roles:

A. PRIMARY_SOURCE_CURATED
   - original frozen 60-case benchmark
   - 5 parent rules
   - MATCH 30 / NO_MATCH 30
   - 59 CAS-operational cases
   - parent-majority label baseline (operational): 0.5254
   - PRIMARY inferential comparison set

B. SECONDARY_STRUCTURE_ANCHORED
   - expanded 134-case benchmark
   - 13 parent rules
   - MATCH 64 / NO_MATCH 70
   - parent-majority label baseline: 0.7910
   - structure-QC was used for inclusion
   - SECONDARY coverage/stress set, not unbiased RDKit-performance estimate

The two sets overlap in 37 rule/CAS pairs. Step 04C creates 157 unique identity
questions (156 CAS-operational). In the final fair shared-DB design, Claude is
called only when PubChem resolves the CAS to one exact verified candidate structure.
DB-unresolved/multi-CID rows are retained in the denominator as REVIEW without a
Claude call, and the same case/repeat result is reused by Hybrid.

Stereochemistry sensitivity
----------------------------
(R)-Warfarin sodium is retained as a curated MATCH, but the reference Warfarin
structure is stereochemically unspecified. It is explicitly flagged and Step 06
reports a sensitivity analysis with that case excluded.

Phenolic / neutral-drawn salt representation
---------------------------------------------
cheminformatics_identity_V4_3_SALT_AWARE.py adds a general salt-aware resolver:
- compares all disconnected candidate fragments to the reference parent;
- permits stoichiometric duplicate parent fragments;
- recognizes common counterions and solvates;
- recognizes neutral-drawn acid/base salt pairs;
- includes phenol/phenolate in the general acid class, covering halo- and
  nitro-substituted phenols such as PCP/DNOC without compound-specific rules;
- covalent derivatives (ester, N-oxide, hydroxylated analogs) do not become MATCH;
- unexplained extra components return REVIEW.

IMPORTANT: V4.3 was developed after preflight identified representation limits.
It is therefore reported only as POST-PREFLIGHT SENSITIVITY / METHOD REFINEMENT.
Do not silently replace the frozen primary deterministic comparator with V4.3
for the main inferential claim.

Run order
---------
1. Put these files in the same presentation repository directory:
   broad_salt_rules(1).csv
   broad_salt_validation_cases(1).csv
   broad_salt_rules_CURATED_13.csv
   broad_salt_validation_cases_CURATED_BALANCED_134.csv
   cheminformatics_identity.py                 (your existing frozen comparator)
   cheminformatics_identity_V4_3_SALT_AWARE.py
   04C_prepare_dual_benchmark_METHODSAFE.py
   05_compare_identity_SHARED_DB_METHODSAFE_V4.py
   06_make_identity_results_METHODSAFE_V4.py

2. Freeze dual benchmark:
   python .\04C_prepare_dual_benchmark_METHODSAFE.py

   Expected key values:
   n_unique_eval_cases = 157
   n_unique_operational_cases = 156
   n_primary_total = 60
   n_primary_operational = 59
   n_secondary_total = 134
   n_overlap_rule_cas = 37
   ready_for_claude_api = true

3. Dry-run Step 05 (NO Claude cost):
   python .\05_compare_identity_SHARED_DB_METHODSAFE_V4.py

   Expected planned Claude calls if repeats=3:
   147 exact-CAS-resolved cases x 3 = 441 paid Claude calls
   9 PubChem multi-CID/unresolved evaluation rows = automatic REVIEW, no Claude call
   Hybrid additional paid calls = 0
   V4.3 sensitivity additional paid calls = 0

4. Inspect:
   intermediate\05_pubchem_operational_identity_audit.csv
   intermediate\05_rdkit_preflight_predictions.csv
   intermediate\05_claude_prompt_manifest.csv

5. Only after inspection, paid run:
   $env:IDENTITY_EXECUTE_CLAUDE="1"
   python .\05_compare_identity_SHARED_DB_METHODSAFE_V4.py
   Remove-Item Env:IDENTITY_EXECUTE_CLAUDE

6. Final analysis:
   python .\06_make_identity_results_METHODSAFE_V4.py

Step 06 outputs
---------------
- primary/secondary performance separately
- micro metrics
- parent-macro accuracy / balanced accuracy / coverage / correct resolution
- secondary difficulty-stratified metrics
- within-benchmark parent-majority label baseline
- sensitivity excluding stereochemically ambiguous R-warfarin sodium
- V4.3 sensitivity results kept separate from primary frozen comparator

Interpretation guardrail
------------------------
Primary claim:
  PRIMARY_SOURCE_CURATED supports the main LLM+DB vs frozen DB+RDKit vs frozen
  Hybrid comparison.

Secondary claim:
  SECONDARY_STRUCTURE_ANCHORED supports structural coverage/stress analysis and
  can show how systems behave on a larger structure-anchored set, but should not
  be described as an unbiased generic RDKit accuracy benchmark.