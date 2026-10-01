VALIDATION V2 — THREE-CATEGORY BENCHMARK CONSTRUCTION
=====================================================

Goal
----
Build and freeze validation datasets BEFORE running Rule / LLM / Hybrid evaluation.
The benchmark is separated into three chemical-scope categories:

1) PARENT_SALT
   Parent substances and their salts.

2) CHEMICAL_GROUP
   Generic chemical-group scopes. This subset is designed to include candidates
   that are NOT directly enumerated in the closed regulatory CAS list but can be
   supported by external chemical-identity evidence.

3) MIXTURE
   Named mixtures/compositions, including CAS-less regulatory mixture identities.

Critical anti-leakage rule
--------------------------
Each category is exported as two files:
  *_INPUT.csv : fields available to the evaluated system
  *_GOLD.csv  : truth label, curation rationale, and source audit information

Never feed *_GOLD.csv to an evaluated LLM/Hybrid system.

Directory
---------
validation_v2/
  validation_common.py
  01_build_parent_salt_validation.py
  02_build_chemical_group_validation.py
  03_build_mixture_validation.py
  04_merge_validation_master.py
  05_freeze_validation_v2.py
  seeds/
    chemical_group_candidates_seed.csv
    mixture_candidates_seed.csv
  data/       # generated validation files
  sources/    # generated source manifests / summaries
  cache/      # local retrieval cache if later added

Parent/Salt subset
------------------
Source: repository data/identity_e2e_v6_72.csv
n = 72.
These are the same controlled chemical cases used in Study 2 V6.
They MUST NOT be described as a new independent holdout.

Chemical-group subset
---------------------
The seed file starts with a small number of externally supported open-set cases,
including lead compounds not directly enumerated in the closed starter registry,
a Cr(VI) open-set positive, and oxidation-state / homolog hard negatives.

02_build_chemical_group_validation.py:
- resolves supplied CAS through PubChem;
- verifies exact CAS as a PubChem synonym;
- checks whether that CAS is directly enumerated in data/regulatory_group_rules_v1.csv;
- does NOT create a GOLD label from PubChem;
- exports only rows already marked curation_status=APPROVED.

The starter seed count is deliberately small. Expand and curate it before a
paper-facing freeze.

Mixture subset
--------------
The seed file begins with real mixture identities found in Korea's chemical
information system, including:
- Mixture of methylsilanetriol triphosphate and phosphoric acid (CAS not assigned)
- a four-component acrylate mixture identified by CAS numbers

MATCH and reciprocal composition hard-negative examples are included only as a
starter for the data model. Expand real mixture cases before final freeze.

Recommended execution
---------------------
From the repository root:

  cd validation_v2
  python 01_build_parent_salt_validation.py
  python 02_build_chemical_group_validation.py
  python 03_build_mixture_validation.py
  python 04_merge_validation_master.py

Do NOT freeze for the final paper while the starter external subsets are small.
The freeze script requires by default:
  PARENT_SALT    >= 72
  CHEMICAL_GROUP >= 20
  MIXTURE        >= 20

Final freeze:
  python 05_freeze_validation_v2.py

Development-only small freeze:
PowerShell:
  $env:VALIDATION_V2_ALLOW_SMALL_FREEZE="1"
  python 05_freeze_validation_v2.py
  Remove-Item Env:VALIDATION_V2_ALLOW_SMALL_FREEZE -ErrorAction SilentlyContinue

Generated files
---------------
data/validation_parent_salt_INPUT.csv
data/validation_parent_salt_GOLD.csv
data/validation_chemical_group_INPUT.csv
data/validation_chemical_group_GOLD.csv
data/validation_mixture_INPUT.csv
data/validation_mixture_GOLD.csv
data/validation_master_INPUT.csv
data/validation_master_GOLD.csv
VALIDATION_V2_FREEZE.json

Curation policy
---------------
External DB retrieval is evidence, not truth by itself.
A row reaches the final validation dataset only when curation_status=APPROVED and
its GOLD label has an explicit rationale and source URL.

For ambiguous scope or insufficient composition evidence, use GOLD=REVIEW rather
than forcing MATCH/NO_MATCH.
