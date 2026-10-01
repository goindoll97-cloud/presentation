VALIDATION V2 — THREE-CATEGORY BENCHMARK CONSTRUCTION
=====================================================

Goal
----
Build and freeze validation datasets BEFORE running Rule / LLM / Hybrid evaluation.
The benchmark is separated into three chemical-scope categories:

1) PARENT_SALT
   Parent substances and their salts.

2) CHEMICAL_GROUP
   Generic chemical-group scopes. This subset includes candidates that are NOT
   directly enumerated under the target rule's frozen closed-registry CAS list,
   but whose chemical identity can be supported by external chemical databases.

3) MIXTURE
   Named mixtures/compositions, including CAS-less regulatory/assessment mixture
   identities.

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
  regulatory_group_reference.py
  01_build_parent_salt_validation.py
  02_build_chemical_group_validation.py
  03_build_mixture_validation.py
  04_merge_validation_master.py
  05_freeze_validation_v2.py
  seeds/
    chemical_group_candidates_seed.csv
    mixture_candidates_seed.csv
    mixture_candidates_seed_additional.csv
  data/       # generated validation files
  sources/    # generated source manifests / summaries
  cache/

Parent/Salt subset
------------------
Source: repository data/identity_e2e_v6_72.csv
n = 72.
These are the same controlled chemical cases used in Study 2 V6.
They MUST NOT be described as a new independent holdout.

Chemical-group subset
---------------------
Current curated seed: n = 30 (MATCH 15 / NO_MATCH 15).

The current seed deliberately emphasizes open-set chemical-scope challenges:
- Pb-containing compounds whose candidate CAS is not directly enumerated in the
  target rule's frozen closed-registry CAS set;
- tributyltin compounds not directly enumerated in the closed registry;
- a Cr(VI) open-set positive;
- oxidation-state, substituent-class, homolog, and metal-analog hard negatives.

02_build_chemical_group_validation.py:
- enriches supplied CAS through PubChem when available;
- records PubChem exact-CAS verification as live QC metadata;
- does NOT fail dataset construction solely because a PubChem API lookup is
  unresolved, because live API behavior is not the GOLD-label authority;
- checks direct membership against validation_v2/regulatory_group_reference.py;
- does NOT create a GOLD label from PubChem;
- exports only rows already marked curation_status=APPROVED.

Important interpretation:
This subset tests chemical-scope classification under the benchmark's encoded
regulatory-scope definition. It should not be described as an independent legal
opinion for every real-world regulatory application.

Mixture subset
--------------
Current curated seed: n = 30 (MATCH 15 / NO_MATCH 15), representing 15 unique
named mixture identities.

The first 10 unique target identities are anchored to the K-REACH Chemical
Information Processing System classification/labeling list. Five additional
CAS-unassigned mixture identities (2017-886, 2017-887, 2017-890, 2017-891,
2017-892) are anchored to archived NIER hazard-assessment-result records.

For each target mixture identity, the benchmark contains:
- one exact-composition MATCH;
- one cross-mixture NO_MATCH using another real named mixture as a hard negative.

No reaction-mixture/reaction-product entries were added in this expansion. They
remain outside the current three-category benchmark and can be studied later as
a separate category.

Sources
-------
K-REACH classification/labeling PDF:
https://kreach.me.go.kr/repwrt/ghs/ghsClassLabeling/ghsPdf.do

Archived NIER hazard-assessment-result PDF mirror used for the five additional
CAS-unassigned identities:
https://resource.chemlinked.com.cn/old/cdn/img/newcl/file/hazard_evaluation_reusults.pdf

Recommended execution
---------------------
From the repository root:

  cd validation_v2
  python 01_build_parent_salt_validation.py
  python 02_build_chemical_group_validation.py
  python 03_build_mixture_validation.py
  python 04_merge_validation_master.py

Default final-freeze requirements:
  PARENT_SALT    >= 72 rows
  CHEMICAL_GROUP >= 20 rows
  MIXTURE        >= 20 rows
  unique MIXTURE target identities >= 15
  unique CHEMICAL_GROUP target rules >= 3

Final freeze:
  python 05_freeze_validation_v2.py

Expected current category sizes after rebuilding:
  PARENT_SALT       72
  CHEMICAL_GROUP    30
  MIXTURE           30
  TOTAL            132

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
