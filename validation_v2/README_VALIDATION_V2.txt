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
   but whose chemical identity can be supported by an external database.

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
  regulatory_group_reference.py
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
Current curated seed: n = 20 (MATCH 10 / NO_MATCH 10), representing 10 unique
named mixture identities from the K-REACH classification/labeling list.

Examples include:
- Mixture of methylsilanetriol triphosphate and phosphoric acid (CAS-less mixture)
- a four-component acrylate mixture
- phosphine-oxide mixtures
- isomeric ketone / ester / amine mixture identities

For each of the 10 target mixture identities, the seed contains:
- one exact-composition MATCH;
- one cross-mixture NO_MATCH using another real K-REACH mixture as a hard negative.

Thus n=20 should not be interpreted as 20 independent mixture identities; the
current effective unique-mixture count is 10. Future expansion should increase
unique mixture identities rather than only adding more cross-pair rows.

Source
------
Mixture identities are anchored to the K-REACH Chemical Information Processing
System classification/labeling PDF:
https://kreach.me.go.kr/repwrt/ghs/ghsClassLabeling/ghsPdf.do

Recommended execution
---------------------
From the repository root:

  cd validation_v2
  python 01_build_parent_salt_validation.py
  python 02_build_chemical_group_validation.py
  python 03_build_mixture_validation.py
  python 04_merge_validation_master.py

Default final-freeze minimums:
  PARENT_SALT    >= 72
  CHEMICAL_GROUP >= 20
  MIXTURE        >= 20

Final freeze:
  python 05_freeze_validation_v2.py

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
