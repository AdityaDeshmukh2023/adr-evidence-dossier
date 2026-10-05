# Bounded biological role layer

This layer derives **mechanism-supported possibilities** from documented enzyme
and transporter roles. It does not predict patient harm, assign pair severity,
replace DDInter records, or treat absent biological data as non-interaction.

The frozen `data/biological_roles.json` snapshot contains selected factual rows
from the [FDA clinical CYP/transporter role table](https://www.fda.gov/drugs/drug-interactions-labeling/healthcare-professionals-fdas-examples-drugs-interact-cyp-enzymes-and-transporter-systems).
The scope is 36 selected drug names and four other substances: grapefruit juice,
St. John's wort, curcumin, and diosmin. The initial snapshot contains 93 roles;
34 drug names map unambiguously to existing ingredient IDs. `rifampin` and
`dabigatran etexilate` remain explicitly unmapped because the existing exact
terminology does not contain them. The importer does not guess identities.

## Source and refresh

Each snapshot records the actual UTC retrieval time, SHA256 of downloaded HTML,
hashes of the normalized selected source rows and role records, importer
version, exact ingredient mappings, and source URL. The full FDA HTML is hashed
during download but is not copied into the project; the selected factual rows
and their superscript footnotes are retained. Source row text normalizes only
whitespace and curly apostrophes. Parser changes or missing selected rows fail
before replacing a snapshot. Dynamic page/HTML changes can change the raw
document hash without changing the selected facts.

Ordinary screening and training feature extraction perform no network requests.

```powershell
.\.venv\Scripts\python.exe scripts/import_biology.py --check
.\.venv\Scripts\python.exe scripts/import_biology.py --refresh
.\.venv\Scripts\python.exe scripts/import_biology.py --from-html path\to\fda.html
```

`--refresh` requires network access and changes the biological snapshot. An
optional `--retrieved-at` argument is intended for replaying an independently
recorded original retrieval, not inventing a publication or retrieval time.

## Output semantics

The engine joins directed modifier roles (`inhibitor` or `inducer`) with a second
resolved drug's `substrate` role for the **same source target label**. It retains
CYP3A as a group and never expands it to CYP3A4 or CYP3A5. It does not infer an
interaction merely because two drugs are substrates. Each path includes the
two source role IDs, exact source cell text, row hashes, and applicable dose,
route, preparation, or enantiomer footnotes. Transporter labels are retained
without assuming a universal change in systemic exposure.

For example, an inhibitor/substrate path through CYP2C9 supports a possible
mechanism. It is not proof that the pair produces a clinically significant
effect. FDA inhibitor potency and substrate sensitivity categories are source
role properties, not drug-pair severity or probabilities.

Only normalized, resolved ingredient IDs create drug paths; unresolved
candidate names do not. Negated or uncertain supplement/food exposure produces
no hypothesis. Repeated exposures and repeated ingredients are deduplicated.
The source evidence for grapefruit **juice** is not generalized to whole fruit;
curcumin is not expanded to turmeric or any herb containing that constituent.
This intentionally bounded scope avoids inferring whole-product effects from
the presence of a chemical constituent.

## Public APIs

`adr_system.biology` exposes:

- `mechanism_hypotheses(medications, foods=None)`: dictionaries with
  `status="mechanism_supported_possible"`, canonical entity names, ingredient
  and mention IDs, a directed `path`, source `evidence`, `conditions`, and
  limitations. There is deliberately no `severity` or patient risk field.
- `snapshot_manifest()`: source, data hashes/version, coverage counts, unmapped
  names, limitations, and feature semantics.
- `roles_for_ingredient(ingredient_id)`: independently copied sourced records.
- `role_feature_names()` and `role_features(ingredient_id)`: 52 stable binary
  features: a documented-role-present mask and three role types across 17 exact
  target labels. Zero means no observation in this limited snapshot; it is not
  a negative biological label. These features contain no DDInter severity.
- `redacted_foods(foods)`: canonical supported exposure names with `no`/`maybe`
  context, omitting unrelated free text. Merge with existing food-rule redaction
  to preserve private replay semantics across both catalogs.
- `validate_snapshot(snapshot)`: checks role/row hashes, references, and mapping
  consistency. This detects accidental data drift; it is not a digital signature.

## Evaluation boundaries

The FDA table is a non-exhaustive role resource. It does not supply true negative
interaction labels and it omits other mechanisms, including many pharmacodynamic
and absorption interactions. Mechanism-path coverage can be measured against
independent held-out pair evidence, but source-role joins must not be evaluated
as if the imported statements themselves proved pair outcomes.

For a learned model, freeze the biology snapshot and record it in dataset and
model manifests. Compare role-aware and role-free models to detect whether role
features actually help. Document source overlap with interaction labels; absence
of severity features alone does not eliminate source leakage. Held-out drugs
outside these 34 mapped names commonly have an all-zero biological vector.

Unit tests cover source integrity, role grouping, enantiomer conditions,
unresolved/negated/uncertain inputs, whole-product boundaries, private exposure
replay, and feature missingness. They establish software behavior, not clinical
validation.
