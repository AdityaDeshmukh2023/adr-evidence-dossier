# Hybrid research protocol

## Scope

This is an automatic source-conformance and robustness experiment. It measures recovery of existing source records from noisy medicine inputs. No simulated human correction, oracle-selected candidate, or clinician-time claim is part of this benchmark. The experiment's reference identities and clean-source records are known because the inputs are generated.

## Dataset and split rules

Text: 500 independent generated prescription groups, balanced across 2/5/10/20/30 entries. Each group supplies clean, single-character and mixed-corruption observations. A source-known interaction anchors each prescription and remaining distinct ingredients are sampled from the terminology. Mixed corruption includes deletion, transposition, digit/letter confusion and some unusable strings. This interaction-enriched sampling is explicitly not a clinical prescribing distribution.

All variants of a base prescription stay in the same split. Development/calibration/test allocation is 60/20/20 within each size stratum at the default size. No thresholds or model choices may be fitted on the test groups. Split labels in the current source-propagation benchmark document data separation; they do not mean a calibrated candidate model has already been implemented or fitted. Identical observation sequences cannot cross splits.

Images: 40 generated base medicine-section images, three visual variants each, for 120 actual OCR cases. Development, calibration and test use distinct fonts and layouts. Variants are clean, three-degree rotation, and combined blur/low contrast. Rendered text and boxes are reference annotations. Entry boundaries are not supplied to OCR or normalization. Geometry-only one-to-one matching associates detected mentions with reference entries after inference, exclusively for scoring. Missed entries, extra entries and OCR failures remain in the denominators. Synthetic images include no private patient information.

## Methods

| Method | Identity behavior | Finding semantics |
| --- | --- | --- |
| Resolved only | Screen only the identifiers accepted by normalization. | Existing resolved-identity findings. |
| Top-1 assumption | Use the first retained candidate for each unclear mention without reference access. | A baseline that assumes identity; it is not a confirmed result. |
| Candidate propagation | Evaluate all retained candidate combinations for each entry pair. | Potential source records, with separate stable entry-pair state predicates. |

Methods share normalization outputs to isolate screening behavior. No gold identity is passed to candidate generation, ranking or pair screening. Resolved findings and candidate-potential findings are never pooled as if they had the same certainty. Adding an OCR-aware or calibrated candidate generator requires a separately labeled ablation against the original generator; the current comparison alone does not prove a candidate-generation improvement.

## Metrics

The exact source-finding key is `(sorted ingredient IDs, source severity)`. True/extra/missed findings are computed against the clean prescription's source-record set. An "extra" means absent from that generated prescription reference, not an independently proven false clinical interaction.

* Source precision/recall/F1 and high-severity source-record recall, reported by split, size and corruption.
* Candidate coverage: fraction of reference entries whose true ingredient tuple appears in the inferred candidate/resolved set. Candidate count and mean set size describe the cost of increasing coverage.
* Conditional burden: unique candidate-potential source findings beyond resolved findings, plus the subset absent from the reference prescription.
* Incorrect stable-predicate rate: for each entry pair and source severity, count stable states contradicted by its reference identity combination. Unmatched image entries are reported as unscorable stable predicates and do not disappear from boundary errors.
* Image CER/WER, entry-boundary precision/recall, OCR failures and end-to-end latency.
* Candidate-response cache cold/warm p50 and p95, with exact cached-output equality. "Cold" clears that cache only: source files, terminology and OS caches may already be warm. Optional tracemalloc samples report Python allocation peaks, not process RSS.

No-record and unknown source states must remain distinct from a verified absence of interaction. The severity model is conditional on a documented interaction. It does not sample unrecorded pairs as safe negatives or predict whether an interaction exists.

## Statistical reporting

The primary text comparison uses held-out corrupted test groups: candidate-potential high-severity source recall minus resolved-only recall. Report its conditional burden and precision adjacent to the recall difference. A paired bootstrap resamples whole prescription groups, keeping their related variants together, with 1,000 replicates and a fixed seed. Small exploratory runs may have no test groups or too few labels; their interval is explicitly unavailable.

Use multiple seeds for the final research claim and preserve each run. Do not select the best seed or report development/calibration performance as test performance. A descriptive all-split run is useful for debugging but does not replace the held-out test tables.

## Optimization and mathematical scope

For strictly pairwise findings, retained-candidate joins require at most `sum(k_i * k_j)` entry-pair candidate combinations, rather than enumerating `product(k_i)` full-prescription assignments. Combination products may expand into several ingredient-pair checks. Stable states are intersection across outcomes of one entry pair; possible states are union across those outcomes.

These local computations do not give exact global minimum/maximum counts of simultaneously present prescription-wide findings: different entry pairs share identity variables, and their extrema may require incompatible choices. Do not sum pairwise maxima and call it a reachable global prescription risk. Any later constraints between candidate identities must be documented; current Cartesian products consider all retained combinations.

## Optional model experiment

Treat learned conditional-severity scores as research predictions in a separate namespace. Train on documented records with known severity; the deployed model applies only where an existing source record has unknown severity. It cannot establish the correct severity of those unknown records without independent labels, so evaluate by holding out known source-severity labels. Fit on training edges, select hyperparameters on development data and calibrate using an independent calibration partition. Record unseen-pair and unseen-drug results separately, macro/classwise AUPRC, macro F1, calibration, missing-feature ablations and source versions. Keep held-out label edges out of training adjacency, feature construction and any target-derived text. Biological feature provenance and molecule/identifier join coverage must be recorded.

A trained artifact, code, manifest and actual held-out report are necessary before marking this component executed. No checkpoint means the UI should report model unavailability; it must not invent a score or silently present a source lookup as a model prediction. Known source labels remain unchanged. Unrecorded pairs receive no conditional-severity prediction.

## Reproducibility and claim limits

Scripts save generated datasets, case metrics, aggregate tables and original manifests. The bundle script copies an allowlist, hashes every included artifact and refuses to overwrite a changed historical bundle. Generated-only runs do not count as actual image OCR or model training. Synthetic source-label recovery supports a software robustness claim, not clinical validity, prescribing recommendations or patient outcome estimates.
