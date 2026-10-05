# Methods and measured results draft

This draft distinguishes executed experiments from remaining work. It supports a software/source-label paper; it does not report patient outcomes, independent clinical annotations or human-study results.

## Research question

We investigate whether molecular and network representations improve prediction of source interaction severity conditional on an existing interaction record, particularly for unseen medicines and incomplete biological features. A separate recognition experiment measures the consequences of carrying uncertain medicine identities through source screening instead of assuming one identity or dropping unresolved entries. Novelty requires comparison with related methods; implementing established GNN and explanation algorithms is not itself a novelty claim.

## System

Typed medication entries or printed English prescription-section images enter an identifier-normalization pipeline. Accepted identities produce source-backed findings. Uncertain entries retain up to five lexical ingredient candidates, generated independently of interaction outcomes. Candidate screening checks each entry-pair Cartesian product and records source states for every retained interpretation. State intersection gives stable predicates, and union gives possible states. Stability is conditional on the retained candidate sets and does not establish the prescribed identity. Source absence remains a coverage gap.

A separate directed biological layer joins documented inhibitor/inducer roles to substrate roles for the same target label. The frozen FDA-derived snapshot records 93 selected role rows, 34 mapped drug names and bounded food/supplement exposures. These paths show mechanism-supported possibilities without assigning pair severity. Role feature missingness is explicit; missing documentation is not a negative biological label. See [biological data](BIOLOGICAL_DATA.md).

The optional learned component classifies high/moderate/low source severity only. Unknown source-severity records are excluded from supervision; no-record pairs are not sampled as safe interaction negatives. Deployment preserves known source labels and can emit a separate experimental conditional-severity estimate or abstention for eligible documented records with unknown severity. Molecular joins, graph construction, calibration and deployment integrity are documented in [the engine guide](HYBRID_ENGINE.md) and [architecture](hybrid_architecture.mmd).

## Recognition experiment

The text dataset comprises 500 generated base prescriptions and 1,500 clean/corrupted observations, balanced over 2, 5, 10, 20 and 30 medication entries. Prescriptions are interaction-enriched through source-known anchor pairs, then extended with distinct sampled ingredients. Development/calibration/test groups are separated 60/20/20 within size strata, and related variants never cross splits. Reference identities and source records are used for scoring only. Entry boundaries are supplied for the text experiment.

The image experiment executes PaddleOCR on 120 rendered medicine-section images from 40 base groups, each with clean, three-degree rotation and blur/low-contrast variants. Fonts and layouts are held out by split. OCR receives no reference entry boundaries; detected mentions are aligned to rendered boxes using geometry after inference solely for scoring. These generated printed images do not represent handwriting or real prescribing distributions.

Methods are resolved-only screening, a top-1 lexical identity-assumption baseline, and candidate propagation. Exact source-finding keys contain ingredient-pair identifiers and source severity. Candidate-potential precision counts findings absent from the clean prescription reference as extra; this is not independent clinical false-positive labeling. We separately report identity coverage, conditional burden and incorrect stable severity predicates. The detailed definitions and grouped-bootstrap procedure are in [the protocol](HYBRID_RESEARCH_PROTOCOL.md).

## Executed recognition results, 2026-10-04

The corrupted held-out text subset contains 200 observations from 100 base groups.

| Method | High-severity source recall | Source-finding precision | Source-finding recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Resolved identities only | 20.45% | 100.00% | 24.10% | 38.83% |
| Top-1 assumed identity | 99.26% | 97.15% | 98.87% | 98.00% |
| Candidate-potential records | 99.44% | 16.09% | 99.11% | 27.68% |

The candidate-potential versus resolved-only high-severity recall difference is 79.00 percentage points, with a paired base-group bootstrap 95% interval of 72.73–84.86 points. This comparison has different output certainty: candidate-potential findings retain unresolved identities. Its larger recall must be read alongside its 16.09% precision and conditional-alert burden, rather than interpreted as better automatic interaction decisions.

Candidate identity coverage is 99.33%, with 2.64 retained interpretations per detected entry on average. Candidate propagation produces 86.72 unique conditional source records per prescription on average, of which 75.73 are absent from the reference prescription. Eleven of 883 stable severity predicates are contradicted by the reference identities, an error rate of 1.25%. Candidate omission can therefore undermine a stable predicate even though the enumeration of retained candidates is exact.

Top-1 assumptions perform strongly on the generated corruption distribution. Candidate propagation increases high-severity recall by only 0.19 points over that baseline and substantially worsens reference precision. These results motivate studying candidate calibration and alert selection; they do not establish that showing every possible source finding is an effective automatic decision policy.

Candidate-response screening has a held-out corrupted-text median of 7.70 ms with its response cache cleared, compared with 3.05 ms for an identical repeated request. Median normalization time is 447.31 ms. Source and OS caches may already be warm, and this is not an application-wide cold-start benchmark. The cache improves a screening substep while lexical normalization dominates these measured medians.

The full image run has zero execution failures, character error rate 0.0180% and word error rate 0.1049%. Geometry entry-boundary precision and recall are 100%. All three screening methods have 100% source-finding precision/recall on the 24 held-out images from eight base groups, so this image set does not demonstrate a benefit from candidate propagation. Original OCR timing records isolate inference and screening; their `end_to_end_ms` omitted normalization time, which was corrected in the subsequent benchmark implementation. No total-pipeline timing claim uses that field from this run.

Original summaries, case-level metrics and execution hashes are preserved in [text artifacts](../artifacts/hybrid/text/summary.json) and [image artifacts](../artifacts/hybrid/ocr/summary.json). Figures show [source recovery and precision](../artifacts/hybrid/text/finding_recovery.png), [conditional burden](../artifacts/hybrid/text/conditional_burden.png) and [response-cache latency](../artifacts/hybrid/text/candidate_latency.png).

## Conditional-severity model experiment, 2026-10-05

The original fixed full-batch experiment completed all 24 architecture/protocol/seed runs. Those models largely collapsed to the majority source class, with macro F1 about 0.284 and zero accepted coverage under the conservative calibration policy. These results are preserved, rather than replaced, in `artifacts/ml/training`.

A training/validation-only diagnostic extended the same fusion/seed-17 experiment to 60 optimizer updates: training loss fell from approximately 1.108 to 0.449 and validation macro F1 reached 0.674. This identified premature stopping with one optimizer update per epoch, rather than evidence that the architecture itself lacked learning capacity. A corrected schedule uses pair minibatches of 2,048 while retaining the four model choices, two protocols, three seeds and frozen source-label splits. All 24 corrected runs completed in `artifacts/ml/training-minibatch`; their summed training duration was 1,961.55 seconds (32.69 minutes), excluding data retrieval, explanations and application verification. The schedule change was justified from training/validation evidence before final comparison, rather than selected from final test results.

The source-label dataset stays frozen to the original mapping cohort: 1,107 valid molecules, 53,779 known-severity records and 28 drugs with observations in the biological catalog. The later completed catalog mapping is preserved separately and is not silently added to the training cohort. The pair test has 5,380 records; cold-drug tests contain 7,251 one-unseen and 504 both-unseen records. Both public molecule caches retain PubChem provenance and exclusions in the final bundle.

Source class counts are 11,435 high, 39,861 moderate and 2,483 low. Molecular input is a deterministic 2,048-bit, radius-two Morgan fingerprint with chirality. Pair splits separate training, validation, calibration and test at 70/10/10/10; half of the training records supply message-passing adjacency and half supply supervised labels. The model never sees held-out pair labels or reversed held-out edges in its training graph. Ingredient identifiers, source severities and drug names are excluded from predictive features. Frozen split and feature audits accompany the dataset.

The table reports mean and sample standard deviation over training seeds 17, 29 and 43 on fixed splits. These deviations describe training-seed variation, not confidence intervals over independent datasets. All metrics assess known source-severity labels; they do not evaluate unknown-label deployment cases.

| Architecture | Held-pair macro F1 | Held-pair high-severity PR AUC | One-unseen macro F1 | Both-unseen macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Molecular fingerprint baseline | 0.7888 ± 0.0116 | 0.8940 ± 0.0071 | 0.5778 ± 0.0168 | 0.3856 ± 0.0263 |
| GraphSAGE | 0.7919 ± 0.0088 | 0.9013 ± 0.0055 | 0.5394 ± 0.0130 | 0.4040 ± 0.0208 |
| Molecular/network fusion | 0.8170 ± 0.0050 | 0.9161 ± 0.0016 | 0.5560 ± 0.0119 | 0.3751 ± 0.0080 |
| Fusion with documented roles | 0.8178 ± 0.0098 | 0.9168 ± 0.0029 | 0.5512 ± 0.0080 | 0.3815 ± 0.0141 |

Fusion improves held-pair mean macro F1 by 2.82 percentage points over the molecular baseline. The role increment is only 0.08 points, with observations for 28 of 1,107 drugs, so these results do not establish a useful role-feature advantage. Fusion also does not improve unseen-drug performance over the strongest corresponding baseline. The large decline under drug holdout limits claims about transfer to new medicines. [Corrected results](../artifacts/ml/training-minibatch/summary.json), [classification figure](../artifacts/ml/training-minibatch/classification_comparison.png), and [calibration figure](../artifacts/ml/training-minibatch/calibration_comparison.png) preserve these comparisons.

## Calibration and selected deployment

The deployment architecture is selected by validation mean across seeds, followed by a representative seed near that architecture's validation mean. This selects role-fusion seed 29, version `5a592aa163c3902a4989`, rather than a test-selected winner. Its individual held-pair macro F1 is 0.82857 and high-severity PR AUC is 0.91791. Temperature scaling gives held-pair ECE 0.01225.

The calibration partition is internally separated for temperature fitting, threshold proposal and an independent policy check. The proposed threshold accepts 1,304 of 1,345 independent-check cases with 122 errors. Its Wilson 95% upper error bound is 0.11058, exceeding the configured 0.10 criterion. The selected checkpoint therefore has no qualified threshold and accepted coverage is **zero**. Accepted error is undefined when there are no accepted decisions. The application correctly returns an abstention alongside unknown source severity; the raw research classifier remains available for benchmark analysis. No threshold is substituted after observing the test results, and no formal clinical or distribution-free risk guarantee is claimed. Raw proposal metadata, qualified-policy counts and their explicit bookkeeping correction records are all preserved.

## Controlled missing features

All 24 corrected runs were evaluated with controlled missing inputs, without retraining. For the selected checkpoint, the results below use the full 5,380-record held-pair test. Affected-pair counts describe which records touch changed nodes; they are not additional test samples.

| Selected-model condition | Changed nodes | Affected test pairs | Macro F1, all 5,380 pairs |
| --- | ---: | ---: | ---: |
| Full inputs | 0 | 0 | 0.82857 |
| Zero molecular features for 30% of nodes | 333 | 2,779 | 0.58490 |
| Remove documented biological roles | 28 | 517 | 0.82905 |
| Remove graph neighbors | 1,042 | 5,380 | 0.72874 |

Molecular and network perturbations reduce classification quality, while removing the sparse role observations does not reduce overall F1. These artificial shifts assess feature dependence; calibration under shifted features is unvalidated. Actual unavailable molecule mappings cause application abstention rather than zero-filled inference. [Selected perturbation report](../artifacts/ml/deployment/missing_modalities.json).

## Model recognition under noisy identities

A separate stress experiment executes 100 held-out source-known test pairs under clean, single-character-error and mixed-corruption variants, giving 300 cases. Source severity is hidden from model input. Every queried alternative must have an existing source interaction record; this gate uses record existence only, not its severity. Candidate stability additionally requires every retained interpretation to be documented, mapped, usable and class-consistent. Missing mappings and absent source records remain visible conditions.

Top-1 identities and model availability are 100% on this generated sample in every variant. The raw top-1 classifier has macro F1 0.82519 and high-severity PR AUC 0.89082 across the same 100 pairs, whose class counts are 23 high, 74 moderate and three low. This synthetic corruption set consequently does not establish improved classification from candidate propagation. Candidate stable-class coverage falls from 100% clean to 0% under single-character errors and 2% under mixed corruption. In those two corrupted variants, 98 and 96 cases respectively include an alternative without a source record, and 73 and 60 include an unmapped alternative; the counts can overlap. All calibrated decisions abstain because the selected checkpoint has no qualified threshold. The two-case mixed stable subset has sparse class support and is not evidence of a model-performance difference. [Stress-test summary](../artifacts/hybrid/model-recognition/summary.json).

## Executed explanations

SHAP GradientExplainer is executed on 30 held-out pairs for the final molecular seed-29 baseline, and GNNExplainer on 30 held-out pairs for the selected role-fusion checkpoint. Sampling is balanced across known source classes with a declared seed; nine pairs without training-graph neighbors are excluded from GNN eligibility. Explanations remove selected present fingerprint bits or undirected training edges and compare the change in the model's original predicted-class probability against matched-count random removal. Biological-role inputs are held fixed in GNN explanations.

| Explanation | Cases | Feature probability drop | Matched random feature drop | Second-seed feature Jaccard |
| --- | ---: | ---: | ---: | ---: |
| SHAP molecular baseline | 30 | 0.29455 | 0.05774 | 0.62586 |
| GNN selected model | 30 | 0.03618 | -0.00039 | 0.01754 |

For GNN edges, mean predicted-class probability drop is 0.06412, compared with 0.00020 for random edge removal. Second-seed edge Jaccard is 0.14216. Each stability estimate uses only three repeated cases, and the low GNN overlap limits confidence in specific feature/neighbor explanations. These results establish executed model-attribution experiments with measured limitations, not validated pharmacological mechanisms. Explicitly sourced biological paths remain a separate output. [GNN cases](../artifacts/ml/deployment/explanations.json) and [SHAP cases](../artifacts/ml/training-minibatch/pair/fingerprint/29/explanations.json) retain case-level details.

## Executed GraphSAGE compute optimization, 2026-10-05

For a linear mean-aggregation layer, averaging neighbor projections is mathematically equivalent to projecting the neighbor average. Moving the projection before sparse aggregation reduces the sparse feature width from 2,048 to 128 without changing the architecture or labels. On the measured 1,107-node/37,644-directed-edge graph, the original first-layer operation has median 211.50 ms and p95 224.25 ms, compared with median 47.83 ms and p95 49.45 ms for projected aggregation. The local median ratio is 4.422, with maximum output difference approximately `2.09e-7`; gradient equivalence is checked separately in tests.

These timings concern the cached first layer on CPU with four threads while training was running in the background. A separate final measurement on the same real frozen graph with seeded synthetic weights and 30 alternating repetitions gives median 203.58 ms versus 47.58 ms, a ratio of 4.28, and maximum output difference `1.79e-7`. Neither measurement establishes a full-training or whole-application speedup, and the algebraic reassociation is not claimed as a novel GNN method. Both the [original measurement](../artifacts/hybrid-release/graphsage_optimization.json) and [final microbenchmark](../artifacts/ml/aggregation_benchmark_final.json) retain their measured scope.

Original experiment manifests recorded file hashes at artifact-write time, rather than capturing an immutable pre-run source snapshot. They should not be reinterpreted as a guaranteed record of in-memory code if files changed during execution. Corrected runs capture source hashes before inference/training and retain their own manifests. Thirteen files whose bytes match their captured execution hashes are archived with a source-snapshot manifest; two subsequently changed inference/explanation helpers are explicitly listed as unrecovered, rather than claimed as original source. Original October 4 recognition manifests remain unchanged.

## Release verification and artifact handoff

The selected checkpoint passes 13 source-authority, symmetry, graph, export/replay and fallback integration checks, with an additional 12 application UI checks. Final local suites pass in three installed environments: core-only 120 passed/three skipped; existing application 122 passed/one skipped; optional ML application 126 passed/two skipped. Source fixtures pass 11/11, all three environments pass `pip check`, and 75 Python source files parse without syntax errors. These are local verification records; remote CI, container execution, cloud/GPU training and external clinical evaluation were not executed.

With no training process active, a fresh-process first three-entry source/model request takes 1.907 seconds. Thirty repeats of that request have median 116.16 ms and p95 132.99 ms, with native peak process working set 389.68 MiB. This excludes Python launch, Streamlit rendering, OCR and live services; it is not a general prescription-size latency estimate. Separately sampled training peak working set is 489.63 MiB. [Final integration/timing](../artifacts/hybrid-release/model_integration_idle.json), [verification](../artifacts/hybrid-release/verification.json) and [dependency checks](../artifacts/hybrid-release/dependency_checks.json) preserve measured scopes and original timestamps.

The [immutable October 5 paper bundle](research_artifacts/hybrid_checkpoint_2026-10-05/BUNDLE_MANIFEST.json) preserves 614 experiment files plus its manifest (184,605,400 bytes total). An [independent read-back check](research_artifacts/hybrid_checkpoint_2026-10-05_VERIFICATION.json) verifies every archived SHA-256 value, all selected checkpoint hashes, and the selected-model/frozen-dataset/molecule-cache relationship. The bundle contains both training schedules, public mapping caches, case-level errors, explanation outputs, source-snapshot records, figures and local release evidence. Original execution timestamps are retained even when their UTC calendar date differs from the October 5 local release date.

## Limitations

Source severity is a database annotation, not an individual probability of harm. Unknown source-severity records lack independent labels; successful known-label holdout does not prove accuracy on that deployment subset. The biological role catalog is bounded, source overlap must be considered, and many drugs lack observed roles. Random pair holdouts and unseen-drug holdouts measure different generalization conditions. Synthetic identity errors and controlled image degradation do not establish recognition performance on real prescriptions. The observed candidate-potential recall/precision tradeoff should remain visible in the paper.
