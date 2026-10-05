# Verification results

Recorded on 2026-10-03, updated on 2026-10-04 after the low-cost Groq migration, using Windows and Python 3.12. The working tree includes uncommitted changes; the Git revision alone does not identify this implementation. Generated manifests include code and dataset SHA-256 hashes, and each experiment retains the manifest from its own execution.

| Check | Observed result | What it establishes |
|---|---|---|
| Automated suite | 53 tests passed | Core, Groq contracts, typed graph provenance/privacy/hypotheses/version changes, and Streamlit graph/review/navigation behavior |
| Exact source-conformance fixtures | 11/11 passed | Expected source record IDs, severities, evidence, and completeness states |
| Dependency consistency | `pip check` passed | No reported incompatibility among installed Python package requirements |
| Python compilation and diff whitespace | Passed | Syntax compilation and tracked diff formatting checks |
| Application startup | Local `/_stcore/health` returned `ok` | Streamlit starts; the temporary smoke-test server was stopped afterward |
| Actual OCR smoke | Passed with cached CPU models | Printed warfarin/aspirin image reaches the expected source finding |
| Actual OCR image benchmark | 8 images, 2 base prescription groups; zero execution failures | On these controlled images only: CER/WER 0, ingredient/finding recall 1 |
| Live evidence smoke | Warfarin label passages and both PubChem identity responses retrieved | Live connectivity, document/version capture, and record attachment; aspirin label search explicitly returned no relevant passage |
| Explanation benchmark | 12 outputs: 4 templates, 4 prompt-only, 4 validated extractive; 4/4 live extractive outputs passed | Live GPT-OSS 20B integration and source-contract checks; expert claim-support ratings remain pending |
| Review benchmark provenance | Original code/data execution hashes retained | Saved review results identify their execution snapshot; subsequent Groq/UI changes do not rewrite that manifest |
| Typed evidence graph | 14 software scenarios; 7 detected findings structurally traceable; graph/flat evidence sets equal | Provenance reachability, evidence constraints, correction changes and graph schema conformance; no measured clinical or reviewer benefit |

OCR required access outside the IDE sandbox to the existing user model cache. The final smoke and eight-image benchmark both passed with that access. This is an environment requirement, not evidence that arbitrary prescription photos are recognized correctly.

## Preliminary review experiment

The test split contains 96 synthetic text cases from 32 independent prescription groups. At a simulated review budget of 50%, impact-based review improved recovery of high-severity source findings over uncertainty-only review by **14.29 percentage points** on corrupted text. The paired prescription-group bootstrap 95% interval was **6.25 to 23.34 percentage points**, using 1,000 resamples and seed 17.

This is a preliminary, dataset-specific result. Labels come from local source tables, medication-entry boundaries are provided, and a simulated reviewer applies gold corrections. It does not measure independent clinical accuracy, real OCR performance, reviewer time, or patient outcomes. The comparison needs replication on the permitted expert-reviewed cases before supporting a conference effectiveness claim.

The research contribution to investigate is **prioritizing identity corrections by the interaction findings they could change**. The implementation now supports controlled comparisons, an original-engine baseline, severity ablation, review histories, and reproducible exports. Whether this contribution is novel requires a thorough related-work comparison; no priority claim is established here.

## Local artifacts

- [Exact fixture results](../artifacts/evaluation/results.json)
- [Review benchmark summary and manifest](../artifacts/research/summary.json)
- [Review figures and tables](../artifacts/research/)
- [Final OCR smoke](../artifacts/ocr_smoke/result.json)
- [Final image benchmark](../artifacts/ocr_benchmark/summary.json)
- [Live evidence check](../artifacts/live_evidence/result.json)
- [Explanation experiment status](../artifacts/explanations/summary.json)
- [Live explanation experiment](../artifacts/explanations-live-2026-10-04/summary.json)
- [Tracked paper handoff](PAPER_HANDOFF.md)
- [Graph conformance checkpoint](research_artifacts/graph_checkpoint_2026-10-04/summary.json)

Runtime artifacts are intentionally ignored by Git. A selected synthetic checkpoint is archived under `docs/research_artifacts/` for the writing team. Preserve each run's own manifest when sharing results. The live eight-request experiment had an estimated token cost of $0.00051315 using published standard rates; this is not an invoice or an expert faithfulness score.

## Still pending

Follow [manual validation](MANUAL_VALIDATION.md) for browser/accessibility/print checks, permitted real images, genuine expert annotations and claim-support ratings, key rotation, and container checks. The live Groq synthetic integration is now tested. Docker Desktop's Linux daemon was unavailable, so Linux lock resolution is not a successful container-runtime test. Remote CI has not been run. The 150-case private annotation workbook contains pending placeholders, not completed research data.

See [implementation status](IMPLEMENTATION_STATUS.md) for the full retained roadmap and [research protocol](RESEARCH_PROTOCOL.md) for dataset and experiment requirements.

## Hybrid upgrade verification, 2026-10-05

This checkpoint implements the agreed automatic candidate screening, bounded biological reasoning, and optional molecular/GraphSAGE research pipeline. The earlier checks above describe their original execution snapshots. New experiment summaries retain their own code/data hashes; subsequent edits do not rewrite historical manifests.

| Check | Executed result | Scope |
| --- | --- | --- |
| Fresh core-only installation | Dependency installation and `pip check` passed | No PyTorch, RDKit, SHAP, PyG, PaddleOCR or Groq installed in this verification environment |
| Full core-profile suite | 120 passed, 3 skipped in 10.94 seconds | Optional Torch model tests and two optional Groq contracts skipped; candidate joins, biology, graph/replay boundaries, recognition stress-case generation, calibration bookkeeping, fallbacks and UI run |
| Existing application environment | 122 passed, 1 skipped in 20.70 seconds | Includes installed Groq contract checks; optional Torch test module skipped |
| Full optional ML environment | 126 passed, 2 skipped in 22.08 seconds | Includes actual model, feature, split, calibration and inference tests; two Groq contracts skipped |
| Actual selected-model integration | 13/13 checks passed | Unknown-only gate, source authority, unavailable-artifact fallback, symmetric probabilities, typed graph, JSON replay and repeatability |
| Actual model-enabled Streamlit flow | 12/12 checks passed | Model selection survives navigation; toggling marks old results stale and blocks exports; rebuild, graph and Research workspace run |
| Source-conformance fixtures | 11/11 passed | Existing source record IDs and severity remain intact |
| Core-only startup | Streamlit health returned `ok` on port 8517; test server stopped | Physical application startup without ML dependencies |
| Biological snapshot | 93 sourced roles, 34 mapped drugs; integrity check passed | Bounded FDA clinical CYP/transporter roles, retained source conditions and whole-product exposure boundaries |
| Candidate combination products | Two new correctness cases passed | Internal ingredient findings survive uncertain single-entry combination products; no candidate finding becomes a resolved alert |
| Full text experiment | 500 base groups, 1,500 clean/corrupted observations executed | Related variants stay grouped; sizes 2/5/10/20/30 |
| Actual OCR experiment | 120 images from 40 base groups, zero execution failures | OCR executed with held-out fonts/layouts; no reference boundaries passed into inference |
| Final model comparisons | 24 corrected runs completed | Four models, pair/cold protocols and seeds 17/29/43; all 24 original full-batch runs preserved separately |
| Model explanations | SHAP 30/30 and GNNExplainer 30/30 executed | Seeded, balanced held-out sample; fidelity, sparsity, runtime and three-case stability recorded |
| Missing-input and error analysis | All 24 model reports exported | Molecular loss, graph loss, bounded biological-role loss, case-level predictions and high-severity undercalls |
| Model recognition stress test | 100 held-out pairs, 300 observations executed | Clean and controlled text corruption; actual local inference, source-existence gate and audited held-out provenance |

On 200 corrupted observations from 100 held-out text groups, high-severity source recall was **20.45%** for resolved identities, **99.26%** for a top-1 identity assumption, and **99.44%** for candidate-potential records. Candidate-potential precision was only **16.09%**, with **86.72** conditional records per prescription on average; **75.73** were absent from the generated reference prescription. Eleven of 883 stable predicates were incorrect (**1.25%**). These results show a substantial uncertainty/alert-burden tradeoff, not a generally superior automatic decision policy.

The full OCR run recorded CER **0.0180%** and WER **0.1049%**. On its 24 held-out images from eight groups, all screening methods recovered the same source findings, so these controlled images do not establish a candidate-propagation benefit. The original image `end_to_end_ms` omitted normalization; the later runner fixes that field, and no total-pipeline timing claim uses the original field.

Initial storage measurement: approximately 1,396 MiB for the isolated ML environment, 417 MiB for the core verification environment, and 39 MiB for model data/recognition artifacts before final training checkpoints. The host has 16 logical CPUs and approximately 16 GiB RAM; Torch training uses at most four threads. Exact final model timings and sizes are measured from completed runs.

The corrected minibatch trainer's sampled peak process working set was **489.63 MiB**, with a separate **10.82 MiB** Python launcher. This is process memory measured during this experiment, not a minimum RAM specification for Windows plus the application. A core-only release verifier executed ten source/fallback/export checks successfully; its fresh-process first request was 600 ms and warm five-request median 0.44 ms, excluding interpreter startup, OCR, UI rendering and external services.

A mathematical GraphSAGE implementation optimization reduced the first mean-aggregation width from 2,048 to 128. On the frozen 1,107-node graph, the first-layer-only CPU microbenchmark measured a **4.42×** median speed ratio (211.50 ms to 47.83 ms) with maximum output difference 2.09e-7. Value and gradient equivalence checks passed. This is neither a whole-application speed claim nor a new graph-learning algorithm.

The model-enabled three-entry verifier recorded **1.907 seconds** for its first request, **116.16 ms** warm median and **132.99 ms** warm p95 over 30 repeats. Its peak process working set was **389.68 MiB**, including native Torch allocations. These timings include source/artifact validation and CPU inference, and exclude interpreter launch, OCR, Streamlit rendering and live services. A separate post-training first-layer microbenchmark measured 4.28×; the earlier 4.42× run retains its own execution conditions.

### Executed source-severity comparisons

The immutable training snapshot contains **1,107 identity-verified molecules and 53,779 labeled pairs**. Only 28 molecules have documented roles in the bounded biological snapshot. The subsequently completed catalog contains 1,337 mapped ingredients and 602 exclusions across 1,939 attempted identities, with no remaining request errors. This expanded mapping was not substituted into the frozen comparisons.

| Model | Held-out pair macro-F1, mean ± SD | One-unseen macro-F1, mean | Both-unseen macro-F1, mean |
| --- | --- | --- | --- |
| Fingerprint MLP | 0.7888 ± 0.0116 | 0.5778 | 0.3856 |
| GraphSAGE | 0.7919 ± 0.0088 | 0.5394 | 0.4040 |
| Molecular/network fusion | 0.8170 ± 0.0050 | 0.5560 | 0.3751 |
| Fusion plus documented roles | 0.8178 ± 0.0098 | 0.5512 | 0.3815 |

Fusion improves this held-out pair comparison, but the fingerprint baseline performs best in the one-unseen condition. The small role-fusion difference does not establish a material biological-feature benefit. No clinical or unknown-label accuracy follows from these known-source labels.

Validation selected pair-protocol role fusion, representative seed 29, version `5a592aa163c3902a4989`. Its individual held-out pair macro-F1 is 0.8286. The proposed threshold passed the tuning-subset criterion but failed its independent check: **122 errors in 1,304 accepted cases**, with Wilson upper error **0.11058**, exceeding the agreed 0.10 limit. The qualified deployment therefore accepts **zero predictions**. Proposed coverage remains separately recorded; zero qualified coverage is not replaced by the proposal's coverage. Other model policies and all before/after calibration metrics remain reported.

For SHAP, mean target-probability decrease after influential-bit removal was 0.29455, versus 0.05774 for matched random removal. For the selected GNN explanation, feature removal decreased probability by 0.03618 versus -0.00039 for random removal; edge removal decreased it by 0.06412 versus 0.00020. GNN stability was weak: feature Jaccard 0.01754 and edge Jaccard 0.14216 on **three** repeated cases. These rationale measurements do not prove biological mechanisms or general explanation reliability.

The 100-pair recognition stress experiment recorded raw macro-F1 **0.8252** and high-class PR-AUC **0.8908** across its 300 related observations. Top-1 recognition recovered the original identities throughout these controlled variants. Candidate class-stability coverage was 100% on clean input, 0% on single-character corruption and 2% on mixed corruption; alternate interpretations often lacked mappings/records or disagreed. Qualified model coverage remained zero, with accepted classification error undefined. Related variants are not independent clinical observations.

Windows sandbox restrictions caused temp-directory fixture failures in an initial test attempt. The complete suites and final Streamlit check passed with workspace temp directories or authorized cache access; these were environment corrections rather than application fixes. Dependency consistency passed in all three installed environments, 75 Python files parsed successfully, and `git diff --check` passed. Remote CI, container runtime, free-cloud/GPU execution and external clinical evaluation remain unexecuted; human-study results are not required or implied by this release.

See [the hybrid checklist](HYBRID_TASK_CHECKLIST.md), [setup/demonstration guide](HYBRID_ENGINE.md), and [paper methods/results draft](HYBRID_PAPER_DRAFT.md). Raw executed summaries are under [text](../artifacts/hybrid/text/summary.json) and [OCR](../artifacts/hybrid/ocr/summary.json).

Final local evidence: [model comparison](../artifacts/ml/training-minibatch/summary.json), [recognition stress test](../artifacts/hybrid/model-recognition/summary.json), [release verification](../artifacts/hybrid-release/verification.json), [CPU request measurements](../artifacts/hybrid-release/model_integration_idle.json), and [paper bundle](research_artifacts/hybrid_checkpoint_2026-10-05/BUNDLE_MANIFEST.json).

The completed bundle contains 614 archived files plus its manifest, totaling 184,605,400 bytes (176.05 MiB). [Independent archive read-back](research_artifacts/hybrid_checkpoint_2026-10-05_VERIFICATION.json) verifies every file SHA-256 and selected model/dataset/frozen-mapping consistency. The completed research/verification footprint is **2.46 GiB**, excluding the pre-existing application/OCR environment, user OCR cache and temporary test directories. The selected deployment occupies 10.54 MiB; both model schedules, case-level predictions and the paper archive account for much of the retained experiment storage. [Post-bundle storage measurement](../artifacts/hybrid-release/storage_post_bundle.json).
