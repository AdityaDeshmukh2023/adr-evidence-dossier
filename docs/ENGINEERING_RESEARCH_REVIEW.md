Project engineering and research review — 3 October 2026
=======================================================

This review covers the current working tree, including existing uncommitted changes. It is an engineering assessment and proposed study design, not clinical validation. Application code was not changed. Findings distinguish executed checks, source inspection, and proposals that still require experiments.

The project is a known-interaction screening and evidence-display prototype. Its active path is Streamlit → medication normalization → local pair lookup / food rules → optional live evidence → deterministic or Groq explanation → graph and HTML report. It does not train a model, estimate patient-specific adverse-event probability, discover interactions, or establish causality. The graph is a visualization, not a graph neural network. The LLM explains results; it does not generate the engine's alerts.

The separation between deterministic detection and optional explanation is worth preserving. Modular core code, offline operation, source hashes, unordered pair deduplication, conservative no-match wording, and synthetic fixtures provide a useful foundation. The main weaknesses are information loss, inconsistent identifiers, incomplete evidence provenance, and inadequate evaluation.

The local dataset and executed checks
------------------------------------

The processed lookup contains 1,939 drug names and 160,235 unordered pairs. Stored metadata reports 222,383 raw rows and 62,148 duplicates removed. Severity counts are 26,914 high, 96,675 moderate, 6,833 low, and 29,813 unknown. These are database labels, not patient outcomes or a machine-learning training result. All eight raw-file SHA-256 hashes match the recorded metadata. Rebuilding the cleaned data and lookup in memory reproduced these counts and exactly matched the stored lookup, without rewriting the dataset.

Commands run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
.\.venv\Scripts\python.exe scripts\evaluate.py
```

Results: eight unit tests passed; all four synthetic evaluation cases passed. Unscoped pytest collection failed because an unrelated `pytest-cache-files-xf2_sk77` directory is inaccessible. Explicitly targeting `tests` succeeded. Configure test discovery rather than treating the directory issue as an application failure.

Additional offline probes exercised real engine behavior. OCR was checked with a 3.x-shaped mock and installed package source; live-evidence attachment was checked with mocked evidence. No prescription images were sent to external services, no paid LLM evaluation was run, and OCR recognition quality, browser behavior, deployment, and clinical validity were not tested.

| Probe | Observed behavior |
|---|---|
| `Warfarin 5 mg` and `Aspirin 75 mg` on separate lines | One alert |
| `Warfarin 5 mg Aspirin 75 mg` on one line | Only warfarin retained; zero alerts and no unsupported-name warning |
| `Tab. Warfarin 5 mg` followed by aspirin | Warfarin entry unrecognized; zero alerts |
| `warfarin daily` followed by aspirin | Warfarin entry unrecognized; zero alerts |
| Simvastatin with `no grapefruit` | Food alert despite negation |
| Warfarin with `kaleidoscope` | Food alert because `kale` is a substring |
| Simvastatin with `grapefruit,grapefruit` | Duplicate alerts |
| Unrecognized medicine exported to HTML | Unrecognized-name warning missing from report |
| Abacavir + naltrexone with mocked live evidence | Status updated to live, but alert evidence count remained one |
| OCR adapter given a representative 3.x result dictionary | Returned `n e e`, not recognized medication text |

Prioritized defects and corrections
-----------------------------------

1. **High: aspirin aliases disconnect from the dataset.** In `adr_system/normalization.py:10`, acetylsalicylic acid maps to aspirin. The full index contains `acetylsalicylic acid` / `DDInter20`, but no `aspirin` key. The small `KNOWN` set still declares aspirin recognized, so the problem is hidden. The dataset contains 654 pairs involving that ID. Enumerating its high-severity pairs through the current engine found 43 with no alert, including aspirin + apixaban and aspirin + ketorolac. These are disagreements with the local source, not independent clinical findings. Resolve synonyms to stable ingredient IDs, use IDs for matching, and separate preferred display names from identities. Add an alias-invariance regression test over source records.

2. **High: medication extraction silently drops information.** `adr_system/normalization.py:27` deletes everything after the first recognized dose. `paddle_ocr.py:41` flattens OCR lines into spaces, making this especially consequential. Strength and frequency fields exist but are never populated. Preserve line geometry and original spans, extract every medication entity, store dose/route/frequency separately, and require review when text remains unexplained. Do not silently fuzzy-match uncertain names.

3. **High: the OCR adapter uses the wrong output contract.** Requirements and the installed environment use PaddleOCR 3.2.0. Installed `ocr()` delegates to `predict()`, and OCR results expose `rec_texts` and `rec_scores`. The adapter loops over results as legacy nested lists. Its 3.x-shaped mock returns characters from dictionary keys. Update the adapter to the supported result schema, preserve confidence and bounding boxes, and validate it with an actual synthetic image. Cache model initialization and record model/checkpoint identity. Changing CPU flags alone does not repair result parsing. The upstream migration guide documents the API change. [PaddleOCR migration guide](https://paddlepaddle.github.io/PaddleOCR/v3.0.0/update/upgrade_notes.html)

4. **High: displayed confidence is unsupported.** `adr_system/engine.py:41` assigns 1.0 to every full-dataset alert. Demo and food rules contain other fixed values without a calibration experiment. `llm.py:102` displays these as percentages. Exact source lookup does not imply 100% certainty of harm. Replace the percentage with explicit match status and source evidence grade. If probabilistic confidence is later introduced, define its target and calibrate against held-out labels. Keep OCR certainty, entity-linking certainty, evidence quality, and clinical risk separate.

5. **High: live evidence does not attach reliably.** `llm.py:85` retrieves using lowercase canonical names, then joins against source display names such as `Abacavir`. The case-sensitive lookup fails. A mocked invocation of the actual function confirmed the mismatch while source status still said live. Join by stable IDs and test both attachment and source-status semantics.

6. **High: prompt instructions are presented as a stronger guarantee than implemented.** `adr_system/explanations.py:21` sends a Python dictionary representation to Groq and accepts free text without claim or citation validation. Temperature zero is not a factuality guarantee. Introduce a minimal structured payload, stable claim/evidence IDs, schema-validated output, explicit token-budget handling, and validation of cited IDs and asserted severities. Reject invalid output and use the deterministic explanation. Citation existence alone does not establish that the cited text supports a claim; evaluate support separately with reviewers.

7. **High: raw text crosses the external boundary when an API key is present.** The LLM payload is `result.to_dict()`, which includes original medication strings, unsupported entries, and foods. If users paste identifiers or OCR headers, these can be transmitted. There is no separate LLM opt-in corresponding to the live-evidence toggle. Use an explicit external-explanation control and an allowlist of normalized alert data. Provide session clearing and a truthful distinction between temporary local processing, session memory, downloaded reports, and external requests. Tracked image files also exist; their contents were not examined in this review, so their provenance must be established before public release.

8. **Medium: food matching lacks linguistic context.** `any(term in food ...)` in `adr_system/engine.py:55` causes substring false positives, ignores negation, and allows duplicates. Use normalized food concepts, token/phrase boundaries, deduplication, and explicit handling of uncertain consumption. Quantities and timing should be captured only when supported by the underlying rule. Expand the four curated rules through reviewed evidence rather than unrestricted generation.

9. **Medium: evidence is not sufficiently specific.** Several curated URLs point to the DailyMed homepage or broad searches. Full records link to a download page. `fetch_openfda_label` selects one unsorted record and the first 700 characters; this neither establishes recency nor ensures relevance to the pair. PubChem molecular formula and weight support identity, not an interaction mechanism. Store exact label/set identifiers, versions, relevant sections, retrieval time, excerpt hashes, and the claim each passage supports. openFDA describes product-dependent label variation and cautions that returned labeling is not necessarily the labeling on distributed products. [openFDA labeling documentation](https://open.fda.gov/apis/drug/label/)

10. **Medium: full lookup overrides useful curated explanations.** The `continue` in `adr_system/engine.py:43` prevents reviewable curated text from accompanying an overlapping full-dataset record. Clopidogrel + warfarin is a current overlap. Keep authoritative source severity separate from supplementary mechanism evidence; merge provenance without silently resolving disagreements.

11. **Medium: reports omit important context.** `adr_system/report.py:17` exports raw inputs and source status but omits structured warnings, unsupported-name status, and per-alert record/version details. A random UUID is a run identifier, not a reproducible experiment specification. Export normalized IDs, unresolved inputs, completeness status, full artifact hashes, software revision, configuration, prompt/model versions, and evidence record IDs. Keep any original input optional and clearly identified.

12. **Medium: repeated calls and indefinite caches affect behavior.** `render_alerts` invokes the LLM on Streamlit reruns. API calls are sequential across drugs; success and failure results are cached without expiry, and an unavailable enrichment status prevents session retry. Cache explanations by immutable evidence/configuration hash, use explicit refresh, TTL and retry policies, bounded concurrency, and clear latency/error reporting. Capture one explanation per analysis rather than silently regenerating report text.

13. **Medium: deployment and maintenance need a reproducible baseline.** Most dependencies have only minimum versions; OCR's transitive stack remains unpinned. There is no test-discovery configuration, dependency lock, or CI workflow in the reviewed tree. `.gitignore` omits `.venv/`; `.Dockerfile` copies the directory without a `.dockerignore`; the devcontainer launch disables CORS and XSRF protection. Lock a tested environment, separate optional OCR dependencies, exclude local environments/secrets/data from container context, retain normal request protections, and run core tests in CI. Docker execution itself was not tested.

14. **Medium: HTML interpolation is inconsistent.** Downloaded HTML escapes text, but alert cards in `llm.py:102` interpolate entities into `unsafe_allow_html=True`; food entities can contain user input. Escape dynamic values or use native Streamlit components. This is an HTML-injection surface; JavaScript execution was not demonstrated.

15. **Research blocker: the evaluation measures a handful of expected outputs.** `scripts/evaluate.py` checks only alert count and maximum severity. It can miss the wrong interacting pair, missing citations, OCR failures, and incorrect extra findings that preserve those aggregates. It also prints failure counts without returning a failing process status. Compare exact pair/type/severity/provenance sets, then evaluate against independently annotated cases. Four passing cases must never be described as 100% clinical accuracy.

Which methods should change
--------------------------

| Current method | Assessment | Recommended direction |
|---|---|---|
| Deterministic source lookup | Appropriate for screening known interactions | Keep it; repair ID mapping, coverage reporting, and provenance |
| Small alias dictionary + dose regex | Too brittle for realistic prescriptions | Terminology-backed ingredient linking, structured extraction, ambiguity review |
| Legacy PaddleOCR output parsing | Incompatible with the installed 3.x contract | Use structured results and benchmark recognition separately |
| YAKE keyword extraction in `preprocess.py` | Generic keywords are not medication entities; module is inactive | Archive it or benchmark an entity extractor against it |
| Per-document TF-IDF in `context.py` | Inactive; fitting IDF to each query/document pair produces inconsistent retrieval scores | Corpus-wide lexical baseline, then compare hybrid retrieval and reranking |
| arXiv-only retrieval with 200-token truncation | Inactive; limited source suitability and lost evidence context | Task-specific label/literature corpus with passage provenance |
| FoodData Central top-result nutrients | Inactive; nutrient lookup is not interaction evidence | Reviewed drug–food relation data; use nutrition lookup only for an explicit separate task |
| LLM instructions without output checks | Insufficient to establish factual grounding | Structured claims, evidence validation, fallback, and measured error rates |
| Static pairwise graph | Useful display, limited research novelty | Better filtering and traceability; learned graphs only for a separately defined prediction study |

`context.py`, `context_FoodDataCentral.py`, and `preprocess.py` are not called by the active Streamlit path. Their dependencies are also not fully declared in current requirements. The paper must describe what actually runs. TF-IDF, rules, and Streamlit are not inherently obsolete; replacing them without a measured benefit is not modernization.

RxNorm offers terminology matching, including approximate candidates. Use candidates as suggestions with ambiguity handling, not automatic clinical identity decisions. Add a reviewed local brand/combination dictionary if the intended setting requires names outside its coverage. RxNav's interaction API was discontinued in January 2024; do not plan to use it as the DDI source. [RxNorm matching API](https://lhncbc.nlm.nih.gov/RxNav/APIs/api-RxNorm.getApproximateMatch.html), [NLM RxNav FAQ](https://lhncbc-portal.lhcaws-prod-pub.nlm.nih.gov/RxNav/information/FAQs.html)

The local snapshot label `2026-09-01` is a project version, not proof of upstream freshness. The DDInter 2.0 paper reports expanded coverage and additional food, disease, and therapeutic-duplication interactions. Audit the local export's upstream release and acquisition process. Compare its schema and actual records before migrating; the paper's total records cannot be compared directly with this repository's deduplicated unordered-pair count. [DDInter 2.0 primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11701621/)

A focused research contribution
------------------------------

Recommended working title: **“Evidence-Grounded Medication Interaction Screening from Noisy Prescription Text: Evaluating Entity Linking, Abstention, and Explanation Faithfulness.”**

The proposed contribution is a measured pipeline that preserves medication information, exposes uncertainty, and connects each explanation to evidence. This is a hypothesis and study direction, not an established novelty claim. A broader literature review is necessary: ExDDI already studies natural-language explanations for DDI predictions, while TextDDI studies prediction involving new drugs. Merely combining an LLM and DDInter is not enough to claim novelty. [ExDDI, 2024 preprint](https://arxiv.org/abs/2409.05592), [TextDDI, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.918/)

Proposed research questions:

- Does terminology-aware linking improve medication and alert recall under OCR errors, abbreviations, brand names, and combination products?
- Does explicit abstention on ambiguous entities reduce wrong-drug mappings at an acceptable review burden?
- Do retrieved, source-specific passages and claim checks reduce unsupported explanation statements compared with the existing prompt-only explainer?
- Does displaying unresolved inputs and evidence gaps help reviewers detect incomplete analyses?

The proposed architecture is: image/text → structured OCR spans → medication entities → ingredient IDs and review decisions → deterministic interaction records → relevant evidence passages → validated explanation → results/report. Carry a completeness status through every stage. Distinguish “not assessed,” “unresolved medicine,” “no record in this snapshot,” and “source severity unknown.” None is synonymous with safe.

Build the first research version around three additions: reliable entity linking, source-specific explanation validation, and a benchmark that measures their effects. Disease context, therapeutic duplication, multilingual OCR, and local brands are useful extensions when the intended population and annotation resources justify them. Individualized dosing or clinical risk prediction requires different evidence and outcomes; it should not be inferred from generic severity labels.

Evaluation protocol to implement
-------------------------------

Start with a feasible pilot of approximately 200–500 independently reviewed cases and a separately annotated image subset. This is a planning range, not a statistical sample-size justification. Use pilot error rates and the desired precision/effect size to determine the final sample. Synthetic data can establish controlled robustness but cannot establish clinical effectiveness or real-world handwriting performance.

Include typed and image inputs, severity strata, no-record cases, unknown severity, unresolved drugs, aliases, combination medicines, duplicate drugs, prefixes, decimals, negation, dietary ambiguity, and multi-line/layout corruption. Include realistic 2–10 medicine lists where feasible. Label drug spans, ingredient IDs, known source pairs, severity, exact evidence passages, and explanation support separately.

Use two independent domain reviewers and adjudication for clinical/evidence annotations, with written instructions and agreement reporting. Keep augmented versions of the same prescription, source template, and writer/patient group together across splits. Freeze train/development/test partitions before tuning thresholds. Maintain dataset version, provenance, permissions, and a data dictionary.

| Comparison | What it establishes |
|---|---|
| Current implementation vs corrected deterministic implementation | How much improvement comes from fixing software defects |
| Corrected baseline vs terminology-aware linking | Contribution of entity resolution |
| Corrected baseline vs proposed linking with abstention | Error/review-burden tradeoff |
| Deterministic template vs existing LLM vs evidence-retrieving LLM vs validated LLM | Explanation benefit and faithfulness cost |
| Clean text vs OCR text vs manually corrected OCR | Where recognition errors propagate |
| Full proposal with one component removed at a time | Ablation evidence for each contribution |

| Stage | Metrics |
|---|---|
| OCR | Character/word error rate and medication-name recall, stratified by document type |
| Extraction/linking | Span precision/recall/F1, ingredient-ID accuracy, wrong-drug mapping rate, abstention rate |
| Screening | Exact interaction precision/recall, high-severity recall, severity confusion matrix, completeness reporting |
| Explanation | Unsupported-claim rate, citation correctness, evidence support, omission rate, blinded reviewer assessment |
| Operations | End-to-end and per-stage median/p95 latency, failure rate, memory, external requests, cost per case |
| Human review | Time to resolve ambiguous inputs, correction rate, and ability to recognize incomplete results |

Report denominators and confidence intervals. Resample at the independent prescription/group level rather than treating correlated pairs or image variants as independent. Predefine the primary endpoint and use paired comparisons because methods see the same cases. Report performance on abstained cases and coverage together to avoid artificially high accuracy obtained by refusing difficult cases.

Use an independent reference to assess clinical correctness. Comparing lookups to the same database is a software-conformance experiment, not external validation. A missing edge is unlabeled, not a confirmed negative; “unknown” severity is also not a negative. If a prediction model is later added, use an explicit label/negative strategy and evaluate unseen-drug or temporal splits. Remove test edges from all graph-derived features. Do not train a severity classifier on record fields that directly reveal the target.

For LLM experiments, freeze prompts and retrieval snapshots, record provider/model identity and decoding settings, and retain generated outputs for audit. Report variability across repeated calls where applicable. Follow relevant TRIPOD-LLM reporting items for the explanation study; the guideline does not itself confer clinical validity. [TRIPOD-LLM, Nature Medicine 2025](https://www.nature.com/articles/s41591-024-03425-5)

Implementation order and paper deliverables
------------------------------------------

| Stage | Deliverable | Completion evidence |
|---|---|---|
| 1. Correctness | ID-based matching, repaired OCR adapter, no silent medication loss, accurate reports, evidence attachment | Reproduced defects fixed with targeted regression checks |
| 2. Provenance and boundaries | Specific citations, explicit external processing, truthful confidence/completeness, environment lock | Replayable offline analysis and reviewed outbound payload |
| 3. Benchmark | Annotation manual, reviewed cases, frozen split, stage-level metrics | Auditable labels and baseline results |
| 4. Research methods | Terminology-aware linking, abstention, retrieval and claim validation | Paired comparisons and ablations against the corrected baseline |
| 5. Manuscript | Methods, results, uncertainty analysis, limitations, reproducibility package | Every empirical claim traceable to an experiment/artifact |

The manuscript should explain the problem and intended users, distinguish screening from ADR prediction, compare related work, describe data acquisition and IDs, specify algorithms and uncertainty handling, report benchmark construction, present component and end-to-end results, analyze failures, and discuss generalizability. Include an architecture figure, source/data table, baseline/ablation table, error taxonomy, and latency/coverage tradeoff plots.

Current defensible claims are limited to implemented functionality, source-derived dataset statistics, and passing software checks. Do not yet claim patient-specific prediction, clinical validation, calibrated risk, hallucination-free output, prevention of adverse events, or state-of-the-art performance. A research paper becomes stronger through an independent benchmark and a clear measured contribution, not through a longer feature list.
