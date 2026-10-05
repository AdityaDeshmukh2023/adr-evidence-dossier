# Hybrid core-engine upgrade

The implementation separates known interaction records, findings conditional on uncertain medicine identities, and research predictions for source records whose severity is explicitly unknown. The intended research contribution is a molecular/network model for conditional severity, evaluated under unseen drugs, missing features and noisy medicine identities. The candidate-propagation benchmark is a separate robustness experiment that requires no clinical reviewer in the runtime loop.

## Components and claims

| Component | Concrete behavior | Evidence needed |
| --- | --- | --- |
| Candidate propagation | Carry retained ingredient candidates through pairwise source checks instead of discarding unclear entries or silently choosing the first spelling match. | Finding recovery and conditional-alert burden under held-out corruption. |
| Stable predicates | Report a source severity state as stable only when it occurs in every retained candidate combination for a prescription-entry pair. | Incorrect stable-predicate rate; candidate identity coverage. |
| Factorized evaluation and cache | Evaluate candidate combinations for each entry pair and cache repeat computations. | Output equivalence, cold/warm response-cache latency, scaling and allocation samples. |
| Biological context | Attach explicitly sourced target/mechanism information where imported evidence exists. | Import coverage, identifier joins and evidence traceability; not invented pharmacology. |
| Optional research model | Predict conditional severity only for documented source interaction records whose source severity is unknown, when a checkpoint exists. | Held-out severity labels, unseen-drug evaluation, calibration, feature ablations and an explicit model card. |

The uncertain-identity layer does not infer a medicine identity from whether the resulting combination appears dangerous. Candidate generation remains independent of interaction outcomes. An exact source finding, a stable severity predicate, and a conditional-severity model prediction are different outputs and must remain visibly separate. The model does not predict interaction existence for pairs with no source record.

For an entry with candidates A/B and a partner C, both A-C and B-C may have high source severity. This makes the proposition "this entry pair has a high-severity source finding" stable across retained candidates. It does not identify whether A or B was prescribed, and neither exact source pair becomes confirmed from that predicate.

## Implementation sequence

1. Keep resolved source findings intact and add candidate assessments as a separate result field.
2. Add source-conditioned outcome enumeration, stable/possible states, unusable-identity status and an observable cache.
3. Build grouped synthetic text and actual image benchmarks, with reference identities used only for scoring.
4. Import biological evidence and model datasets through reproducible, independently documented scripts.
5. Train and calibrate the optional model; record a checkpoint and held-out metrics before enabling predictions.
6. Preserve executed summaries and hashes in a paper handoff bundle.

Code availability is not evidence that a benchmark or full training run has executed. Each experiment writes its own execution status and original source/code manifest. A dataset-generation run explicitly records that inference has not executed. Do not mark a full training task complete merely because a trainer or notebook exists.

## Evaluation commands

Run from the repository root with the project virtual environment:

```powershell
.\.venv\Scripts\python.exe scripts/hybrid_benchmark.py --groups 500 --output artifacts/hybrid/text
.\.venv\Scripts\python.exe scripts/hybrid_ocr_benchmark.py --groups 40 --output artifacts/hybrid/ocr
.\.venv\Scripts\python.exe scripts/prepare_hybrid_bundle.py --original-model-directory artifacts/ml/training --corrected-model-directory artifacts/ml/training-minibatch --diagnostic-directory artifacts/ml/diagnostic_fullbatch --model-recognition-directory artifacts/hybrid/model-recognition --include-images --dry-run
```

For a quick implementation check, use five text groups and two image groups in a separate output directory. Such checks do not supply the requested full-run evidence. The OCR runner requires installed English PaddleOCR models and distinct development/calibration/test fonts; it fails explicitly if fonts are missing instead of silently destroying the font holdout.

The text default is 500 base prescription groups, each with clean and two corrupted variants. Sizes cycle through 2, 5, 10, 20 and 30 entries. The image default is 40 groups with three visual variants each, yielding 120 images, with no patient content or handwriting. Neither dataset establishes performance on real prescriptions.

The [Colab notebook](../notebooks/hybrid_training_colab.ipynb) runs the repository's optional molecular-data preparation and model trainer in a separate research environment. It compares pair/cold-drug protocols, four model choices and three seeds. It verifies a project checkout and source CSV before executing; the notebook file itself is not an executed training artifact. Downloaded molecule caches are resumable and their coverage must accompany model metrics.

The [engine guide](HYBRID_ENGINE.md) explains the separate core and research environments, device selection, storage and demonstration. The final paper bundle preserves the frozen training molecule cache and completed catalog cache separately, both training schedules, calibration and explanation records, and the selected CPU checkpoint. Validate it without writing files using:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_hybrid_bundle.py --original-model-directory artifacts/ml/training --corrected-model-directory artifacts/ml/training-minibatch --diagnostic-directory artifacts/ml/diagnostic_fullbatch --model-recognition-directory artifacts/hybrid/model-recognition --include-images --dry-run
```

The [October 5 bundle](research_artifacts/hybrid_checkpoint_2026-10-05/BUNDLE_MANIFEST.json) has now been written and independently verified. This command can validate the same preserved run inputs again; select a new `--output` directory when an experiment changes. The bundle checks that its public molecule cache matches the frozen dataset and that the selected checkpoint's file hashes and dataset version agree. Existing historical archives are never silently replaced.

## Paper positioning

Possible title: **Molecular and Network Models for Conditional Interaction Severity under Unseen Drugs and Noisy Medication Identities**.

Primary question: do molecular/network features improve conditional source-severity classification under unseen-drug evaluation, and how does performance change with missing features and uncertain medicine identities? The candidate benchmark separately asks whether propagation recovers more clean-source interaction findings under noisy inputs, and what conditional-alert burden and latency it introduces. Literature comparison is still required before claiming novelty.

Conformal answer sets and graph interaction prediction are established approaches, not original inventions of this project: [conformalized graph answer sets](https://aclanthology.org/2025.naacl-long.32/), [Decagon](https://pmc.ncbi.nlm.nih.gov/articles/PMC6022705/). If model claims concern unseen medicines, include explicit drug-held-out evaluation because random pair splits can misrepresent that capability: [generalization study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8489092/).

Do not claim interaction-existence prediction, individual clinical risk, guaranteed medicine identity, or safety from an absent source record. Model probabilities describe severity conditional on a documented interaction; spelling scores describe lexical ranking. Known source severities remain authoritative, and unknown source labels remain unchanged alongside any separate research prediction.
