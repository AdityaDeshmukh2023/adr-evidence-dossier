# Hybrid engine: setup and demonstration

This release implements an automatic medication screening pipeline and a reproducible source-severity experiment. It combines uncertain medicine identities, molecular features, a training-only interaction network, sourced biological roles, and calibrated model abstention. Source findings, candidate-dependent conclusions, biological hypotheses, and model predictions are stored separately.

## What to explain to the teacher

“Our system starts from prescription text or an image. A misspelled medicine can represent several ingredients, so we screen every retained candidate combination and show which source conclusions remain stable and which depend on identity. For resolved medicines we preserve documented interaction records. A separate directed biological graph shows compatible enzyme/transporter paths, with source conditions. We train four models to investigate whether molecular and network features improve prediction of high/moderate/low **source severity**, including when drugs were unseen during training. Temperature scaling and a conservative error-bound policy allow the model to abstain. Every report records the data and model versions needed to reproduce it.”

The research question is whether molecular/network fusion improves calibrated source-severity prediction under unseen-drug, missing-feature and recognition conditions. Candidate propagation supplies a separate robustness experiment. A GNN or SHAP alone is an established method; this implementation does not establish a novelty or effectiveness claim before comparisons support it. No new human study is required to run these experiments, and no human-study results are implied.

## Environments

Core-only installation:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-core.txt
.\.venv\Scripts\python.exe scripts/doctor.py
.\.venv\Scripts\python.exe -m streamlit run llm.py
```

Separate research/training environment:

```powershell
python -m venv artifacts/ml-env
.\artifacts\ml-env\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\artifacts\ml-env\Scripts\python.exe -m pip install -r requirements-ml.txt pytest
```

For the exact tested Windows CPU environment (including the typed application), install [requirements-ml-windows.lock](../requirements-ml-windows.lock). The portable ML profile supports a separate GPU setup. The tested lock does not include the optional OCR or Groq profiles.

To run the application with model inference, install `requirements-core.txt` in that environment too, or install `requirements-ml.txt` into the application environment. The core profile intentionally does not include the ML profile. Training needs no Groq key. On a free GPU runtime, use [the notebook](../notebooks/hybrid_training_colab.ipynb), which calls the same CLIs; notebook creation does not count as a GPU run.

The installed research environment can run the model-enabled application directly:

```powershell
.\artifacts\ml-env\Scripts\python.exe -m pip install -r requirements-core.txt
.\artifacts\ml-env\Scripts\python.exe -m streamlit run llm.py
```

For a fresh checkout, the paper bundle contains the selected deployment. Point the application to it with:

```powershell
$env:HYBRID_MODEL_DIR = (Resolve-Path docs/research_artifacts/hybrid_checkpoint_2026-10-05/deployment).Path
.\artifacts\ml-env\Scripts\python.exe -m streamlit run llm.py
```

The final selected model is role fusion, seed 29. Its independent calibration check exceeds the agreed error limit, so application model decisions currently abstain. Source screening and sourced mechanism paths continue to work; raw severity comparisons and explanation evaluations remain available in the Research workspace and paper bundle.

OCR requires its optional profile in whichever environment starts the app. Core-only source screening remains usable when the research environment is absent. On the current 16 GB RAM / 16 logical CPU machine, training is limited to four Torch threads. The 24 corrected runs sum to 32.69 minutes of measured training time, excluding calibration, retrieval and explanation evaluation; the application performs inference without retraining.

The completed research/verification setup occupies **2.46 GiB**: ML dependencies 1,446 MiB, core verification environment 417 MiB, molecular/model experiments 461 MiB, recognition benchmarks 22 MiB, and the immutable paper bundle 176 MiB. The selected deployment directory is 10.54 MiB and is included in the experiment total. Reserve roughly 3–4 GiB beyond an existing application/OCR installation and user OCR cache. The measured model-enabled inference process peaks at about 390 MiB RAM; the sampled training process peaks at about 490 MiB. These process measurements do not include Windows, the IDE, browser or a separate OCR process. [Storage measurement](../artifacts/hybrid-release/storage_post_bundle.json).

## Data and training commands

Run these from the repository root. Molecular retrieval sends ingredient names to the public PubChem service, never prescription text or patient data. The importer caches provenance and exclusions; a repeated run resumes the cache. Biological refresh is explicit; ordinary application use reads its local snapshot.

```powershell
.\.venv\Scripts\python.exe scripts/import_biology.py --check
.\artifacts\ml-env\Scripts\python.exe scripts/prepare_molecules.py
.\artifacts\ml-env\Scripts\python.exe scripts/prepare_ml_dataset.py --molecules artifacts/ml/molecules_frozen.json
.\artifacts\ml-env\Scripts\python.exe scripts/train_ml.py --device cpu --output artifacts/ml/training-minibatch
.\artifacts\ml-env\Scripts\python.exe scripts/explain_ml.py --help
```

`prepare_ml_dataset.py` freezes eligible known-severity pairs and split audits before training. `train_ml.py` defaults to all four models (`fingerprint`, `graphsage`, `fusion`, `role_fusion`), pair and unseen-drug protocols, seeds 17/29/43, maximum 60 epochs and patience 8. Checkpoints are chosen using validation macro-F1. Calibration uses a separate partition; test labels do not choose deployment. Training summaries include metrics, calibration, missing-modality results and execution hashes.

The release comparison uses the earlier immutable mapping of 1,107 molecules and 53,779 labeled pairs. Retrieval subsequently completed the catalog, mapping 1,337 of 1,939 ingredients; the expanded cache is a separate snapshot and was not substituted midway through training. For an additional experiment on that expanded cache, select `artifacts/ml/molecules.json` explicitly and use a new dataset/training output directory.

To reproduce the release dataset from the archived mapping in a fresh checkout:

```powershell
.\artifacts\ml-env\Scripts\python.exe scripts/prepare_ml_dataset.py --molecules docs/research_artifacts/hybrid_checkpoint_2026-10-05/molecules/molecules_frozen.json --output artifacts/ml/reproduced-dataset
.\artifacts\ml-env\Scripts\python.exe scripts/train_ml.py --dataset artifacts/ml/reproduced-dataset --output artifacts/ml/reproduced-training --deployment artifacts/ml/reproduced-deployment --device cpu
```

The pair split uses 70/10/10/10 train/validation/calibration/test. Half the training pairs supply untyped message-passing edges; the other half supply supervised severity labels. Reversed or held-out pairs are excluded from adjacency. The unseen-drug protocol partitions identities and reports one-unseen and both-unseen conditions separately. Equivalent full InChIKeys share a cold-drug partition, so fractions are approximate at chemical-group boundaries. Unknown source-severity records are excluded from supervision, and missing records never become negative interaction labels.

The release trainer uses pair minibatches of 2,048. The original one-update-per-epoch experiment is preserved separately: with patience 8 it usually stopped before learning minority classes. A training/validation-only 60-epoch diagnostic justified correcting the optimizer schedule; test labels did not motivate the correction. Check the executed summary for the exact schedule and number of updates. Mean GraphSAGE projects neighbor features before mean aggregation using a linear transform without bias; this is mathematically equivalent and reduces the first sparse aggregation width from 2,048 to 128. It is an implementation optimization rather than a new graph algorithm.

Within the dedicated calibration partition, separate subsets fit temperature, tune the threshold, and independently check the selected policy. At least 100 accepted cases and a nominal 95% Wilson upper error bound of at most 10% are required; otherwise the policy abstains. Adaptive threshold selection is not a clinical risk guarantee, and known-label calibration does not validate predictions on actual unknown-severity records.

Features are Morgan radius 2, 2,048 bits. Drug IDs join datasets but are not learned predictive features. Source interaction text and severity labels do not become node features. Missing biological documentation is represented explicitly and is not a negative biological label. Mapping conflicts, unsupported mixtures and unmapped structures remain available for source lookup but are excluded from molecular supervision.

## Application behavior

1. Open **Review desk** and choose **Known source interaction**: warfarin/aspirin retains its documented severity.
2. Choose **Ambiguous medicine reading**: the candidate panel shows all retained alternatives and stable/conditional source states. Lexical scores are similarities, not identity probabilities.
3. Choose **Biological pathway**: inspect the sourced directed roles and applicable conditions for clarithromycin/midazolam. A pathway does not assign clinical severity.
4. Choose **Supplement pathway**: inspect the bounded, whole-product St. John’s wort role; negated or uncertain exposure produces no consumed-exposure hypothesis.
5. Choose **Unknown source severity** and enable the local model. Eligible pairs receive an experimental estimate or an explicit abstention. Missing dependencies, mapping, calibration or checkpoints have explicit statuses. Existing source severities remain intact.
6. Download redacted JSON/HTML, inspect the separate evidence-graph relationships, and replay JSON in **Research workspace** using matching artifacts.

`HYBRID_MODEL_DIR` defaults to `artifacts/ml/deployment`. The application caches CPU checkpoints; it never trains during reruns. A masked known-pair experiment is a research evaluation, not an automatic finding on an unrecorded application pair. Original text is omitted by default; downloaded medication information still deserves careful handling.

## Benchmarks and release artifacts

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\artifacts\ml-env\Scripts\python.exe -m pytest tests/test_ml_splits.py tests/test_ml_models.py tests/test_ml_inference.py -q
.\.venv\Scripts\python.exe scripts/hybrid_benchmark.py --groups 500
.\.venv\Scripts\python.exe scripts/hybrid_ocr_benchmark.py --groups 40
.\artifacts\ml-env\Scripts\python.exe scripts/verify_hybrid_release.py
.\artifacts\ml-env\Scripts\python.exe scripts/verify_hybrid_ui.py
.\artifacts\ml-env\Scripts\python.exe scripts/hybrid_model_robustness.py --count 100
.\artifacts\ml-env\Scripts\python.exe scripts/score_ml_robustness.py
.\artifacts\ml-env\Scripts\python.exe scripts/export_ml_errors.py
.\.venv\Scripts\python.exe scripts/prepare_hybrid_bundle.py --original-model-directory artifacts/ml/training --corrected-model-directory artifacts/ml/training-minibatch --diagnostic-directory artifacts/ml/diagnostic_fullbatch --model-recognition-directory artifacts/hybrid/model-recognition --include-images
```

The text runner evaluates 500 base groups with related clean/corrupted variants. The image runner executes OCR on 120 generated images from 40 base prescriptions, preserving font/layout holdouts. Metrics distinguish candidate identity coverage, exact high-severity source recovery, incorrect stable predicates, conditional burden and runtime. These are controlled software experiments rather than clinical validation.

The [recognition protocol](HYBRID_RESEARCH_PROTOCOL.md), [biological data documentation](BIOLOGICAL_DATA.md), [architecture diagram](hybrid_architecture.mmd), and [executed verification record](VERIFICATION_RESULTS.md) explain scope and results. Run outputs live under ignored `artifacts/`; selected synthetic artifacts are archived in `docs/research_artifacts/` with immutable execution manifests. Dependency profiles and CI configuration support reproducibility; CI/container execution is reported only when actually run.
