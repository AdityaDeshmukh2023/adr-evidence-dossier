# Medication Review Dossier

An educational, evidence-linked hybrid medication interaction prototype. It automatically screens retained medicine candidates, preserves known source findings, derives sourced enzyme/transporter possibilities, and optionally predicts source severity using locally trained molecular/GraphSAGE models. It produces reproducible reports. It does not estimate individual clinical risk or recommend treatment changes.

## Run on this Windows environment

```powershell
.\.venv\Scripts\python.exe scripts\doctor.py
.\.venv\Scripts\python.exe -m streamlit run llm.py
```

For a fresh Python 3.12 Windows installation:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.lock
```

`requirements-core.txt` installs the typed-input app. Add `requirements-research.txt`, `requirements-ocr.txt`, and `requirements-llm.txt` for their respective features; `requirements.txt` includes all four. `requirements-windows.lock` captures the tested local environment. The resolved Linux locks are used by CI and `.Dockerfile`; Linux execution still needs CI/container verification.

The optional ML profile is separate: [hybrid setup, training and demonstration guide](docs/HYBRID_ENGINE.md). Install `requirements-ml.txt` in a research environment, or use `requirements-ml-windows.lock` for the tested Windows CPU app/training environment. Core screening works without PyTorch, RDKit or checkpoints. Training runs through CLI commands and never starts on a Streamlit rerun. Models predict **high/moderate/low source severity for documented interactions**, with calibrated abstention; unrecorded pairs retain their original no-record status.

## Workflow

1. Enter medication text or upload a cropped English printed prescription image.
2. Build a dossier and inspect source findings, stable/conditional candidate assessments and biological paths.
3. Confirm candidate identities, enter verified ingredients manually, or exclude non-medication text.
4. Inspect source records and coverage. Enable the local research-model option for documented pairs with unknown source severity when compatible artifacts are installed. Optional external services are explicitly enabled by the user.
5. Download redacted HTML/JSON, or explicitly include original text. Replay JSON from Research workspace using the same software/data version.

The **Interaction map** workspace includes a typed **Evidence graph** explorer alongside the pairwise map. Trace finding provenance, inspect candidate-dependent possibilities, review before/after changes, and download graph JSON. The graph runs locally and explanation evidence is retrieved through its allowed support relationships.

Candidate ranking uses lexical similarity. Interaction impact changes review order, never the proposed identity. Missing records, unknown severity, and unassessed pairs remain distinct. Exact matches remain reviewable.

## Automated verification and research artifacts

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\evaluate.py
.\.venv\Scripts\python.exe scripts\benchmark.py
.\.venv\Scripts\python.exe scripts\ocr_smoke.py
.\.venv\Scripts\python.exe scripts\ocr_benchmark.py
.\.venv\Scripts\python.exe scripts\explanation_benchmark.py
.\.venv\Scripts\python.exe scripts\graph_benchmark.py
```

Outputs go under ignored `artifacts/`. The review benchmark defaults to 120 synthetic text cases grouped into 40 source prescriptions, with 96 cases in the test split. The image benchmark executes real OCR on eight synthetic images. Neither is clinical validation. The explanation command defaults to offline templates; add `--live` only when a Groq key is configured and API use is intended.

For expert annotation:

```powershell
python scripts/prepare_annotations.py --count 150
python scripts/benchmark.py --dataset data/private/annotations.json --output artifacts/expert-study
```

The evaluator rejects incomplete expert annotations. Read [the research protocol](docs/RESEARCH_PROTOCOL.md) and [the expert review packet](docs/EXPERT_REVIEW_PACKET.md) before collecting or labeling cases. If the pending workbook already exists, preserve it; the generator deliberately refuses to overwrite annotations.

## Privacy and external services

Core screening works without network services. Prescription images are handled locally and kept only in session. OCR may download public model weights on first use. Downloaded reports can still contain sensitive medication information even with original text omitted.

Copy `.env.example` to `.env` only if optional Groq explanations are needed. No key enables external explanation automatically: the UI checkbox is also required. Model requests include only allowlisted source records. Public evidence retrieval sends normalized ingredient names, not raw text or images. PubChem formula/weight data are identity information, not interaction evidence.

The low-cost default is `GROQ_MODEL=openai/gpt-oss-20b`. It uses low reasoning effort and strict JSON output, followed by source-ID, severity, citation, and exact-quote checks. Requests remain bounded; invalid responses use deterministic fallback. Published standard rates and observed live-test costs are recorded in the paper handoff.

## Documentation

- [Human/expert review invitation, instructions, rubric and coordinator checklist](docs/EXPERT_REVIEW_PACKET.md)
- [Hybrid engine setup, architecture and teacher demonstration](docs/HYBRID_ENGINE.md)
- [Hybrid implementation tasks and scope](docs/HYBRID_UPGRADE_PLAN.md)
- [Tracked T0–T11 release checklist](docs/HYBRID_TASK_CHECKLIST.md)
- [Hybrid paper methods and measured results draft](docs/HYBRID_PAPER_DRAFT.md)
- [Hybrid recognition benchmark protocol](docs/HYBRID_RESEARCH_PROTOCOL.md)
- [Bounded FDA biological dataset and provenance](docs/BIOLOGICAL_DATA.md)
- [Blank annotation, explanation and usability response sheets](docs/review_templates/README.md)
- [Typed evidence graph, queries, validation and paper scope](docs/EVIDENCE_GRAPH.md)
- [Detailed before/after implementation brief for teammates](docs/TEAM_IMPLEMENTATION_BRIEF.md)
- [Paper team handoff and archived synthetic results](docs/PAPER_HANDOFF.md)
- [Observed verification results and preliminary research findings](docs/VERIFICATION_RESULTS.md)
- [Manual acceptance and expert-validation steps](docs/MANUAL_VALIDATION.md)
- [Implementation status against the full roadmap](docs/IMPLEMENTATION_STATUS.md)
- [Study protocol, dataset format, and paper artifacts](docs/RESEARCH_PROTOCOL.md)
- [Hybrid architecture diagram](docs/hybrid_architecture.mmd)
- [Earlier source-screening/review architecture](docs/review_architecture.mmd)
- [Historical engineering/research audit](docs/ENGINEERING_RESEARCH_REVIEW.md)

## Source preparation and container

`python scripts/preprocess_ddinter.py` rebuilds the source artifacts and EDA, checks ambiguous names and invalid rows, and assigns a content-derived snapshot version. Restart the application after updating snapshots. Confirm dataset terms and cite the relevant DDInter primary paper before redistributing data. Existing dates describe local snapshots, not a verified upstream freshness guarantee.

```powershell
docker build -f .Dockerfile -t medication-dossier .
docker run --rm -p 8501:8501 medication-dossier
```

The default container includes OCR; `--build-arg WITH_OCR=0` builds the lighter profile. `--build-arg WITH_ML=1` installs optional model dependencies; mount an existing deployment bundle under `/app/artifacts/ml/deployment` to use it. Docker Desktop must be running. The frozen original engine in `legacy/frozen_v1/` is only a research baseline. Other archived modules are not active application components.
