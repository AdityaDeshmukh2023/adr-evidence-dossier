# ADR Evidence Dossier

An evidence-first Streamlit research prototype for known drug–drug and curated drug–food interaction alerts. It is an educational decision-support demonstration, **not a medical device or prescribing tool**.

## What it does

- Normalizes a small reviewable set of medicine names and brand aliases.
- Detects deterministic interactions from versioned local snapshots.
- Displays severity, explanation, management wording, evidence links, data version, and a network map.
- Optionally retrieves openFDA/PubChem supporting evidence; live sources never create alerts.
- Produces privacy-safe HTML evidence reports and includes synthetic showcase cases.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run llm.py
```

Copy `.env.example` to `.env` and add `GROQ_API_KEY` only for bounded LLM explanations. Without a key, the application uses a deterministic source-linked explanation.

## Reproducibility and evaluation

Build the full DDInter processed snapshot and EDA assets from the downloaded raw
CSV files with:

```powershell
python scripts/preprocess_ddinter.py
python scripts/evaluate.py
pytest -q
```

This creates `data/processed_ddinter/` with a deduplicated CSV and Parquet dataset,
app lookup index, metadata, quality report, EDA tables, charts, and a paper-ready
EDA summary. The app automatically prefers this full snapshot when it exists.

See `data/manifest.json` and `data/processed_ddinter/metadata.json` for sources and
snapshot descriptions. Before publication, verify DDInter's terms and cite its
primary database paper; have every clinical statement independently reviewed.

## Important limitations

The prototype has incomplete data coverage and does not assess dose, full medical history, laboratory values, kidney/liver function, pregnancy, or patient-specific contraindications. “No known interaction found” is never a safety guarantee. Do not start, stop, or change treatment based on this system.

## Showcase flow

Open **Case Library**, select a synthetic case on **Analyze**, review evidence, then open **Interaction Map** and download the HTML report. Use only synthetic examples in slides or paper screenshots.
