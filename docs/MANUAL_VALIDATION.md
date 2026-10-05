# Manual acceptance and expert validation

The automated suite covers core behavior, review simulation, report replay, API failure handling, and the Streamlit correction workflow. The remaining checks below require a person, credentials, permitted data, or Docker Desktop. Record actual outcomes; do not mark a step passed without performing it.

## 1. Start and inspect the application

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\doctor.py
.\.venv\Scripts\python.exe -m streamlit run llm.py
```

Open the local URL printed by Streamlit. Use the Review desk. Check readability at desktop width and at approximately 390 px width. Use Tab, Shift+Tab, Enter, and Space to navigate controls. Confirm that keyboard focus is visible and status labels remain understandable without relying on color.

## 2. Confirm the principal user journey

1. Select **02 / Ambiguous medicine reading** and click **Build review dossier**.
2. Check that `Warfarln` and `Metformln` are unresolved; no guessed identity should appear as confirmed.
3. Inspect the review reason and the candidate-dependent preview. These must be visibly distinct from resolved alerts.
4. Select the warfarin candidate for `Warfarln`, then click **Confirm review decision**.
5. Confirm that the aspirin/warfarin source finding appears and the review history records the change.
6. Resolve the remaining entry. Confirm that the completeness status changes.
7. Edit the original input without rebuilding. The application must request a new dossier and suppress review/export of the changed input until rebuilt.
8. Open **Interaction map** and select an edge record. Confirm it corresponds to a finding and its evidence record.

The ordering reflects the configured snapshot, not a clinical priority judgment about an individual patient.

## 3. Check images and corrections

```powershell
.\.venv\Scripts\python.exe scripts\ocr_smoke.py
.\.venv\Scripts\python.exe scripts\ocr_benchmark.py
```

Upload `artifacts/ocr_smoke/synthetic_prescription.png`. Click **Read prescription image**, then **Build review dossier**. Select an entry and check that its highlighted image region matches the text. Repeat with a permitted, de-identified photograph of a printed prescription. Review every extracted medicine against the source, including missing text.

For a wrong candidate, choose **Enter ingredients manually** and enter recognized generic ingredient names separated by commas. For a header or other non-medication text, use **Exclude non-medication text**. Do not exclude an unreadable medicine merely to make completeness look better.

Test a rotated photo, glare, low contrast, a blank image, and a file over 10 MB. Expect clear errors or unresolved entries, never a claim of safety. Handwriting results belong to a separate challenge set.

Initial OCR use downloads public model files. Images are processed locally. Model cache access can be blocked by an IDE sandbox even when the same command works in an ordinary terminal; run the command from your terminal in that situation.

## 4. Check the previously identified edge cases

The new graph explorer has a separate [manual journey and query guide](EVIDENCE_GRAPH.md#run-and-manually-inspect). Check candidate-dependent records before correction, detected provenance paths afterward, input persistence when navigating, redacted graph downloads, and the changed-input guard.

| Input | Expected behavior |
|---|---|
| `Warfarin 5 mg Aspirin 75 mg` | Two medicines retained and a high source finding |
| `Tab. Warfarin 5 mg` then aspirin | Prefix handled; warfarin preserved |
| Aspirin then apixaban | The full source record is accessible through the aspirin alias |
| Acetaminophen then aspirin | Unknown-severity source record, with an explicit uncertainty warning |
| Simvastatin + `no grapefruit` | Negated dietary entry; no consumed-food alert |
| Warfarin + `kaleidoscope` | No substring food match |
| Simvastatin + repeated grapefruit phrases | One alert for the same rule |
| An unidentified medicine | Incomplete assessment and unassessed pairs |
| Empty input | No assessment performed |

These are software/source-conformance expectations, not a clinical validation reference.

## 5. Check downloads and privacy controls

1. Enter a synthetic header such as `Patient TEST_ONLY` alongside medicines.
2. Download HTML and JSON with **Include original entered text** unchecked.
3. Confirm `TEST_ONLY` is absent, while the unresolved entry ID and incomplete status remain visible.
4. Enable the checkbox and download again. Confirm original text appears only in this explicitly requested export.
5. In **Research workspace**, upload the JSON and click **Verify and replay**. Same-version findings should reproduce without external requests.
6. Alter the JSON manually and try again. Integrity verification should reject the change.
7. Click **Clear session**. The image, input, and review history must disappear.

Normalized medication information is still sensitive. Default redaction removes original text, not all health information. Report integrity checks detect accidental changes; they are not digital signatures.

## 6. Optional live evidence and LLM checks

Use synthetic examples only for this check. Click **Fetch / refresh supporting evidence** and inspect exact label document IDs, versions, excerpts, and source status. PubChem identity data must remain labeled as identity information. Failure or no relevant passage must leave local findings unchanged.

For Groq, put `GROQ_API_KEY` in a local `.env` and restart the app. Enable **Allow Groq to select source excerpts from normalized findings**. Confirm the mode reports validated output or an explicit fallback. Never share the key in screenshots or the paper.

```powershell
.\.venv\Scripts\python.exe scripts\explanation_benchmark.py --live --max-cases 4
```

This makes bounded API requests and uses the account's quota; on the configured Groq Free account it is not a paid-rate billing estimate. The existing archived pilot can be reviewed without any new request. Follow [the expert review packet](EXPERT_REVIEW_PACKET.md#6-task-b-blinded-explanation-assessment) to distribute blinded material and record genuine ratings. Review `blinded_review.json` and fill `ratings.csv` in `artifacts/explanations/`. Score completed ratings with:

```powershell
.\.venv\Scripts\python.exe scripts\score_explanations.py artifacts\explanations\ratings.csv
```

A domain reviewer must decide whether passages actually support claims. Formatting checks cannot establish clinical faithfulness. An invalid or absent key should produce the deterministic fallback in the app.

## 7. Expert work needed for the paper

Use [the detailed expert review packet](EXPERT_REVIEW_PACKET.md) for the invitation, task instructions, independent-review rules, and [blank response sheets](review_templates/README.md).

1. Confirm permission for the 100–200 examples and remove patient identifiers before entering them into the study workspace.
2. Use the existing pending `data/private/annotations.json` workbook. Only if it does not exist, create one with `python scripts/prepare_annotations.py --count 150`; preserve existing annotations.
3. Follow `RESEARCH_PROTOCOL.md` to label medicine identities, explicit source-supported pairs, severity, and source references.
4. Review ambiguous names, combination ingredients, food rules, and curated explanation text. Do not add brand combinations without evidence.
5. Have a second reviewer independently label a subset if available; document disagreements and adjudication. If unavailable, disclose the single-reviewer limitation.
6. Freeze the test partition before tuning and give the writing team the versioned output directory and its manifest.

The assistant cannot supply genuine expert judgments, permission, clinical outcomes, or a human usability study on your behalf.

## 8. Container validation

Docker Desktop must be running with Linux containers. It was not running during the local implementation checks.

```powershell
docker build -f .Dockerfile -t medication-dossier .
docker run --rm -p 8501:8501 medication-dossier
```

Verify the synthetic typed-input workflow. To omit optional OCR dependencies during a lightweight build, add `--build-arg WITH_OCR=0` to the build command. The default build includes OCR. Test model downloads/cache persistence before an offline presentation.

## Acceptance record

For each manual check record: date, software revision, data hashes, OS/browser, case ID, expected result, observed result, pass/fail, and a synthetic or de-identified screenshot. Do not put real prescription screenshots in the public repository.
