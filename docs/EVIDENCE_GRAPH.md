# Typed evidence graph and paper scope

Implemented on 2026-10-04. This is a local, typed provenance knowledge graph for each review dossier. It connects observed mentions, candidate interpretations, resolved ingredient identities, source findings, evidence, and review decisions. It uses the existing NetworkX dependency and works without an API key or a graph database.

The graph is implemented in `adr_system/evidence_graph.py`. It is used for provenance queries, candidate-impact inspection, correction/evidence comparisons, report metadata and paths, standalone exports, and retrieving the allowed evidence for optional explanations. The ingredient-pair screening and impact queue retain their existing algorithms. No PrimeKG, DRKG, Hetionet, graph neural network, or learned link predictor has been imported.

## Entity and relationship model

| Entity | Examples of its relationships | Meaning |
|---|---|---|
| Dossier | has mention, has finding, has review, has coverage | Groups one assessment |
| Medication mention | resolved to ingredient, has candidate, has priority | Preserves the distinction between reading, identity and review |
| Candidate interpretation | contains ingredient | One interpretation can contain multiple ingredients; alternatives remain separate |
| Ingredient / food | participates in finding | Links known identities to detected source findings |
| Detected finding | uses record, supported by passage, identity context | A finding produced by source screening |
| Source record | record involves ingredient, has evidence | A versioned knowledge-base relationship, possibly reachable in a candidate scenario |
| Evidence passage | from document | Carries purpose, text/hash, evidence ID, section and retrieval time |
| Source document | source, identifier, version, URL properties | Explicitly records whether a specific document identifier is available |
| Review decision | changes mention, selected/previous ingredient, introduced/removed/retained historical finding | Audits a correction without rewriting the observation |
| Review priority / candidate preview | has preview, assumes candidate/partner, could retrieve source record | Hypothetical possibilities, distinct from detected findings |
| Coverage assessment | assesses mention, ingredient or food | Unknown, no-record, unassessed and consumption states |
| Source disagreement | linked from finding/source record | Exposes disagreement without changing the selected source severity |

Relationships are directed and checked against allowed source/target node types. Stable node/edge IDs are derived from structured identifiers. Graph serialization is deterministic for a given dossier and includes a SHA-256 content checksum. Passage hashes are checked when constructing the graph.

## Queries now available

- `trace_finding(record_id)` returns typed paths from a resolved mention through its ingredient and detected finding to a supporting passage and source document.
- `evidence_for(record_id)` retrieves excerpts connected to detected findings. Identity-only information is excluded by default. `explanation_payload()` now obtains its allowed excerpts through this query.
- `candidate_impacts(mention_id)` shows the candidate/partner interpretations, source states and records that could be retrieved. It marks whether the preview enumeration is complete within the configured bounds.
- `unresolved_for_record(record_id)` identifies unresolved mentions represented in candidate previews for that record; consult graph coverage metadata when preview enumeration is bounded.
- `review_changes(mention_id)` returns the introduced, removed and retained source-record IDs for each correction.
- `compare_graphs(before, after)` compares actual findings, severity, source versions, excerpt hashes and source-document versions. Hypothetical records are excluded from this comparison.
- `neighborhood()` supplies a bounded graph view for an entity. Direction is preserved in the underlying graph and relationship tables; display neighborhood selection considers both incoming and outgoing links.

The graph does not infer patient harm from a path. A structurally complete path documents provenance; it does not establish clinical relevance or authoritative citation quality. PubChem identity information cannot enter an interaction support path. A candidate can reach a possible source record without creating a detected finding or an LLM claim.

## Bounds and privacy

The explorer includes up to 120 candidate previews by default, with at most 32 ingredient-pair checks per preview. Truncation is visible in metadata and the UI. Detected findings and their support are not removed by these preview bounds. The display is capped at 100 entities around the selected focus; choosing another focus can reveal another neighborhood. The original review queue is not truncated or changed by the graph preview limit.

Original medication readings and dietary strings are omitted from graph exports by default. Normalized ingredients and findings remain health information. The optional graph-download checkbox includes observed text explicitly. A graph contains no API credentials or prescription image pixels. HTML reports include provenance paths; dossier JSON includes a graph summary/checksum. The full typed graph has a separate JSON download, with graph schema 1.0. Dossier schema is now 2.1. A standalone graph file is not a replayable dossier.

## Run and manually inspect

```powershell
.\.venv\Scripts\python.exe -m streamlit run llm.py
```

1. Restart the app after this code update. Choose **02 / Ambiguous medicine reading** and build a dossier.
2. Open the sidebar **Interaction map** workspace, then its **Evidence graph** tab.
3. Inspect entry `m1` under **Identity review and changes**. Warfarin is a candidate interpretation and its aspirin pair is a possible source record, not a detected finding.
4. Return to **Review desk**. The entered text must remain present. Confirm warfarin for `m1`.
5. Return to **Evidence graph**, select the detected aspirin/warfarin record, and inspect the provenance table. Its paths should identify entries, ingredients, evidence IDs and source versions.
6. Check the latest-correction comparison and review history: the pair should be introduced by this decision.
7. Optionally refresh supporting evidence in **Review desk**. New or changed passages/document versions should appear in the graph comparison; failed/unchanged retrieval need not create a difference.
8. Download the graph with original text disabled. With a synthetic private header in the input, confirm that the header is absent. Include it only through the explicit original-text option.
9. Edit medication input without rebuilding. The graph workspace must block exploration/export until the new dossier is built.
10. Inspect desktop/mobile layout, keyboard controls, graph tooltips, relationship tables and HTML print output on your devices.

## Executed verification

The expanded test suite passes 53 tests, including typed provenance, alias identity, hypothesis separation, identity-only exclusion, redaction, review additions/removals, source-version changes, graph determinism, preview bounds, and the graph UI/navigation workflow. All 11 exact source regression fixtures still pass.

```powershell
.\.venv\Scripts\python.exe scripts\graph_benchmark.py
```

The graph conformance experiment uses 11 existing software fixtures plus three stages of a synthetic review example: 14 scenarios. It observed 7 detected findings across the scenarios, all structurally traceable, with 17 support paths. Graph-retrieved and flat-record evidence sets were equal. These are software structure checks, not 14 independent prescriptions or a clinical accuracy experiment. The same finding can occur in more than one scenario.

The new [graph checkpoint](research_artifacts/graph_checkpoint_2026-10-04/summary.json) includes its execution manifest, [case metrics](research_artifacts/graph_checkpoint_2026-10-04/case_metrics.csv), [support paths](research_artifacts/graph_checkpoint_2026-10-04/support_paths.json), [review transitions](research_artifacts/graph_checkpoint_2026-10-04/review_changes.json), and before/after graph JSON. No external calls were needed. The previous review/OCR/Groq runs remain earlier experiments with their own original manifests.

The [before/after graph figure](research_artifacts/graph_checkpoint_2026-10-04/graph_review_transition.svg) is generated from actual graph relationships in the synthetic correction example. It is a methods illustration, not an effectiveness plot.

## Paper readiness

There is now a defensible systems-paper draft: a specified task, implemented identity-review method, typed provenance graph, comparisons, ablation, reproducibility package and preliminary experiments. A suitable working title is **Provenance-Aware, Impact-Prioritized Human Review for Prescription-Based Interaction Screening**.

The strongest effectiveness claim still requires independent expert-reviewed prescriptions and an actual comparison at equal review budgets. The graph's addition establishes new traceability/query functionality, not an improvement in interaction accuracy. To claim a reviewer benefit, evaluate evidence-location time, correction success or audit completeness with real reviewers; counterbalance cases and record actual time. To claim explanation quality, complete the blinded expert ratings. Review novelty against prior work.

Real-photo OCR, the separate handwriting challenge, independent source interpretation, manual usability/print checks, remote CI and container runtime verification remain pending. A broad claim of clinical validation, patient-specific risk prediction, hallucination-free output or guaranteed conference acceptance is unsupported by the current studies.
