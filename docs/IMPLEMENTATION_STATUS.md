# Implementation scope and remaining validation

The current release scope is the [hybrid T0–T11 checklist](HYBRID_TASK_CHECKLIST.md), with the [hybrid engine guide](HYBRID_ENGINE.md) and [measured paper draft](HYBRID_PAPER_DRAFT.md). Its automated research experiments do not require a human-review study. The table below retains the earlier review-oriented roadmap and its pending external validation; those studies are not prerequisites for this hybrid release.

| Original workstream | Implemented deliverable | Remaining human/external requirement |
|---|---|---|
| Stable identity and aliases | Ingredient-ID catalog, complete aspirin source reachability, exact/approximate separation | Review new regional brand mappings before adding them |
| Preserve inputs and attributes | Multiple-medicine extraction, dose/route/frequency fields, unresolved spans | Evaluate on permitted real prescription layouts |
| OCR | PaddleOCR 3.x result adapter, geometry/scores, cached CPU models, image smoke/benchmark scripts | Real printed-image benchmark and separate handwriting challenge set |
| Core screening | ID pair lookup, duplicate ingredients, food boundaries/negation, unknown/unassessed states | Domain review of clinical interpretation and food rules |
| Human review | Candidate selection, manual ingredient correction, non-medication exclusion, change history | Observe real reviewers using the workflow |
| Research contribution | Candidate-impact queue, required-review category, severity ablation | Independent benchmark and comprehensive novelty comparison |
| Evidence | Specific live label passages/IDs/versions, identity-only PubChem data, TTL/retry/refresh | Expert assessment of relevance; static demo citations remain illustrative |
| Typed evidence graph | Typed mentions/candidates/ingredients/findings/passages/documents/review nodes, provenance and hypothetical queries, change comparisons, graph-backed evidence retrieval, explorer and redacted export | Human traceability/time study; external biomedical graphs remain outside the implemented graph |
| Explanations | Deterministic default, low-cost GPT-OSS 20B, strict schema, allowlisted extractive output, source checks and fallback; 4/4 live selections passed | Expert claim-support ratings and key rotation |
| Reports/replay | Redacted default HTML/JSON, full-text opt-in, hashes, same-version offline replay | Manual browser/print inspection |
| Interface | Four-step review desk, highlighted source regions, graph record inspection, research workspace, session clearing | Visual/accessibility checks on your devices |
| Reproducibility | Windows tested lock, Linux resolved locks, content-derived preprocessing versions, code/data/model manifests | Linux runtime verification via CI/container |
| Engineering checks | Unit/regression/Streamlit/research tests, exact fixture evaluator, CI, input limits, sanitized diagnostics | CI runs after pushing; Docker Desktop must be started for local build |
| Research output | Frozen legacy baseline, corrected baseline, review methods, group bootstrap, CSV/JSON/PNG/SVG artifacts | Complete expert annotations; simulated counts do not establish reviewer-time benefits |
| Manuscript handoff | Research protocol, annotation workbook generator, detailed expert-review packet with invitation/rubric/coordinator instructions and blank response sheets, metric artifact descriptions, tracked synthetic tables/figures and paper handoff | Teammates write the manuscript and verify sources; recruit reviewers and collect/adjudicate actual judgments |

Synthetic cases and controlled printed images are deliberately labeled. They are not replacements for expert-reviewed examples. Claims of clinical accuracy, patient benefit, real handwriting robustness, or unprecedented novelty remain unsupported until those studies are completed.

The original `ENGINEERING_RESEARCH_REVIEW.md` is a historical pre-implementation audit. Use automated results and this implementation status to assess the upgraded code, rather than treating every historical finding as an unfixed defect.
