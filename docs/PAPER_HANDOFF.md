# Paper team handoff

Checkpoint: 2026-10-04. The engineering and research tooling are ready for a shared development checkpoint. The manuscript can progress alongside the remaining real-data and expert studies.

For the subsequent automatic hybrid engine release, use [the hybrid methods/results draft](HYBRID_PAPER_DRAFT.md), [release checklist](HYBRID_TASK_CHECKLIST.md), and [setup/teacher demonstration](HYBRID_ENGINE.md). The human-study material in this earlier handoff remains pending and is not required for the new automated experiments.

To recruit reviewers and collect that evidence, share [the human and expert review packet](EXPERT_REVIEW_PACKET.md) and [blank response sheets](review_templates/README.md). The packet covers independent reference labels, blinded explanations, traceability/usability tasks, disagreement handling, and actual evaluator commands. It contains no completed expert results.

The subsequent [typed evidence graph](EVIDENCE_GRAPH.md) implementation has its own [graph conformance checkpoint](research_artifacts/graph_checkpoint_2026-10-04/summary.json). It adds provenance, hypothesis and review queries, graph-backed explanation retrieval, and redacted graph exports. The latest suite passes 53 tests. The earlier review/OCR/Groq runs remain valid as earlier runs; their manifests have not been relabeled as executions of the new graph code.

## Start writing now

Use [the protocol](RESEARCH_PROTOCOL.md), [architecture](review_architecture.mmd), [implementation status](IMPLEMENTATION_STATUS.md), and [observed verification](VERIFICATION_RESULTS.md). Treat the original engineering review as a historical audit.

| Paper section | Available material | Work still needed |
|---|---|---|
| Problem and system architecture | Medicine extraction, stable ingredient identities, review, source screening, provenance, exports | Define the target user and intended study setting |
| Proposed method | Candidate generation independent of interaction results; candidate-impact review order recomputed after corrections | Explain ordering and compare with prior work |
| Experimental design | Frozen original baseline, corrected baseline, review-order comparisons, severity ablation, grouped split and bootstrap | Freeze the permitted expert dataset before tuning |
| Preliminary results | Synthetic review tables/plots, controlled image OCR, source-conformance fixtures | Report synthetic scope and group counts explicitly |
| Explanation experiment | Live model outputs, token/cost metadata, blinded sheets | Expert claim-support and citation ratings |
| Limitations | Source coverage, uncalibrated lexical/OCR scores, provided entry boundaries, simulation, food-rule scope | Real photographs, handwriting challenge, human review time and usability |

The potential contribution is prioritizing medicine-identity review by possible downstream interaction changes. The project implements and tests that hypothesis. A claim that nobody has done this before still requires a comprehensive related-work comparison.

## Shared experiment artifacts

The [archived checkpoint](research_artifacts/checkpoint_2026-10-04/) is suitable for Git: it contains synthetic examples, tables, plots, experiment manifests, and blank explanation-rating sheets. It excludes user prescription images, API credentials, private annotation workbooks, and model caches.

Begin with:

- [Review summary](research_artifacts/checkpoint_2026-10-04/research/summary.json) and [comparison table](research_artifacts/checkpoint_2026-10-04/research/comparison.csv).
- [Review-budget figure](research_artifacts/checkpoint_2026-10-04/research/review_budget.svg).
- [Controlled OCR results](research_artifacts/checkpoint_2026-10-04/ocr_benchmark/summary.json).
- [Live explanation status](research_artifacts/checkpoint_2026-10-04/explanations_live/summary.json).
- [Blinded explanation sheet](research_artifacts/checkpoint_2026-10-04/explanations_live/blinded_review.json) and [unfilled ratings](research_artifacts/checkpoint_2026-10-04/explanations_live/ratings.csv).

Every run preserves its own code/data hashes. The review experiment predates the low-cost Groq migration; its core screening/review method is unchanged. Do not relabel its manifest as a later execution. Re-run experiments into a new directory for subsequent submission checkpoints. Blank sheets and private pending annotation entries are not collected results.

To archive another synthetic checkpoint:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_paper_bundle.py --output docs\research_artifacts\next_checkpoint --live-directory artifacts\explanations-live-2026-10-04
```

## Push this as a development checkpoint

Share a branch/commit for the upgraded implementation and include this document and the archived synthetic artifacts. This checkpoint supports paper drafting while expert evaluation and manual acceptance continue. Label it as a research prototype rather than a clinically validated release.

Review the staged file list before pushing. The local `.env`, `artifacts/`, `.venv/`, and `data/private/` are ignored. Existing root prescription photographs are already tracked in this repository; confirm permission and de-identification before sharing them, including their Git history. Do not blindly stage the renamed photographs. Keep permitted study images and annotations in the private study workspace.

No commit or push was performed automatically. When preparing the commit, select the implementation files, tests, requirements/configuration, documentation, and synthetic checkpoint explicitly; inspect `git diff --cached --stat` and run `git diff --cached --check`.

## Groq configuration

The default is `openai/gpt-oss-20b` with low reasoning effort, a 1,600-token completion limit, strict JSON schema, and application-level source/citation checks. No automatic upgrade to an expensive model is configured. Larger dossiers use deterministic fallback when they exceed the bounded request size. External model use still requires the explicit UI checkbox.

Rates checked on 2026-10-04 were $0.075 per million input tokens and $0.30 per million output tokens. The eight live requests for four synthetic cases had an estimated standard token cost of $0.00051315. This estimate excludes retries, account credits and discounts, and is not a billing statement. See [Groq models](https://console.groq.com/docs/models) and [strict output documentation](https://console.groq.com/docs/structured-outputs). Groq lists the former Llama defaults as retired for free/developer accounts in [its deprecations](https://console.groq.com/docs/deprecations).

The supplied key was configured only in ignored local `.env`. Because it was pasted into chat, revoke it and replace `GROQ_API_KEY` locally. Teammates should use their own credentials; the paper does not need any key.

The owner subsequently confirmed a Free-tier account. The reported cost is a paid-rate estimate, not a bill. Current published Free-tier limits for GPT-OSS 20B are 30 requests/minute, 1,000 requests/day, 8,000 tokens/minute and 200,000 tokens/day, with organization-specific exceptions. See [Groq rate limits](https://console.groq.com/docs/rate-limits) and the organization's [Limits dashboard](https://console.groq.com/settings/limits). The app continues with deterministic output when the optional model request is unavailable. See the [detailed teammate brief](TEAM_IMPLEMENTATION_BRIEF.md) for the complete before/after guide.
