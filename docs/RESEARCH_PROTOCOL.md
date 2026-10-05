# Research protocol and artifact contract

Working title: **Prioritizing Human Review in Prescription-Based Drug Interaction Screening Through Medication-Identity Uncertainty**.

Hypothesis: prioritizing unresolved medication readings by their possible effect on source findings recovers more high-severity source records per review action than ordering by recognition/matching uncertainty. This is a testable hypothesis, not a claim of global novelty or patient benefit.

For recruitment, independent annotation instructions, explanation-rating definitions, response sheets, and the coordinator workflow, use [the human and expert review packet](EXPERT_REVIEW_PACKET.md). It describes proposed review procedures; blank sheets are not collected evidence.

## Implemented comparisons

- `legacy_baseline`: frozen original committed engine, including its known defects.
- `corrected_baseline`: current deterministic screening without review.
- `prescription_order`: original entry order, with no-candidate entries handled first.
- `uncertainty`: required review first, then descending matching/OCR ambiguity.
- `impact`: required review first, then high-severity changes, other changes, coverage changes, ambiguity, and original order.
- `impact_without_severity`: combines high/other changes without severity precedence.
- `gold_upper_bound`: all identities corrected; explicitly an upper bound, not performance achievable at zero review effort.

All methods share the same terminology, source snapshot, candidate generator, and input. Gold identities are revealed only after the simulator selects the next item for review. The simulator can correct identities missing from the candidate list. Budgets are 0%, 25%, 50%, 75%, and 100% of entered items, rounded upward. Report actual review counts alongside fractions, especially for short lists. The queue is recomputed after each correction.

Candidate scores use string similarity, not clinical probabilities. No interaction outcome influences candidate generation. A candidate's possible findings are compared with unresolved status as well as competing candidate interpretations. Required manual-review items form a common first-priority category across ordering methods. This policy is documented so it cannot silently favor the proposed method.

## What the generated benchmark establishes

```powershell
python scripts/evaluate.py
python scripts/benchmark.py
python scripts/ocr_benchmark.py
python scripts/explanation_benchmark.py
```

The default review benchmark generates 40 source-derived prescription groups with clean and two corrupted-text variants (120 cases total). Eight groups are development and 32 groups are test (96 evaluated cases). Medication-entry boundaries are supplied. These results assess entry resolution and simulated review, not complete document segmentation or clinical accuracy.

The separate image benchmark executes OCR on eight generated printed images from two base groups. Report it as an integration/controlled-robustness experiment. Do not treat eight variants as eight independent prescriptions or claim real-world OCR performance from these results.

The default explanation experiment produces deterministic outputs and blank reviewer sheets. `--live` adds prompt-only and validated extractive Groq outputs. Prompt-only output exists only as a research baseline. It is never used as app guidance. Human ratings are required before unsupported-claim or citation-correctness metrics exist.

## Expert dataset format

```json
{
  "version": "expert-pilot-v1",
  "kind": "expert_reviewed",
  "permission_confirmed": true,
  "label_provenance": "Describe reviewers and actual source/reference process",
  "scope": "English printed prescriptions; annotated medication-entry boundaries",
  "cases": [{
    "id": "CASE-001",
    "group_id": "ORIGIN-001",
    "split": "test",
    "input_kind": "real_printed_text",
    "annotation_status": "reviewed",
    "reviewer_ids": ["R1"],
    "entries": [
      {"text": "Warfarln 5 mg", "gold_names": ["warfarin"], "is_medication": true},
      {"text": "Aspirin 75 mg", "gold_names": ["acetylsalicylic acid"], "is_medication": true}
    ],
    "gold_findings": [{
      "ingredient_ids": ["DDInter1951", "DDInter20"],
      "severity": "high",
      "source_reference": "Record the actual source document/record/version reviewed"
    }]
  }]
}
```

This is a format illustration, not a completed expert annotation. A non-medication entry uses an empty `gold_names` list and `is_medication: false`. A combination product lists all independently verified ingredients. IDs must correspond to the frozen terminology. Record an empty finding list only after reviewing the case; an absent database edge is not a confirmed clinically safe pair.

Create a workbook using `python scripts/prepare_annotations.py --count 150`. Keep it under ignored `data/private/`. Allocate approximately 20% to development and 80% to test, grouped by original prescription/template/patient or writer where applicable. Related variants cannot cross splits. The evaluator rejects group leakage, identical-input leakage, incomplete expert annotation, and missing permission declarations.

```powershell
python scripts/benchmark.py --dataset data/private/annotations.json --split test --output artifacts/expert-study
```

Review the pilot to determine whether the final sample supports the desired precision/effect estimate. The 100–200 range is a feasibility target, not a power calculation. Report recruitment/source selection and limits of the convenience sample.

## Real image dataset format

Use a JSON object with `kind: "expert_reviewed"`, `permission_confirmed: true`, and `image_cases`. Each image case requires `id`, `group_id`, `kind` (`printed` or `handwriting`), `image` (local path), `gold_text`, `gold_names`, `gold_findings`, and `annotation_status: "reviewed"`. Relative image paths resolve against the dataset file. Use the same ingredient-ID/severity finding format above.

```powershell
python scripts/ocr_benchmark.py --dataset data/private/image_annotations.json --output artifacts/expert-images
```

CER/WER use normalized whitespace and case; report this preprocessing. Image failures remain in the denominator. Character/word edits may exceed reference length. Image-derived ingredient and finding recall are measured before manual correction. Real-image OCR text is not written into the default output files. Keep printed and handwriting strata separate in reporting.

## Metrics and analysis

Primary endpoint: difference in high-severity source-record recall between `impact` and `uncertainty` at 50% review budget, evaluated on noisy inputs. Report numerator, denominator, review count, and a paired 95% interval. The implementation uses 1,000 prescription-group bootstrap samples with seed 17. Groups with no high-severity reference still count toward general resolution/coverage analysis but cannot define high-severity recall on their own.

Secondary endpoints: exact pair/severity precision and recall, identity accuracy, candidate recall@5, unresolved entries, severity confusion, median/p95 runtime, and explanation ratings. False-positive labels in these tables mean disagreement with the reference, not proof that a combination is clinically non-interacting.

Do not tune thresholds on test results. Freeze the 0.60 lexical-candidate cutoff, five-candidate limit, and 0.90 OCR review threshold for the first study; alternative thresholds require development-only experiments followed by a fresh locked evaluation. These thresholds are engineering defaults, not calibrated probabilities.

## Files for the writing team

| File | Use |
|---|---|
| `dataset.json`, `summary.json` | Dataset version/hash, split, scope, code and package provenance |
| `case_results.jsonl`, `case_metrics.csv` | Case-level predictions, actions, metrics, timing |
| `comparison.csv`, `ablation_at_half_budget.csv` | Method and ablation tables |
| `identity_summary.csv`, `extraction_metrics.csv` | Identity accuracy and candidate recall |
| `severity_confusion.csv`, `failure_categories.csv` | Error analysis; categories can overlap |
| `review_budget.png/.svg` | Primary recovery-versus-review figure |
| `identity_resolution.png`, `failure_categories.png` | Supporting figures |
| `latency_cost.csv` | Local computation timing and zero external-request cost for this experiment |
| Image benchmark `image_metrics.csv`, `summary.json` | Actual OCR results; report dataset origin |
| Explanation `blinded_review.json`, `ratings.csv` | Expert evaluation material |

Timing depends on hardware, warm caches, and model initialization. Preserve these conditions in methods. No API pricing is guessed: live explanation outputs record token usage, and actual costs must be computed from the provider/model price at the experiment date.

For a real reviewer-time study, recruit reviewers, define the protocol and permissions, counterbalance case assignments, record measured time/corrections, and analyze reviewer and prescription clustering. Simulated review counts cannot replace this study.

## Positioning and claims

The implemented typed evidence graph has a separate `scripts/graph_benchmark.py` conformance check. It compares graph-retrieved support with the original flat evidence sets, checks provenance paths, and records correction changes. The 14 software scenarios are not independent patient cases. Equivalence preserves screening outputs and does not demonstrate an accuracy gain. For a human graph-versus-table comparison, counterbalance interfaces/cases and measure actual evidence-location time, correction success and audit completeness. See [the graph methods and scope](EVIDENCE_GRAPH.md).

Compare with prescription digitization/catalog matching, medication-normalization review tools, DDI explanation systems, and evidence-grounded drug QA. Start with [Automated Digitization of Unstructured Medical Prescriptions](https://aclanthology.org/2023.acl-industry.76/), [ExDDI](https://ojs.aaai.org/index.php/AAAI/article/view/34709), and [DrugRAG](https://arxiv.org/abs/2512.14896). Follow relevant [TRIPOD-LLM](https://www.nature.com/articles/s41591-024-03425-5) items for any LLM study.

Allowed now: implemented capabilities and accurately qualified software/synthetic results. Pending evidence: independent clinical correctness, expert explanation support, reviewer-time benefit, generalization to handwriting/other languages, and a novelty claim supported by a comprehensive related-work comparison.
