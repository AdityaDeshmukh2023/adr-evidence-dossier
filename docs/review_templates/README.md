# Blank human-review response sheets

Use these with [the expert review packet](../EXPERT_REVIEW_PACKET.md). They contain headers only. No expert labels, permissions, timings, or ratings have been filled.

Copy sheets into separate private reviewer folders, for example:

~~~powershell
New-Item -ItemType Directory -Path data/private/expert_review/R1
Copy-Item -Path docs/review_templates/*.csv -Destination data/private/expert_review/R1
~~~

Use a new folder per reviewer/round; preserve returned originals. Completed material belongs under ignored `data/private/`, not in this tracked template directory. Keep reviewer contact details in a separate controlled record.

## Shared conventions

- Save CSV as UTF-8 with the original headers. Excel can edit these files.
- Use one row per unit described below. Quote cells containing commas, line breaks, or quotes using standard CSV export.
- Use `|` between multiple ingredient names/IDs or checklist IDs in one cell. Do not use that separator inside a single name.
- Use ISO dates/times including time-zone offset where relevant, for example `2026-10-04T14:00:00+05:30`.
- Use `true`/`false` for established booleans. Leave pending answers blank; do not interpret a blank as zero, false, or absence.
- Keep case IDs, entry IDs, pair IDs, output IDs, and reviewer codes stable across sheets.
- Record actual source IDs/versions and locations, not a vague statement that something was checked.
- These sheets are working records. The coordinator converts them to the evaluator JSON; the app does not import these CSVs automatically.

## Sheet instructions

| File | Row unit and fields |
|---|---|
| `case_register.csv` | One case per reviewer. Link case to real `group_id`, declared split, material origin, kind and private image path; record `pending`, `reviewed`, or `excluded` status. |
| `medication_annotations.csv` | One entry per reviewer. Preserve observed text, ingredient names and catalog IDs where coordinator-verified. `identity_status`: verified / non_medication / unreadable / identity_uncertain / out_of_catalog. Leave is-medication blank if itself uncertain. |
| `pair_annotations.csv` | One unordered distinct-ingredient pair per reviewer. Include no-record and not-assessable judgments, not only positives. `reference_status`: present / no_record_in_reference / not_assessable. Mapped severity: high / moderate / low / unknown for positive records; blank otherwise. `clinical_judgment`: agrees / disagrees / uncertain / not_assessed. |
| `claim_ledger.csv` | One factual claim per reviewer/output. Judgment: supported / partial / unsupported / contradicted / unevaluable. Only eligible claims enter rating totals; give the exclusion reason for noneligible procedural text. |
| `citation_ledger.csv` | One claim-to-source association per reviewer/output. `correct`: true / false, or blank while pending. Every citation counted in totals must have a row. |
| `expected_claims.csv` | One expected item per case, prepared by the coordinator before rating methods. Give a stable checklist ID, reference, and rationale for requiring it. |
| `explanation_ratings.csv` | One output/reviewer pair with numerical totals. Exact scorer headers; blank template has no assigned IDs. Prefer copying the checkpoint's ratings file, which already lists actual output IDs. |
| `workflow_tasks.csv` | One attempted task per participant/case/condition. Record order and real times. `completed` true/false; timeout/abandonment stays in outcomes. Durations/errors are actual observations, not estimates. |
| `domain_audit.csv` | One rule/mapping/passage/wording issue per reviewer. Verdict: accept / revise / reject / needs_reference. `item_type` can be alias, combination, food_rule, management_text, evidence, or source_disagreement. |
| `adjudication_log.csv` | One disagreement. `unit_type`: entry / pair / claim / citation / omission / domain. Preserve both original values, adjudicator's decision, rationale/source, and resolved/unresolved status. |
| `exclusion_log.csv` | One case/output/image exclusion or coverage issue. Record stage, reason, attempted resolution, decision, and impact on denominators. Never use this to silently drop an unfavorable result. |

## Before scoring explanations

The totals sheet must have:

~~~csv
output_id,supported_claims,total_claims,correct_citations,total_citations,omissions,reviewer_id,notes
~~~

Check the completed totals against both ledgers and the frozen expected-claims list. No negative or fractional counts; supported/correct must be at most total. No duplicate output/reviewer rows. Account for missing assigned outputs. Keep any unresolved output out of the scored copy with an explicit exclusion record.

`scripts/score_explanations.py` needs the unblinded `outputs.json` in the same folder as the completed scoring CSV. Only the coordinator should hold that file during blinded review. Multiple independent reviewer rows may be analyzed, but report their dependence; they are not extra independent cases.

For the text/image evaluator formats and actual commands, use [the coordinator instructions](../EXPERT_REVIEW_PACKET.md#10-coordinator-convert-and-run-the-existing-evaluation-tools).
