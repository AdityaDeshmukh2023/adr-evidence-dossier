# Human and expert review packet

Project: **Medication Review Dossier**  
Packet version: **1.0, 2026-10-04**  
Coordinator: **[name, institution, email]**  
Study/version identifier: **[complete before distribution]**

This packet explains what reviewers should check, how to record their judgments, and how the team will use those judgments in the paper. It is a proposed study procedure, not evidence that participants have been recruited or that review has occurred. Replace bracketed fields before sharing.

## 1. Invitation you can send

**Subject: Request for expert review of a student medication-interaction research project**

Dear [name/title],

We are [team and institution], developing Medication Review Dossier for our final-year project and a proposed conference paper. The prototype reads medication text or printed prescription images, identifies ingredients, screens a fixed interaction-source snapshot, prioritizes uncertain readings for human correction, and links findings to evidence and review history.

We would appreciate an independent review of [choose: medication identities and interaction references / generated explanations / the review and evidence interface]. We want to understand mistakes and limitations, including missing coverage and misleading explanations.

For an initial pilot, we propose [five cases or four explanation outputs] and a [30–45 minute] session, subject to your availability. This is a planning estimate; we will adjust the workload after the pilot. You can accept only one task. No patient-care decisions or treatment recommendations are requested.

We will provide permitted, de-identified material, written instructions, and a response sheet. Your judgments will be recorded under a reviewer code. We will agree on acknowledgement, any compensation, and the handling of professional details before participation. We will seek separate permission before naming you in the paper.

Coordinator: [contact]. Proposed dates: [dates]. Institutional study/permission status: [actual determination or pending]. Material delivery and secure return method: [method].

Would you be available, and which review task best matches your expertise?

Thank you,  
[sender and team]

**Coordinator:** send this yourself after completing the placeholders. This repository does not send invitations or recruit reviewers.

## 2. What the project does and what we are testing

The core flow is:

**Entered text/image → medication mentions → verified ingredient IDs → source pair screening → human correction → recomputed findings → evidence graph and report.**

An unresolved reading remains unresolved until a person confirms it. Candidate suggestions come from name similarity; interaction possibilities influence review order, not candidate identity. A missing source record is a coverage state, not a statement that a combination is safe. The evidence graph links mentions, ingredients, findings, source records, passages, documents, and correction decisions. Candidate possibilities are distinct from confirmed findings.

Our research hypothesis is that reviewing uncertain identities according to their possible effect on interaction findings recovers more high-severity **source records** per correction than reviewing by recognition uncertainty alone. Additional studies can assess explanation support and whether the evidence interface helps reviewers locate and audit sources. These are questions to test; no clinical benefit, time saving, or unique novelty is assumed.

The prototype does not model the full patient context. Extracted dose, route, and frequency do not make pair screening a patient-specific risk assessment. The graph is a local evidence/provenance graph; it does not discover new clinical interactions.

## 3. Select reviewers and assign independent tasks

| Task | Suitable reviewer | Material provided | Return |
|---|---|---|---|
| A. Reference annotation | Pharmacist, clinician, or researcher with demonstrable medication-identification and interaction-review expertise | De-identified source text/images and declared reference resources; no app predictions initially | Medication labels, reviewed pairs, source references, unresolved cases |
| B. Explanation assessment | Domain expert able to interpret the supplied drug-interaction evidence | Blinded outputs, evidence packets, and a fixed omission checklist | Claim ledger, citation judgments, and completed rating rows |
| C. Workflow and traceability | Intended users; separate expert and student/nonexpert groups in reporting | App, practice cases, assigned task sheets | Task outcomes, timings, errors, and comments |
| D. Technical acceptance | Teammate or software tester | Synthetic examples and manual test guide | Reproducible pass/fail reports |

Student usability feedback and software test passes cannot substitute for expert medication or clinical-source labels. Record each expert's relevant professional role, experience, and conflicts. Keep contact details and any identity-to-code mapping outside the annotation dataset; use codes such as R1 and R2 in study files.

The coordinator manages permissions, assignment, blinding, data conversion, and the frozen study record. Where practical, use different people for reference annotation and timed usability tasks. If someone has already seen a case or its reference answers, record that exposure.

## 4. Preparation before any reviewer starts

1. **Confirm the institutional process.** Ask the supervisor/institution for its determination on ethics review, participant consent, data permission, and permitted sharing. Record the actual answer. Do not invent an approval number or assume de-identification settles every requirement.
2. **Define the population.** For the main study, specify English printed medication entries or images, their origin, and inclusion rules. Keep handwriting, other languages, and deliberately corrupted text in separate strata.
3. **Confirm use and storage permission.** Remove names, contact details, identifiers, signatures, barcodes, and revealing metadata before distributing material. Inspect the resulting image, not just the filename. Keep medication information private too.
4. **Assign real origin groups.** Related crops, photographs, text variants, repeated prescriptions, and examples from the same template/patient/writer where applicable must stay in the same group and split. Placeholder ORIGIN numbers do not establish independence.
5. **Define the reference target.** Decide whether the primary labels measure agreement with a frozen source snapshot or independently reviewed clinical references. Record the source version and severity mapping. These support different claims.
6. **Freeze the development/test assignment.** Approximately 20%/80% is the current starting plan. The proposed 100–200 cases, including the 150-row workbook, are feasibility targets, not a power calculation. Refine sample size using the pilot and the desired precision.
7. **Prepare separate reviewer folders.** Give R1 and R2 their own response sheets. Do not show another reviewer's answers. Do not supply app predictions while they establish the reference.
8. **Run a practice pilot.** Use 5–10 practice cases outside the locked test set. Resolve unclear definitions, estimate workload, and freeze the final instructions before the main study.
9. **Predefine second review.** Prefer two independent experts on all cases if feasible. Otherwise, choose a reproducible random subset, for example 20–30%, with declared strata and selection seed. Difficult-case review can be added, but report its agreement separately from the random subset.
10. **Document participation terms.** Explain the task, voluntary participation, actual withdrawal deadline/process, storage/access/retention, and publication of aggregate findings. Obtain any institution-required consent before collecting responses.

The existing `data/private/annotations.json` contains 150 **pending, unlabelled** cases. Preserve it. It is neither a completed dataset nor a permission record. Do not overwrite it with a second generator run.

## 5. Task A: medication and interaction reference annotation

### A1. Identify every medication independently

For each assigned case:

1. Read the original permitted image or entered text before seeing model/OCR predictions. Transcribe what is visible; do not silently replace an incorrect spelling with the expected drug name.
2. Mark medication entries and non-medication entries. Include medicines the OCR would miss; supplied entry boundaries are a limitation of the text-only benchmark.
3. Record the intended generic ingredient name separately from the observed text. Record all verified ingredients for a combination product. Strength, route, and frequency can be recorded as context but are not identity substitutes.
4. Verify brand-to-ingredient mappings from an identifiable source. Do not infer a brand from which candidate produces the most interactions. The current combination-product catalog is empty; automatic combination recognition must not be assumed.
5. Classify the entry using the following labels and record the reason/reference.

| Entry label | Meaning | Handling |
|---|---|---|
| verified | Identity independently established | Eligible for primary benchmark if the frozen catalog represents it |
| non_medication | Header, instruction, or other text that is not a medicine | Empty gold names; explicitly mark non-medication |
| unreadable | Source cannot be reliably transcribed | Keep unresolved; log why primary annotation is unavailable |
| identity_uncertain | Text is readable but the intended ingredient is not established | Request clarification or adjudication; never guess |
| out_of_catalog | Real, verified medicine absent from the frozen terminology | Log the coverage failure; do not relabel as non-medication |

The current evaluator requires a known, unambiguous catalog identity for every medication in an evaluated case. Cases containing unreadable, unresolved, or out-of-catalog true medications therefore require adjudication or exclusion from that primary dataset. Keep them in the exclusion/coverage log and report their counts. If the team expands the catalog, do it during development, version it, and freeze it again before test evaluation.

### A2. Review ingredient pairs against the declared reference

After identities are established, list every unordered pair of distinct ingredients, including ingredients within a verified combination product. Repeated mentions of the same ingredient are a separate duplicate-entry issue, not a pair of distinct ingredients.

For each pair, record:

- Whether the declared reference reports an interaction: `present`, `no_record_in_reference`, or `not_assessable`.
- The source's original severity wording and its predeclared mapping to `high`, `moderate`, `low`, or `unknown`.
- An actual source record/document identifier, version/date, section, URL where available, and the relevant passage or location.
- Clinical agreement, disagreement, or uncertainty as a separate judgment, with a reason. Retain conflicting source statements.

Do not invent severity from your impression of a mechanism. If a source has no usable severity, retain `unknown`. A generic homepage or drug name alone is insufficient provenance. If the source is unavailable, mark the pair not assessable and return it for coordinator resolution.

For **source-conformance labels**, check the frozen source record independently rather than copying engine alerts. Expert disagreement with that source belongs in the separate judgment field. For **independent clinical-reference labels**, use the agreed independent resources and severity crosswalk; report disagreement with the software as reference disagreement, not automatically proof of clinical safety or harm.

An empty `gold_findings` list means no positive findings under the **declared reference**, after all eligible pairs were reviewed. It must not hide pairs whose reference review is incomplete.

### A3. Annotate actual images separately

Create the reference transcript directly from the image, not the OCR output. Define the evaluated image region consistently: for a cropped medication region, transcribe all text in that region, including visible strengths and instructions. This makes CER/WER comparisons meaningful.

Record image kind as `printed` or `handwriting`, origin group, legibility issues, and preprocessing. Keep related images in the same group. If text cannot be read reliably, retain the image in a failure/coverage log instead of fabricating a transcript. Report the number excluded from transcript-based evaluation.

Printed and handwriting results must be reported separately. The current image script has no split-selection option: supply an already selected, locked image evaluation dataset. It also has fewer validation checks than the text evaluator, so the coordinator must check IDs, groups, references, catalog coverage, and nonempty transcripts explicitly.

### A4. Review additional domain content

Use the domain-audit sheet to check regional aliases, combination-product mappings, food rules, management wording, static demo evidence, live source relevance, and source disagreements. For each item, return `accept`, `revise`, `reject`, or `needs_reference`, with a rationale and source.

Do not mix these judgments into the primary drug–drug finding JSON. Food quantities, dietary context, dose-dependent interpretations, and advice quality are outside that evaluator's current labels. Findings lacking a specific source document should remain explicitly limited.

## 6. Task B: blinded explanation assessment

### B1. Material and blinding

Existing synthetic outputs are available under:

`docs/research_artifacts/checkpoint_2026-10-04/explanations_live/`

The checkpoint contains four cases and 12 outputs across three methods. This is a small explanation pilot, not a clinical validation sample. It can be reviewed without new Groq requests or Free-account quota use.

**Give reviewers:** `blinded_review.json`, their own copy of the blank `ratings.csv`, and the predeclared omission checklist. Supply additional identifiable source documents if needed to assess clinical context; keep method identities hidden.

**Coordinator only:** `outputs.json`, method mappings, generation metadata, and summary files. The blinded file omits explicit method labels, but writing style may reveal the method. Record this limitation and any guessed identity after ratings are locked. Do not reveal methods during adjudication of support judgments.

Have the coordinator randomize output order with a recorded seed and distribute that order. If only the existing order is used, disclose it. Where practical, avoid showing multiple versions of the same case consecutively.

### B2. Count claims consistently

Break the output into factual propositions: ingredient identity, pair severity, mechanism, consequence, management, and what a source states. Split compound sentences when they make independently assessable assertions. Log each eligible claim verbatim in the claim ledger.

Exclude purely procedural text such as “please review this entry” from factual-claim counts. Predefine this rule during the pilot. A factual safety assertion inside a disclaimer is still a claim.

| Claim judgment | Rule |
|---|---|
| supported | Supplied/verified evidence supports the entire claim about the correct pair and context |
| partial | Some, but not all, of the proposition is supported |
| unsupported | No relevant support in the declared evidence set |
| contradicted | Declared evidence contradicts the proposition |
| unevaluable | Missing/unavailable evidence or unresolved interpretation prevents a judgment |

Only `supported` contributes to `supported_claims`. Partial, unsupported, and contradicted claims count in `total_claims` but not in `supported_claims`. For an unevaluable claim, leave the output pending until clarified; if unresolved, exclude that output from the scored set with a recorded reason and denominator. Do not convert missing evidence into a fabricated judgment.

The scorer calls `1 - supported_claims / total_claims` the unsupported-claim rate. With this rubric it is the fraction **not fully supported**, including partial and contradicted claims. Report that definition and the separate ledger counts in the paper.

Copying an exact quote or using an existing citation ID does not automatically support a clinical conclusion. Check drug pair, direction, qualifications, relevance, and omitted context. The app's schema/quote validation cannot replace this assessment.

### B3. Judge citations and omissions

Define a citation unit as one explicit claim-to-source association. If a citation marker applies to two separately counted claims, assess two associations; if one claim cites two sources, assess each association. Log each unit in `citation_ledger.csv`.

A citation is correct only when its identity is valid and the cited material supports that associated claim in context. A real but irrelevant source is incorrect. An uncited factual claim has no citation unit but may still be unsupported. Explain these denominators when reporting results.

The coordinator must prepare the same expected-claim checklist for all methods on a case before rating, independently of their outputs. Include only information required for the declared explanation task and available in the reference evidence. Count each missing checklist item once in `omissions`; do not penalize an output for failing to invent an unavailable mechanism or recommendation.

For each output, complete exactly these scoring columns:

~~~csv
output_id,supported_claims,total_claims,correct_citations,total_citations,omissions,reviewer_id,notes
~~~

All five counts are nonnegative integers; supported/correct counts cannot exceed totals. Zero eligible claims or citations gives an undefined corresponding rate, not 100% quality. Use notes for partial support, context concerns, and checklist IDs omitted.

R1 and R2 should rate independently. Preserve original sheets and ledgers. Create a separate adjudicated sheet where required; do not replace independent answers with consensus and then report consensus as inter-reviewer agreement.

## 7. Task C: workflow, evidence traceability, and usability

### C1. Start and learn the app

The coordinator starts the project locally:

~~~powershell
./.venv/Scripts/python.exe -m streamlit run llm.py
~~~

Reviewers use the displayed local address. Remote reviewers need an approved access method; `localhost` on their computer does not reach the coordinator's app. Use synthetic practice cases, keep external evidence/model calls off, and warm OCR outside timed tasks if OCR startup is not the endpoint.

For each study task, record participant code, case ID/group, interface condition, task order, software/data version, start/end times, completion, wrong corrections, evidence-location errors, and comments. Use an observer or agreed timer; the app does not automatically measure these endpoints.

### C2. Tasks to observe

| Task | Reviewer action | Observer checks |
|---|---|---|
| Identify uncertainty | Build a dossier from an assigned ambiguous entry | Reviewer distinguishes candidates from confirmed ingredients |
| Correct identity | Verify a medicine against the permitted source; confirm the right candidate or manual ingredient | Correct ingredient, correction count, missed medicines, time |
| Preserve unknowns | Encounter an unreadable/out-of-catalog true medicine | Remains unresolved; not excluded as non-medication |
| Inspect coverage | Find an unassessed pair, no-record pair, and unknown-severity record in assigned examples | States distinguished; absence not interpreted as safety |
| Trace a finding | Find the ingredient pair, source record, passage, and document/version | Correct provenance chain; no identity-only PubChem page used as interaction support |
| Inspect a change | Correct an entry and explain which finding changed | Before/after history agrees with actual source findings |
| Check export/session | Download a default report; clear the session | Original text omitted as configured; no private screenshot shared |

Practice example: select **02 / Ambiguous medicine reading**, click **Build review dossier**, review `Warfarln` against the supplied practice answer, choose warfarin, and click **Confirm review decision**. Then inspect **03 / Inspect findings and evidence** and **Interaction map → Evidence graph**. Practice examples must not contribute to study performance.

After each session ask: Which label was confusing? Could you find the actual source? What made a wrong correction likely? Did anything imply more certainty than the evidence justified? What would prevent you from using this for a research audit? Record comments verbatim where consent allows; keep suggestions separate from measured outcomes.

### C3. Conditions for fair comparisons

For a graph-versus-flat study, use the **Evidence graph** explorer versus the **Review desk → Source records and passages** view. Provide the same frozen dossier and evidence in both conditions. The pairwise interaction picture alone is not an equivalent flat evidence interface.

Use two disjoint, matched case sets. Alternate interface order across participants, and rotate case-set/interface assignments so a case set is not always paired with one interface. Avoid the same participant seeing the same case in both conditions. Record practice, prior case exposure, task success, and time; faster incorrect completion is not a benefit.

The app permits navigation to both views and does not enforce study blinding. The coordinator must supervise condition access or arrange controlled sessions and record deviations. Prepare a counterbalanced assignment schedule before recruitment.

Before timing the main study, specify the task endpoint, time limit, what counts as a correct answer, and the handling of abandoned/failed attempts. Report completion and error rates across all assigned attempts alongside timing; do not silently drop unsuccessful tasks. Repeated tasks from the same reviewer and related cases are not independent samples. Preserve participant and origin-group IDs so the analysis can account for both, and agree on that analysis with the supervisor before collecting the main data.

**Review-order timing comparison has an additional dependency:** the app currently presents the impact queue, while uncertainty/order baselines are implemented in the offline simulator. There is no current app selector for a human uncertainty-queue condition. Prepare equivalent baseline sessions/interfaces before claiming a human comparison. Until then, describe review-order results as simulated corrections, and keep observed impact-workflow usability descriptive.

## 8. Response sheets and handover

Copy the blank [response templates](review_templates/README.md) into each reviewer's private study folder. They are header-only CSV files, not fabricated responses. A reviewer may use Excel or a written document; the coordinator handles JSON conversion.

| Sheet | Required use |
|---|---|
| `case_register.csv` | Case/group/split, origin type, image kind, reviewer status |
| `medication_annotations.csv` | Observed text, verified ingredients, status, identity source |
| `pair_annotations.csv` | Every reviewed ingredient pair, reference severity/provenance, disagreement |
| `claim_ledger.csv`, `citation_ledger.csv` | Explanation decisions underlying numerical ratings |
| `expected_claims.csv` | Coordinator's frozen omission checklist, shared across methods |
| `explanation_ratings.csv` | Scorer-compatible totals; use actual assigned output IDs |
| `workflow_tasks.csv` | Actual human timings, completion, errors, task order |
| `domain_audit.csv` | Food/alias/combination/advice/evidence issues |
| `adjudication_log.csv` | Independent disagreements, resolution, source and rationale |
| `exclusion_log.csv` | Missing permission, unreadable text, coverage and unresolved references |

When returning results, include reviewer code, dates, cases completed, any cases skipped, unresolved questions, and whether someone else's judgments or system predictions were seen. Send through the agreed private channel. Do not commit completed sheets, images, or contact details to Git.

Reviewer completion statement:

> I reviewed the listed items within my stated expertise. My uncertain and incomplete judgments remain marked as such. I have recorded the references actually checked and disclosed any prior exposure to outputs or answers. This review is a research assessment, not a recommendation for patient treatment. Permission to identify or acknowledge me: [separate choice and wording].

The coordinator confirms that this statement matches the actual review; it is not a substitute for institutional consent documentation.

## 9. Resolve disagreements and freeze the reference

Compare independent labels before showing reviewers the software predictions. Use the adjudication log for disagreements over transcription, ingredients, pair presence, severity, support, citation relevance, and omissions. Ask an appropriate domain adjudicator to resolve them with recorded evidence, or retain the uncertainty.

Report the number of experts, number of independently double-reviewed cases, how that subset was chosen, initial agreement with numerator/denominator, disagreement categories, and how consensus was reached. For explanation agreement, compare the same matched claims/citation units after reconciling segmentation; aggregate output totals alone are not claim-level agreement. Avoid presenting a percentage without its sample unit.

Freeze source versions, severity crosswalk, eligibility/exclusion rules, case IDs/groups, split, reviewer IDs, thresholds, and the final reference version before running the locked test analysis. Keep original responses, adjudicated labels, and exclusions separately.

## 10. Coordinator: convert and run the existing evaluation tools

Run commands from the repository root. Reviewers can return spreadsheets; they do not need to install Python or edit evaluator JSON.

### Text dataset

Follow the exact JSON structure in [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md#expert-dataset-format). Convert verified generic ingredients to the frozen local IDs; do not ask reviewers to guess opaque IDs. Explicit positive findings need two distinct IDs, an allowed severity, and an actual `source_reference`.

Set `annotation_status: "reviewed"` only for completed cases and include their pseudonymous `reviewer_ids`. Set `permission_confirmed: true` only after permission is documented. Replace the pending label provenance with the actual process. Remove unused placeholder cases from the frozen evaluation copy and account for them in the recruitment/exclusion record.

The evaluator checks completion, exact catalog identities, group leakage, and identical-input leakage. The coordinator must additionally check that pair IDs belong to the case and all pairs have received a reference judgment; schema validation alone does not establish good annotation.

Use a newly named private reference file and output directory:

~~~powershell
./.venv/Scripts/python.exe scripts/benchmark.py --dataset data/private/expert_review/frozen_text_v1.json --split test --output artifacts/expert-study-v1
~~~

The primary endpoint remains high-severity source-record recall for impact versus uncertainty at 50% simulated review budget. Report group counts, reference denominators, actual correction counts, and the grouped interval. This experiment does not produce human time measurements.

### Actual image dataset

Prepare `data/private/expert_review/frozen_images_v1.json` with `kind: "expert_reviewed"`, confirmed permission, and `image_cases`. Required fields per case are `id`, `group_id`, `kind`, `image`, `gold_text`, `gold_names`, `gold_findings`, and reviewed status. Include reviewer IDs and provenance in the study record even though the image script does not enforce them. Relative image paths resolve beside this JSON file.

~~~powershell
./.venv/Scripts/python.exe scripts/ocr_benchmark.py --dataset data/private/expert_review/frozen_images_v1.json --output artifacts/expert-images-v1
~~~

Report normalized-case/whitespace CER/WER, failures, ingredient/finding recall before correction, printed/handwriting strata, and exclusions. These private datasets and output directories are ignored by Git.

### Explanation ratings

For a **new** private review round, the following copies the existing pilot without using an API:

~~~powershell
New-Item -ItemType Directory -Path data/private/expert_review/explanation_round_1
Copy-Item -LiteralPath docs/research_artifacts/checkpoint_2026-10-04/explanations_live/outputs.json -Destination data/private/expert_review/explanation_round_1/outputs.json
Copy-Item -LiteralPath docs/research_artifacts/checkpoint_2026-10-04/explanations_live/blinded_review.json -Destination data/private/expert_review/explanation_round_1/blinded_review.json
Copy-Item -LiteralPath docs/research_artifacts/checkpoint_2026-10-04/explanations_live/ratings.csv -Destination data/private/expert_review/explanation_round_1/ratings_R1.csv
Copy-Item -LiteralPath docs/research_artifacts/checkpoint_2026-10-04/explanations_live/ratings.csv -Destination data/private/expert_review/explanation_round_1/ratings_R2.csv
~~~

Keep `outputs.json` with the coordinator. Give each reviewer only their assigned blinded material and response sheet. Keep independent completed sheets unchanged. Decide in advance whether the analyzed set is pooled independent ratings or separate adjudicated ratings; label it accurately.

Compile that set into `ratings.csv` in the same folder as `outputs.json`, with one row per reviewed output/reviewer. Check that each assigned output has a completed row, no output/reviewer pair is duplicated, excluded outputs have reasons, and reviewer codes are nonblank. The scorer does not itself enforce assignment completeness or reject duplicate reviewer/output rows.

~~~powershell
./.venv/Scripts/python.exe scripts/score_explanations.py data/private/expert_review/explanation_round_1/ratings.csv
~~~

The scorer requires `outputs.json` beside the ratings file and writes `rated_metrics.json` in that directory. Scoring another sheet there overwrites that metric file; use separate round directories for separate analyses. The displayed “ratings” count is rating rows, not independent patients, outputs, or reviewers. Its aggregate ratios do not provide confidence intervals or inter-rater agreement.

## 11. What the team can report after review

| Completed evidence | Supported reporting |
|---|---|
| Qualified independent labels on permitted locked cases | Identity and finding agreement against the explicitly named reference, with coverage/exclusions |
| Reviewed actual images | OCR and downstream results for the sampled image strata |
| Completed blinded ledgers/ratings | Support, citation, omission results using the declared rubric and pilot sample size |
| Controlled human task records | Observed usability, success, errors, and task time under the actual assignment design |
| Domain audit | Specific accepted/revised/rejected rules and evidence limitations |

No expert endorsement alone establishes clinical safety, treatment benefit, global novelty, or conference acceptance. Keep the current synthetic findings separate from these future observations. Refer to [PAPER_HANDOFF.md](PAPER_HANDOFF.md) for the existing evidence and manuscript scope.

## 12. Coordinator completion checklist

- [ ] Invitation placeholders completed; reviewer task and workload agreed.
- [ ] Institutional determination, source permission, participation terms, and storage arrangements recorded.
- [ ] Material de-identified; private files excluded from Git.
- [ ] Reference target, severity mapping, eligibility, split/group rules, and analysis frozen.
- [ ] Pilot completed outside the test set; actual review time used to plan workload.
- [ ] Appropriate reviewer qualifications and pseudonymous IDs recorded.
- [ ] Independent responses preserved; double-review sampling and exposure recorded.
- [ ] Every eligible medication and pair reviewed; uncertain/excluded cases accounted for.
- [ ] Explanation claims, citations, and expected omissions trace to ledgers/checklists.
- [ ] Human task timings observed; comparison conditions actually available and assigned.
- [ ] Adjudications, source versions, and final reference provenance complete.
- [ ] Evaluators run only on completed references; artifact manifests retained.
- [ ] Manuscript distinguishes software checks, simulation, expert judgments, and observed human results.

Start by sending the invitation for a small pilot. The first actionable review can use the existing synthetic explanation packet and app examples while permitted real cases are collected and independently annotated.
