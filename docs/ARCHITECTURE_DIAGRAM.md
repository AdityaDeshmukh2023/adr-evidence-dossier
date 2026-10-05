# Figure: Evidence-First Drug Interaction Screening Architecture

Use `adr_system_architecture.mmd` as the source for the paper figure. Open it in
the [Mermaid Live Editor](https://mermaid.live), then export as **SVG** for a
paper document or **PNG at 300 dpi** for a presentation.

**Suggested paper caption:**

> **Figure X. Architecture of the evidence-first drug interaction screening
> system.** Downloaded DDInter files are processed offline into a versioned,
> deduplicated interaction lookup and EDA assets. At runtime, manually entered
> or OCR-extracted medication names are normalized and screened through a
> deterministic drug–drug and drug–food rule engine. Optional openFDA/PubChem
> evidence supports a bounded LLM explanation, while alerts, provenance,
> uncertainty, visual interaction maps, and a safety disclaimer are displayed
> to the user.

**Accuracy note for the paper:** DDInter severity determines the DDI alert. The
LLM and optional live APIs do not create interaction alerts or prescribe
treatment; they only provide supporting evidence and an explanation.
