# Simple Architecture Diagram for Review

Use `simple_architecture_diagram.mmd` in the
[Mermaid Live Editor](https://mermaid.live), then export it as SVG or PNG.

**Suggested figure caption:**

> **Figure X. Simplified architecture of the proposed drug interaction screening
> system.** The user enters medication and optional food information. The system
> standardizes the medicine names, checks known drug–drug interactions using the
> DDInter dataset and drug–food interactions using curated rules, then displays
> severity, supporting evidence, a graph, and a downloadable report. The system
> is an educational decision-support tool and does not replace clinical advice.

## How to explain it in a review

> “First, the user enters the medicines they are taking. Our system cleans the
> medicine names and checks the DDInter dataset for known interactions. It also
> checks a small verified list of food interactions. If a risk is found, we show
> its severity and source instead of only giving an AI-generated answer. Finally,
> the result is displayed as an alert, graph, and report. The system does not
> prescribe medicines; it helps the user or healthcare professional identify
> combinations that need review.”
