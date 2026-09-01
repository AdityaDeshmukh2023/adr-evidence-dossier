# DDInter EDA Report

Snapshot version: `ddinter-local-2026-09-01`

## Data preparation

- Raw rows: 222,383
- Unique unordered drug pairs after deduplication: 160,235
- Duplicate raw rows removed: 62,148
- Severity conflicts: 0
- Self-pairs: 0

## Severity distribution

- Major (high): 26,914 unique pairs
- Moderate (moderate): 96,675 unique pairs
- Minor (low): 6,833 unique pairs
- Unknown (unknown): 29,813 unique pairs

## Observed network facts

- Most connected drug: Dexamethasone (910 known interactions).
- Most connected drug in the Major subset: Siponimod (504 Major interactions).

## Paper-ready interpretation

The processed dataset represents known interaction associations from the downloaded DDInter snapshot. Severity is a source label, not a patient-specific prediction. Records labelled Unknown were retained as an evidence-gap class and must not be interpreted as safe combinations. The dataset does not itself provide patient-specific mechanism, management, dose, or treatment recommendations.

Charts are in `eda_charts/`; quantitative tables are in `eda_tables/`.