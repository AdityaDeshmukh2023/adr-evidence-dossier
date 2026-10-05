"""Build the reproducible cleaned DDInter snapshot and its EDA deliverables.

Run from repository root: python scripts/preprocess_ddinter.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "ddinter_dataset"
OUTPUT_DIR = ROOT / "data" / "processed_ddinter"
TABLE_DIR = OUTPUT_DIR / "eda_tables"
CHART_DIR = OUTPUT_DIR / "eda_charts"
VERSION = "ddinter-local-2026-09-01"
LEVEL_MAP = {"Major": "high", "Moderate": "moderate", "Minor": "low", "Unknown": "unknown"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise_name(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def pair_key(first: str, second: str) -> str:
    return "|".join(sorted((str(first), str(second))))


def load_raw() -> tuple[pd.DataFrame, list[dict]]:
    paths = sorted(RAW_DIR.glob("ddinter_downloads_code_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No DDInter CSV files found in {RAW_DIR}")
    frames, sources = [], []
    for path in paths:
        code = path.stem.rsplit("_", 1)[-1]
        frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        frame["source_category"] = code
        frames.append(frame)
        sources.append({"file": path.name, "source_category": code, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return pd.concat(frames, ignore_index=True), sources


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required = ["DDInterID_A", "Drug_A", "DDInterID_B", "Drug_B", "Level", "source_category"]
    missing_columns = [column for column in required if column not in raw.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    missing_values = raw[required].isna().sum().to_dict()
    if any(missing_values.values()):
        raise ValueError(f"Missing required values: {missing_values}")

    frame = raw.copy()
    for column in ["DDInterID_A", "Drug_A", "DDInterID_B", "Drug_B", "Level"]:
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq('').any():
            raise ValueError(f'Blank required values after trimming: {column}')
    invalid_levels = sorted(set(frame["Level"]) - set(LEVEL_MAP))
    if invalid_levels:
        raise ValueError(f"Unexpected severity values: {invalid_levels}")
    self_pairs = int((frame["DDInterID_A"] == frame["DDInterID_B"]).sum())
    if self_pairs:
        raise ValueError(f"Found {self_pairs} self-pairs")

    frame["drug_a_normalized"] = frame["Drug_A"].map(normalise_name)
    frame["drug_b_normalized"] = frame["Drug_B"].map(normalise_name)
    frame["pair_key"] = [pair_key(a, b) for a, b in zip(frame["DDInterID_A"], frame["DDInterID_B"])]
    severity_conflicts = int(frame.groupby("pair_key")["Level"].nunique().gt(1).sum())
    if severity_conflicts:
        raise ValueError(f"Found {severity_conflicts} conflicting severity labels")

    # Vectorized canonical orientation: a Python callback per pair is too slow
    # for this 160k-pair dataset.
    a_first = frame["DDInterID_A"] <= frame["DDInterID_B"]
    frame["ordered_a_id"] = frame["DDInterID_A"].where(a_first, frame["DDInterID_B"])
    frame["ordered_a_name"] = frame["Drug_A"].where(a_first, frame["Drug_B"])
    frame["ordered_b_id"] = frame["DDInterID_B"].where(a_first, frame["DDInterID_A"])
    frame["ordered_b_name"] = frame["Drug_B"].where(a_first, frame["Drug_A"])
    first_rows = frame.drop_duplicates("pair_key", keep="first").set_index("pair_key")
    grouped = frame.groupby("pair_key", sort=True)
    cleaned = pd.DataFrame({
        "drug_a_id": first_rows["ordered_a_id"],
        "drug_a_name": first_rows["ordered_a_name"],
        "drug_b_id": first_rows["ordered_b_id"],
        "drug_b_name": first_rows["ordered_b_name"],
        "severity_source": first_rows["Level"],
        "source_categories": grouped["source_category"].agg(lambda values: ",".join(sorted(set(values)))),
        "raw_row_count": grouped.size(),
    }).reset_index()
    cleaned["severity"] = cleaned["severity_source"].map(LEVEL_MAP)
    cleaned["drug_a_normalized"] = cleaned["drug_a_name"].map(normalise_name)
    cleaned["drug_b_normalized"] = cleaned["drug_b_name"].map(normalise_name)
    cleaned = cleaned[["pair_key", "drug_a_id", "drug_a_name", "drug_a_normalized", "drug_b_id", "drug_b_name", "drug_b_normalized", "severity_source", "severity", "source_categories", "raw_row_count"]]
    quality = {
        "raw_rows": int(len(raw)), "final_unique_pairs": int(len(cleaned)),
        "duplicates_removed": int(len(raw) - len(cleaned)), "self_pairs": self_pairs,
        "severity_conflicts": severity_conflicts, "missing_required_values": missing_values,
        "final_duplicate_pair_keys": int(cleaned["pair_key"].duplicated().sum()),
    }
    return cleaned, quality


def create_lookup(cleaned: pd.DataFrame) -> dict:
    drug_index: dict[str, dict] = {}
    for row in cleaned.itertuples(index=False):
        for name, ident in ((row.drug_a_normalized, row.drug_a_id), (row.drug_b_normalized, row.drug_b_id)):
            if name in drug_index and drug_index[name]['id'] != ident:
                raise ValueError(f'Ambiguous normalized drug name maps to different IDs: {name}')
        drug_index.setdefault(row.drug_a_normalized, {"id": row.drug_a_id, "name": row.drug_a_name})
        drug_index.setdefault(row.drug_b_normalized, {"id": row.drug_b_id, "name": row.drug_b_name})
    interactions = {
        row.pair_key: {
            "drug_a_id": row.drug_a_id, "drug_a_name": row.drug_a_name,
            "drug_b_id": row.drug_b_id, "drug_b_name": row.drug_b_name,
            "severity": row.severity, "severity_source": row.severity_source,
            "source_categories": row.source_categories,
        }
        for row in cleaned.itertuples(index=False)
    }
    return {"version": VERSION, "source": "DDInter local processed snapshot", "drug_index": drug_index, "interactions": interactions}


def save_eda(cleaned: pd.DataFrame, raw: pd.DataFrame, quality: dict) -> dict:
    import matplotlib.pyplot as plt
    TABLE_DIR.mkdir(parents=True, exist_ok=True); CHART_DIR.mkdir(parents=True, exist_ok=True)
    raw_categories = raw.groupby("source_category").size().rename("raw_rows").reset_index().sort_values("source_category")
    severity = cleaned.groupby(["severity_source", "severity"]).size().rename("unique_pairs").reset_index()
    severity["severity_order"] = severity["severity_source"].map({"Major": 0, "Moderate": 1, "Minor": 2, "Unknown": 3})
    severity = severity.sort_values("severity_order").drop(columns="severity_order")
    all_degrees = pd.concat([cleaned[["drug_a_id", "drug_a_name"]].rename(columns={"drug_a_id":"drug_id", "drug_a_name":"drug_name"}), cleaned[["drug_b_id", "drug_b_name"]].rename(columns={"drug_b_id":"drug_id", "drug_b_name":"drug_name"})]).groupby(["drug_id", "drug_name"]).size().rename("interaction_count").reset_index().sort_values("interaction_count", ascending=False)
    major = cleaned[cleaned["severity"] == "high"]
    major_degrees = pd.concat([major[["drug_a_id", "drug_a_name"]].rename(columns={"drug_a_id":"drug_id", "drug_a_name":"drug_name"}), major[["drug_b_id", "drug_b_name"]].rename(columns={"drug_b_id":"drug_id", "drug_b_name":"drug_name"})]).groupby(["drug_id", "drug_name"]).size().rename("major_interaction_count").reset_index().sort_values("major_interaction_count", ascending=False)
    overlap = cleaned.assign(source_category=cleaned["source_categories"].str.split(",")).explode("source_category").groupby("source_category").size().rename("unique_pairs_containing_category").reset_index().sort_values("source_category")

    tables = {"raw_rows_by_category": raw_categories, "severity_distribution": severity, "drug_interaction_degree": all_degrees, "top_drugs_overall": all_degrees.head(15), "top_drugs_major": major_degrees.head(15), "source_category_overlap": overlap}
    for name, table in tables.items(): table.to_csv(TABLE_DIR / f"{name}.csv", index=False)

    graph = nx.from_pandas_edgelist(cleaned, "drug_a_id", "drug_b_id")
    major_graph = nx.from_pandas_edgelist(major, "drug_a_id", "drug_b_id")
    network = pd.DataFrame([{"network":"all_interactions", "nodes":graph.number_of_nodes(), "edges":graph.number_of_edges(), "density":nx.density(graph), "connected_components":nx.number_connected_components(graph), "max_degree":max(dict(graph.degree()).values())}, {"network":"major_only", "nodes":major_graph.number_of_nodes(), "edges":major_graph.number_of_edges(), "density":nx.density(major_graph), "connected_components":nx.number_connected_components(major_graph), "max_degree":max(dict(major_graph.degree()).values())}])
    network.to_csv(TABLE_DIR / "network_metrics.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    def chart(path: str): plt.tight_layout(); plt.savefig(CHART_DIR / path, dpi=220, bbox_inches="tight"); plt.close()
    pd.DataFrame({"stage":["Raw rows", "Unique pairs"], "count":[quality["raw_rows"], quality["final_unique_pairs"]]}).plot.bar(x="stage", y="count", legend=False, color=["#78909c", "#2f6f67"], figsize=(7,4), title="DDInter: raw rows and unique interaction pairs"); plt.ylabel("Rows / pairs"); chart("01_raw_vs_cleaned.png")
    severity.plot.bar(x="severity_source", y="unique_pairs", legend=False, color=["#a33d34", "#d98c35", "#4d7c8a", "#8d99a6"], figsize=(7,4), title="Unique DDInter pairs by severity"); plt.ylabel("Unique pairs"); chart("02_severity_distribution.png")
    all_degrees.head(15).sort_values("interaction_count").plot.barh(x="drug_name", y="interaction_count", legend=False, color="#2f6f67", figsize=(8,6), title="Top 15 drugs by interaction count"); plt.xlabel("Known interactions"); chart("03_top_drugs_overall.png")
    major_degrees.head(15).sort_values("major_interaction_count").plot.barh(x="drug_name", y="major_interaction_count", legend=False, color="#a33d34", figsize=(8,6), title="Top 15 drugs by Major interactions"); plt.xlabel("Major interactions"); chart("04_top_drugs_major.png")
    raw_categories.plot.bar(x="source_category", y="raw_rows", legend=False, color="#4d7c8a", figsize=(7,4), title="Raw DDInter rows by source category"); plt.ylabel("Raw rows"); chart("05_rows_by_category.png")

    top_nodes = set(major_degrees.head(30)["drug_id"])
    subgraph = major_graph.subgraph(top_nodes).copy()
    # A paper figure should show relationships, not isolated high-degree nodes
    # whose connecting partners fall outside the selected subset.
    subgraph.remove_nodes_from(list(nx.isolates(subgraph)))
    plt.figure(figsize=(11,9)); position = nx.spring_layout(subgraph, seed=7, k=0.8)
    nx.draw_networkx_edges(subgraph, position, alpha=.26, edge_color="#8b6f65")
    nx.draw_networkx_nodes(subgraph, position, node_size=110, node_color="#a33d34", alpha=.9)
    labels = {node: all_degrees.set_index("drug_id").loc[node, "drug_name"] for node in subgraph.nodes}
    nx.draw_networkx_labels(subgraph, position, labels, font_size=7)
    plt.title("Major-interaction subgraph: top connected drugs"); plt.axis("off"); chart("06_major_interaction_subgraph.png")
    return {"network": network.to_dict(orient="records"), "severity": severity.to_dict(orient="records"), "top_drug": all_degrees.iloc[0].to_dict(), "top_major_drug": major_degrees.iloc[0].to_dict()}


def write_report(quality: dict, eda: dict) -> None:
    lines = [
        "# DDInter EDA Report", "", f"Snapshot version: `{VERSION}`", "",
        "## Data preparation", "", f"- Raw rows: {quality['raw_rows']:,}", f"- Unique unordered drug pairs after deduplication: {quality['final_unique_pairs']:,}", f"- Duplicate raw rows removed: {quality['duplicates_removed']:,}", f"- Severity conflicts: {quality['severity_conflicts']}", f"- Self-pairs: {quality['self_pairs']}", "",
        "## Severity distribution", "",
        *[f"- {row['severity_source']} ({row['severity']}): {row['unique_pairs']:,} unique pairs" for row in eda["severity"]], "",
        "## Observed network facts", "", f"- Most connected drug: {eda['top_drug']['drug_name']} ({eda['top_drug']['interaction_count']:,} known interactions).", f"- Most connected drug in the Major subset: {eda['top_major_drug']['drug_name']} ({eda['top_major_drug']['major_interaction_count']:,} Major interactions).", "",
        "## Paper-ready interpretation", "", "The processed dataset represents known interaction associations from the downloaded DDInter snapshot. Severity is a source label, not a patient-specific prediction. Records labelled Unknown were retained as an evidence-gap class and must not be interpreted as safe combinations. The dataset does not itself provide patient-specific mechanism, management, dose, or treatment recommendations.", "",
        "Charts are in `eda_charts/`; quantitative tables are in `eda_tables/`."
    ]
    (OUTPUT_DIR / "EDA_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    global VERSION
    raw, sources = load_raw(); cleaned, quality = clean(raw)
    VERSION = 'ddinter-content-' + hashlib.sha256(''.join(s['sha256'] for s in sources).encode()).hexdigest()[:16]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT_DIR / "ddinter_cleaned.csv", index=False)
    cleaned.to_parquet(OUTPUT_DIR / "ddinter_cleaned.parquet", index=False)
    lookup = create_lookup(cleaned)
    (OUTPUT_DIR / "ddinter_lookup.json").write_text(json.dumps(lookup, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    eda = save_eda(cleaned, raw, quality)
    metadata = {"version": VERSION, "processed_on": date.today().isoformat(), "schema": {column: str(dtype) for column, dtype in cleaned.dtypes.items()}, "severity_mapping": LEVEL_MAP, "raw_sources": sources, "quality": quality, "eda_summary": eda}
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    write_report(quality, eda)
    print(json.dumps({"output": str(OUTPUT_DIR), **quality}, indent=2))


if __name__ == "__main__": main()
