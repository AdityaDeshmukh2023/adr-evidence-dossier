"""ADR Evidence Dossier — run with `streamlit run llm.py`."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from adr_system.engine import analyze_medications, severity_rank
from adr_system.evidence import live_evidence_for
from adr_system.explanations import bounded_llm_explanation
from adr_system.report import DISCLAIMER, render_html_report

load_dotenv()
st.set_page_config(page_title="ADR Evidence Dossier", page_icon="✦", layout="wide")

CASES = {
    "Warfarin + aspirin — high risk": {"meds": "Warfarin\nAspirin", "foods": "Spinach"},
    "Simvastatin + grapefruit — food interaction": {"meds": "Simvastatin", "foods": "Grapefruit juice"},
    "Fluoxetine + tramadol — high risk": {"meds": "Fluoxetine\nTramadol", "foods": ""},
    "No known interaction — conservative result": {"meds": "Metformin\nAmoxicillin", "foods": "Banana"},
}


def inject_css() -> None:
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:opsz,wght@9..144,500;9..144,700&display=swap');
    .stApp{background:radial-gradient(circle at 85% 2%,#dbe5df 0,transparent 26%),#f3efe6;color:#172235}h1,h2,h3{font-family:Fraunces,Georgia,serif!important;color:#172235}.stMarkdown,p,label{font-family:Georgia,serif}[data-testid='stSidebar']{background:#172235}[data-testid='stSidebar'] *{color:#f4efe5!important}.eyebrow{font:500 11px 'DM Mono',monospace;letter-spacing:.16em;color:#a64935;text-transform:uppercase}.hero{border-top:1px solid #172235;border-bottom:1px solid #172235;padding:1.5rem 0 1.1rem;margin-bottom:1.2rem}.risk-card{background:#fffdf8;border:1px solid #d5ccbe;border-left:7px solid #a64935;padding:1rem 1.2rem;margin:.6rem 0}.notice{background:#fff0d7;border:1px solid #dfb36d;padding:.85rem 1rem;color:#50320d}.muted{color:#617080;font:12px 'DM Mono',monospace}.stButton button,.stDownloadButton button{border-radius:0;border:1px solid #172235;background:#172235;color:#fff}
    </style>""", unsafe_allow_html=True)


def header() -> None:
    st.markdown("<div class='hero'><div class='eyebrow'>Research prototype · evidence-first pharmacovigilance</div><h1>ADR Evidence Dossier</h1><p>A transparent, deterministic interaction screen with source-linked explanations. It is not a clinical decision-maker.</p></div>", unsafe_allow_html=True)


def run_ocr(uploaded) -> str:
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix or ".png") as temp:
            temp.write(uploaded.getvalue()); path = temp.name
        from paddle_ocr import extract_text_from_image
        return extract_text_from_image(path)
    finally:
        if path and os.path.exists(path): os.remove(path)


def enrich_live(result) -> None:
    evidence, statuses = live_evidence_for([m.canonical_name for m in result.medications if m.canonical_name])
    result.source_status.update(statuses)
    for alert in result.alerts:
        for entity in alert.entities: alert.evidence.extend(evidence.get(entity, []))


def render_alerts(result, live_enabled: bool) -> str:
    # Streamlit reruns on every interaction; do not append cached live evidence twice.
    if live_enabled and not any(key.startswith("openFDA:") for key in result.source_status):
        enrich_live(result)
    explanation, mode = bounded_llm_explanation(result)
    st.caption(f"Explanation mode: {mode} · Run ID: {result.run_id}")
    if result.alerts:
        highest = max(result.alerts, key=lambda item: severity_rank(item.severity)).severity
        st.markdown(f"<div class='notice'><b>{highest.upper()} alert level present.</b> {DISCLAIMER}</div>", unsafe_allow_html=True)
        for i, alert in enumerate(sorted(result.alerts, key=lambda item: severity_rank(item.severity), reverse=True), 1):
            st.markdown(f"<div class='risk-card'><div class='eyebrow'>{alert.interaction_type} · {alert.severity} · confidence {alert.confidence:.0%}</div><h3>{' + '.join(alert.entities)}</h3><p><b>Why it is flagged:</b> {alert.mechanism}</p><p><b>Educational guidance:</b> {alert.management}</p></div>", unsafe_allow_html=True)
            with st.expander(f"Evidence dossier {i}"):
                st.caption(f"Record: {alert.source_record_id} · Dataset: {alert.data_version}")
                for item in alert.evidence:
                    st.markdown(f"[{item.title}]({item.url}) — {item.source} ({item.retrieved_at})")
                    st.write(item.excerpt)
    else: st.info("No known interaction was found in the configured demonstration knowledge base. This is not a safety guarantee.")
    if result.unsupported_medications: st.warning("Unrecognized medication names: " + ", ".join(result.unsupported_medications))
    for warning in result.warnings: st.warning(warning)
    st.subheader("Evidence-bounded explanation"); st.write(explanation)
    st.subheader("Source status"); st.json(result.source_status)
    return explanation


def network_graph(result) -> None:
    if not result.alerts:
        st.info("The network map is available when an interaction is detected."); return
    nodes = sorted({node for alert in result.alerts for node in alert.entities})
    positions = {node: (i % 3, -(i // 3)) for i, node in enumerate(nodes)}
    colors = {"high":"#a64935", "moderate":"#d08b35", "low":"#4c7180", "unknown":"#7c8794"}
    fig = go.Figure()
    for alert in result.alerts:
        a, b = alert.entities; x0, y0 = positions[a]; x1, y1 = positions[b]
        fig.add_trace(go.Scatter(x=[x0,x1], y=[y0,y1], mode="lines", line=dict(width=4,color=colors[alert.severity]), hoverinfo="text", text=[f"{alert.severity}: {alert.mechanism}"]*2, showlegend=False))
    fig.add_trace(go.Scatter(x=[positions[n][0] for n in nodes],y=[positions[n][1] for n in nodes],mode="markers+text",text=nodes,textposition="bottom center",marker=dict(size=34,color="#172235",line=dict(color="#f3efe6",width=2)),textfont=dict(family="Georgia",color="#172235"),hovertemplate="%{text}<extra></extra>",showlegend=False))
    fig.update_layout(height=420,margin=dict(l=20,r=20,t=20,b=20),paper_bgcolor="#fffdf8",plot_bgcolor="#fffdf8",xaxis=dict(visible=False),yaxis=dict(visible=False))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Edge colour encodes high, moderate, or low risk. Use Results for citations.")


def analyze_page() -> None:
    header(); st.markdown(f"<div class='notice'>{DISCLAIMER}</div>", unsafe_allow_html=True)
    left, right = st.columns([3,2], gap="large")
    with left:
        choice = st.selectbox("Load a synthetic showcase case", ["Start with a blank analysis", *CASES])
        case = CASES.get(choice, {"meds":"", "foods":""})
        meds = st.text_area("Medications (one per line)", value=case["meds"], height=160, placeholder="Warfarin\nAspirin")
        foods = st.text_area("Relevant foods or drinks (optional)", value=case["foods"], placeholder="Spinach, grapefruit juice")
        live = st.toggle("Retrieve optional live openFDA/PubChem evidence", value=False)
        if st.button("Analyze evidence", type="primary"):
            st.session_state["result"] = analyze_medications(meds, foods); st.session_state["live"] = live
    with right:
        st.subheader("Optional prescription OCR")
        upload = st.file_uploader("Upload image", type=["png","jpg","jpeg"], help="Processed temporarily and not stored.")
        if upload and st.button("Extract text for editing"):
            try: st.session_state["ocr_text"] = run_ocr(upload)
            except Exception as error: st.error(f"OCR was unavailable: {error}")
        if st.session_state.get("ocr_text"): st.text_area("OCR output — copy/edit into medications", st.session_state["ocr_text"], height=170)
        st.markdown("<p class='muted'>No names, phones, images, or medical records are persisted.</p>", unsafe_allow_html=True)
    if "result" in st.session_state:
        st.divider(); st.header("Results")
        explanation = render_alerts(st.session_state["result"], st.session_state.get("live",False))
        report = render_html_report(st.session_state["result"], explanation)
        st.download_button("Download evidence report (HTML)", report, file_name=f"adr-evidence-{st.session_state['result'].run_id}.html", mime="text/html")


def case_library() -> None:
    header(); st.subheader("Synthetic case library"); st.write("Educational test fixtures, not patient records.")
    for name, case in CASES.items():
        with st.expander(name): st.code(f"Medications:\n{case['meds']}\n\nFoods:\n{case['foods'] or 'None'}")


def about() -> None:
    header(); st.subheader("Methods and limits")
    st.markdown("The alert engine uses a small, versioned demonstration snapshot. The LLM, when configured, only summarizes supplied alert records and evidence. Live sources are supporting evidence and never create alerts.\n\n**Not clinical validation.** Coverage is incomplete; no EHR, dose adjustment, kidney/liver function, pregnancy status, or full patient history is assessed. A missing alert is not a safety guarantee.")
    st.subheader("Reproducibility"); st.json({"local snapshots":["DDI demo 2026-08-31","food rules 2026-08-31"],"privacy":"No persistent patient data","evaluation":"Synthetic case suite and faculty rubric planned"})


inject_css()
page = st.sidebar.radio("Dossier", ["Analyze","Interaction Map","Case Library","About & Methods"])
if page == "Analyze": analyze_page()
elif page == "Interaction Map":
    header(); st.subheader("Interaction map")
    if "result" in st.session_state: network_graph(st.session_state["result"])
    else: st.info("Run an analysis first to create an interaction map.")
elif page == "Case Library": case_library()
else: about()
