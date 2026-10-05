"""Medication review dossier: streamlit run llm.py."""
import streamlit as st
from dotenv import load_dotenv
from adr_system.ui import main

load_dotenv()
st.set_page_config(page_title="Medication Review Dossier", page_icon="?", layout="wide")

def inject_css() -> None:
    st.markdown("""<style>
    :root { --ink:#102A43; --muted:#486581; --paper:#F6F8FA; --panel:#FFFFFF; --line:#BCCCDC; --teal:#0F766E; --teal-dark:#115E59; --rust:#B42318; --amber-bg:#FFF7E6; }
    .stApp, [data-testid="stAppViewContainer"] { background:var(--paper)!important; color:var(--ink)!important; }
    [data-testid="stHeader"] { background:rgba(246,248,250,.92)!important; }
    .main .block-container { max-width:1180px; padding-top:2.25rem; padding-bottom:3rem; }
    h1,h2,h3 { font-family:Fraunces,Georgia,serif!important; color:var(--ink)!important; }
    p,li,label,[data-testid="stMarkdownContainer"] { color:var(--ink)!important; }
    .eyebrow { font:500 11px 'DM Mono',monospace; letter-spacing:.15em; color:var(--teal)!important; text-transform:uppercase; }
    .hero { border-top:3px solid var(--teal); border-bottom:1px solid var(--line); padding:1.4rem 0 1.2rem; margin-bottom:1.35rem; }
    .hero p { color:var(--muted)!important; font-size:1.05rem; }
    .risk-card { background:var(--panel); border:1px solid var(--line); border-left:7px solid var(--rust); box-shadow:0 2px 8px rgba(16,42,67,.08); padding:1rem 1.2rem; margin:.75rem 0; border-radius:4px; }
    .notice { background:var(--amber-bg); border:1px solid #F3C66B; border-left:5px solid #D97706; padding:.9rem 1rem; color:#713F12!important; border-radius:4px; }
    .notice * { color:#713F12!important; }
    .muted { color:var(--muted)!important; font:12px 'DM Mono',monospace; }

    /* Inputs remain readable irrespective of the Streamlit or browser theme. */
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-baseweb="select"] > div, [data-testid="stNumberInput"] input { background:var(--panel)!important; color:var(--ink)!important; border-color:#829AB1!important; }
    [data-testid="stTextInput"] input::placeholder, [data-testid="stTextArea"] textarea::placeholder { color:#627D98!important; opacity:1!important; }
    [data-baseweb="select"] *, [data-baseweb="select"] input { color:var(--ink)!important; }
    [data-baseweb="popover"], [role="listbox"] { background:var(--panel)!important; }
    [role="option"] { color:var(--ink)!important; background:var(--panel)!important; }
    [role="option"]:hover { background:#D9EAE7!important; }
    [data-testid="stFileUploader"] { background:var(--panel); border:1px dashed #829AB1; border-radius:6px; padding:.4rem; }
    [data-testid="stFileUploader"] * { color:var(--ink)!important; }
    [data-testid="stExpander"] { background:var(--panel); border:1px solid var(--line); border-radius:5px; }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color:var(--ink)!important; }
    [data-testid="stAlert"] { border-radius:5px; }

    .stButton button, .stDownloadButton button { border-radius:4px!important; border:1px solid var(--teal-dark)!important; background:var(--teal)!important; color:#FFFFFF!important; font-weight:700!important; }
    .stButton button:hover, .stDownloadButton button:hover { background:var(--teal-dark)!important; color:#FFFFFF!important; border-color:var(--teal-dark)!important; }
    [data-testid="stToggle"] label { color:var(--ink)!important; }

    [data-testid="stSidebar"] { background:#102A43!important; }
    [data-testid="stSidebar"] * { color:#F8FAFC!important; }
    [data-testid="stSidebar"] [data-baseweb="radio"] label { padding:.35rem .25rem; border-radius:4px; }
    [data-testid="stSidebar"] [data-baseweb="radio"] label:hover { background:#243B53; }
    [data-testid="stSidebar"] [data-baseweb="radio"] input:checked + div { background:#2CB1BC!important; }
    </style>""", unsafe_allow_html=True)



inject_css()
main()
