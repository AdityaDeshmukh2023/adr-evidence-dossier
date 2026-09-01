from __future__ import annotations

from html import escape
from datetime import datetime, timezone

from .models import AnalysisResult

DISCLAIMER = "Educational decision-support result only. Do not start, stop, or change medication without a qualified clinician."


def render_html_report(result: AnalysisResult, explanation: str) -> str:
    cards = []
    for alert in result.alerts:
        links = "".join(f'<li><a href="{escape(item.url)}">{escape(item.title)}</a> — {escape(item.source)}</li>' for item in alert.evidence)
        cards.append(f"<section><h3>{escape(' + '.join(alert.entities))} <small>{escape(alert.severity.upper())}</small></h3><p><b>Type:</b> {escape(alert.interaction_type)}<br><b>Mechanism:</b> {escape(alert.mechanism)}<br><b>Guidance:</b> {escape(alert.management)}</p><ul>{links}</ul></section>")
    no_alert = "<p>No known interaction found in the configured knowledge base.</p>" if not cards else ""
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>ADR Evidence Report</title><style>body{{font-family:Georgia,serif;max-width:820px;margin:40px auto;color:#172235}}section{{border-left:5px solid #bb5b42;padding:12px 18px;margin:14px 0;background:#f7f5f1}}small{{color:#9a3b2f}}.notice{{background:#fff1d6;padding:14px}}</style></head><body><h1>ADR Evidence Report</h1><p>Reproducibility ID: <code>{escape(result.run_id)}</code><br>Generated: {datetime.now(timezone.utc).isoformat()}</p><p><b>Medications:</b> {escape(', '.join(m.name for m in result.medications) or 'None')}</p><p><b>Foods:</b> {escape(', '.join(result.foods) or 'None')}</p><div class='notice'>{DISCLAIMER}</div><h2>Alerts</h2>{no_alert}{''.join(cards)}<h2>Evidence-bounded explanation</h2><p>{escape(explanation).replace(chr(10), '<br>')}</p><h2>Source status</h2><pre>{escape(str(result.source_status))}</pre></body></html>"""
