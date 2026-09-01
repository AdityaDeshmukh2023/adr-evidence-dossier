from __future__ import annotations

import os

from .models import AnalysisResult

SYSTEM_PROMPT = """You are an educational medication-safety explainer. Use only the structured alerts and evidence supplied. Do not invent interactions, sources, severity, diagnoses, or alternatives. Do not recommend starting, stopping, or changing treatment. If evidence is incomplete, say so. Use concise, plain language and cite sources by their supplied titles."""


def deterministic_explanation(result: AnalysisResult) -> str:
    if not result.alerts:
        return "No known interaction was found in the configured demonstration knowledge base. This is not evidence that the combination is safe; a clinician or pharmacist should assess the full patient context."
    lines = []
    for alert in result.alerts:
        names = " + ".join(alert.entities)
        source_titles = "; ".join(item.title for item in alert.evidence)
        lines.append(f"{names} is flagged as {alert.severity} concern. {alert.mechanism} Guidance: {alert.management} Evidence: {source_titles}.")
    return "\n\n".join(lines)


def bounded_llm_explanation(result: AnalysisResult) -> tuple[str, str]:
    """Return LLM explanation only when configured; deterministic output is the safe fallback."""
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        return deterministic_explanation(result), "deterministic fallback (no Groq key)"
    try:
        from groq import Groq
        payload = result.to_dict()
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=500,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": f"Structured evidence JSON:\n{payload}"}],
        )
        return completion.choices[0].message.content, "bounded Groq explanation"
    except Exception:
        return deterministic_explanation(result), "deterministic fallback (LLM unavailable)"
