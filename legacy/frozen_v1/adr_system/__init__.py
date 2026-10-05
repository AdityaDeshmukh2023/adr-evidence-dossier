"""Evidence-first ADR decision support components.

This package is for research and educational demonstrations only. It is not a
medical device and does not provide prescribing advice.
"""

from .engine import analyze_medications
from .models import AnalysisResult, InteractionAlert, Medication

__all__ = ["analyze_medications", "AnalysisResult", "InteractionAlert", "Medication"]
