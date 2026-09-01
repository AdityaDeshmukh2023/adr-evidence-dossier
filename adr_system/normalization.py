from __future__ import annotations

import re

from .models import Medication
from .data import ddinter_drug_index

# Keep aliases explicit and reviewable. This is intentionally small for the
# demonstrator; an external terminology service can be added later.
ALIASES = {
    "asa": "aspirin", "acetylsalicylic acid": "aspirin",
    "coumadin": "warfarin", "advil": "ibuprofen", "motrin": "ibuprofen",
    "tylenol": "acetaminophen", "paracetamol": "acetaminophen",
    "zoloft": "sertraline", "nardil": "phenelzine", "cymbalta": "duloxetine",
    "lipitor": "atorvastatin", "zocor": "simvastatin",
}
KNOWN = {
    "aspirin", "warfarin", "ibuprofen", "acetaminophen", "sertraline",
    "phenelzine", "duloxetine", "atorvastatin", "simvastatin", "linezolid",
    "clopidogrel", "fluoxetine", "tramadol", "metformin", "amoxicillin",
}


def normalise_medication(value: str) -> Medication:
    clean = " ".join(value.strip().lower().split())
    # Preserve optional dose/frequency for the report but do not use it to infer risk.
    name = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|%)\b.*$", "", clean).strip(" ,-")
    canonical = ALIASES.get(name, name)
    known = canonical in ddinter_drug_index() or canonical in KNOWN
    return Medication(name=value.strip(), canonical_name=canonical if known else None, normalized=known)


def parse_medication_lines(text: str) -> list[Medication]:
    candidates = [part.strip() for part in re.split(r"[\n,;]+", text) if part.strip()]
    return [normalise_medication(item) for item in candidates]


def normalise_foods(text: str) -> list[str]:
    return [" ".join(item.lower().split()) for item in re.split(r"[\n,;]+", text) if item.strip()]
