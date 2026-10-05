from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def load_json(name: str) -> tuple[dict, str]:
    path = DATA_DIR / name
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def load_interactions() -> tuple[dict, str]:
    return load_json("ddinter_demo_snapshot.json")


def load_food_rules() -> tuple[dict, str]:
    return load_json("drug_food_rules.json")


@lru_cache(maxsize=1)
def load_ddinter_lookup() -> tuple[dict | None, str | None]:
    """Load the generated full DDInter lookup once, falling back when absent."""
    path = DATA_DIR / "processed_ddinter" / "ddinter_lookup.json"
    if not path.exists():
        return None, None
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def find_ddinter_pair(first_name: str, second_name: str) -> dict | None:
    lookup, _ = load_ddinter_lookup()
    if not lookup:
        return None
    first = lookup["drug_index"].get(first_name)
    second = lookup["drug_index"].get(second_name)
    if not first or not second or first["id"] == second["id"]:
        return None
    key = "|".join(sorted((first["id"], second["id"])))
    return lookup["interactions"].get(key)


def ddinter_drug_index() -> dict[str, dict]:
    lookup, _ = load_ddinter_lookup()
    return lookup["drug_index"] if lookup else {}
