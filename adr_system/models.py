from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Severity = Literal["high", "moderate", "low", "unknown"]


@dataclass(frozen=True)
class Medication:
    name: str
    canonical_name: str | None
    strength: str | None = None
    frequency: str | None = None
    normalized: bool = False


@dataclass(frozen=True)
class Evidence:
    title: str
    url: str
    excerpt: str
    source: str
    retrieved_at: str


@dataclass
class InteractionAlert:
    entities: list[str]
    interaction_type: Literal["drug-drug", "drug-food"]
    severity: Severity
    mechanism: str
    management: str
    evidence: list[Evidence]
    data_version: str
    confidence: float
    source_record_id: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnalysisResult:
    run_id: str
    medications: list[Medication]
    foods: list[str]
    alerts: list[InteractionAlert] = field(default_factory=list)
    unsupported_medications: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source_status: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "medications": [asdict(item) for item in self.medications],
            "foods": self.foods,
            "alerts": [item.to_dict() for item in self.alerts],
            "unsupported_medications": self.unsupported_medications,
            "warnings": self.warnings,
            "source_status": self.source_status,
        }
