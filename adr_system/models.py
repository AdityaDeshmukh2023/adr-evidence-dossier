from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Literal

Severity = Literal['high', 'moderate', 'low', 'unknown']
SCHEMA_VERSION = '3.0'


@dataclass(frozen=True)
class Candidate:
    ingredient_ids: tuple[str, ...]
    names: tuple[str, ...]
    score: float  # lexical ranking score, never a probability of harm
    method: str = 'approximate'


@dataclass(frozen=True)
class Medication:
    name: str
    canonical_name: str | None
    strength: str | None = None
    frequency: str | None = None
    normalized: bool = False
    mention_id: str = ''
    ingredient_ids: tuple[str, ...] = ()
    ingredient_names: tuple[str, ...] = ()
    candidates: tuple[Candidate, ...] = ()
    status: str = 'unresolved'
    route: str | None = None
    source_span: tuple[int, int] | None = None
    bbox: tuple[float, ...] | None = None
    ocr_score: float | None = None
    unparsed_text: str = ''


@dataclass(frozen=True)
class Evidence:
    title: str
    url: str
    excerpt: str
    source: str
    retrieved_at: str
    evidence_id: str = ''
    document_id: str = ''
    document_version: str = ''
    section: str = ''
    purpose: str = 'interaction_record'
    excerpt_hash: str = ''

    def __post_init__(self):
        digest = hashlib.sha256(self.excerpt.encode()).hexdigest()
        object.__setattr__(self, 'excerpt_hash', digest)
        if not self.evidence_id:
            identity = f'{self.source}|{self.document_id}|{self.url}|{digest}'
            object.__setattr__(self, 'evidence_id', hashlib.sha256(identity.encode()).hexdigest()[:20])


@dataclass
class InteractionAlert:
    entities: list[str]
    interaction_type: Literal['drug-drug', 'drug-food']
    severity: Severity
    mechanism: str
    management: str
    evidence: list[Evidence]
    data_version: str
    confidence: float | None
    source_record_id: str
    ingredient_ids: list[str] = field(default_factory=list)
    mention_ids: list[str] = field(default_factory=list)
    finding_status: str = 'resolved'
    source_disagreements: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ReviewDecision:
    mention_id: str
    ingredient_ids: tuple[str, ...]
    previous_ids: tuple[str, ...]
    decided_at: str
    action: str = 'resolve'
    before_findings: tuple[str, ...] = ()
    after_findings: tuple[str, ...] = ()


@dataclass
class ReviewItem:
    mention_id: str
    required: bool
    high_changes: int
    other_changes: int
    coverage_changes: int
    ambiguity: float
    reason: str
    previews: list[dict] = field(default_factory=list)


@dataclass
class AnalysisResult:
    run_id: str
    medications: list[Medication]
    foods: list[str]
    alerts: list[InteractionAlert] = field(default_factory=list)
    unsupported_medications: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source_status: dict[str, str] = field(default_factory=dict)
    completeness: str = 'incomplete'
    pair_assessments: list[dict] = field(default_factory=list)
    review_queue: list[ReviewItem] = field(default_factory=list)
    review_history: list[ReviewDecision] = field(default_factory=list)
    manifest: dict = field(default_factory=dict)
    food_assessments: list[dict] = field(default_factory=list)
    candidate_assessments: list[dict] = field(default_factory=list)
    mechanism_hypotheses: list[dict] = field(default_factory=list)
    model_predictions: list[dict] = field(default_factory=list)
    hybrid_status: dict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self):
        return asdict(self)
