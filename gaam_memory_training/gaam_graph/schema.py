from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class NodeType(str, Enum):
    SESSION = "session"
    EVENT = "event"
    ENTITY = "entity"
    FACT = "fact"
    TOPIC = "topic"
    ABSTRACT = "abstract_memory"


class EdgeType(str, Enum):
    BELONGS_TO = "BELONGS_TO"
    NEXT = "NEXT"
    MENTIONS = "MENTIONS"
    EXTRACTED_AS = "EXTRACTED_AS"
    FACT_SUBJECT = "FACT_SUBJECT"
    FACT_OBJECT = "FACT_OBJECT"
    BELONGS_TO_TOPIC = "BELONGS_TO_TOPIC"
    SUPPORTS = "SUPPORTS"
    SUPERSEDES = "SUPERSEDES"
    DUPLICATE_OF = "DUPLICATE_OF"
    COMPLEMENTS = "COMPLEMENTS"
    CONTRADICTS = "CONTRADICTS"
    RELATED_TO = "RELATED_TO"
    ABSTRACTS = "ABSTRACTS"


ALLOWED_ENTITY_TYPES = {
    "person", "user", "assistant", "project", "method", "paper", "code_repository",
    "benchmark", "dataset", "model", "metric", "task", "preference", "location",
    "time", "organization", "product", "medical_entity", "other",
}

ALLOWED_PREDICATES = {
    "has_attribute",
    "has_preference",
    "prefers",
    "rejects",
    "uses_benchmark",
    "refers_to_method",
    "has_design_choice",
    "has_requirement",
    "has_problem",
    "has_solution",
    "updates_previous_choice",
    "asks_to_optimize",
    "belongs_to_project",
    "current_subproblem",
    "mentions_evidence",
    "did_action",
    "will_do_action",
    "said",
    "other",
}

ALLOWED_FACT_RELATIONS = {
    "duplicate", "complement", "refine", "contradict", "supersede", "unrelated"
}


class EntityProposal(BaseModel):
    surface: str
    canonical_name: str
    entity_type: str = "other"
    confidence: float = Field(default=0.8, ge=0, le=1)


class FactProposal(BaseModel):
    subject: str
    predicate: str
    object: str
    text: str
    fact_type: str = "general"
    temporal_scope: str = "unknown"
    confidence: float = Field(default=0.8, ge=0, le=1)


class EventAnalysis(BaseModel):
    memory_worthy: bool = True
    event_category: str = "general"
    importance: int = Field(default=3, ge=1, le=5)
    topic: Optional[str] = None
    entities: List[EntityProposal] = Field(default_factory=list)
    facts: List[FactProposal] = Field(default_factory=list)
    reason: str = ""


class FactRelation(BaseModel):
    relation: str
    reason: str = ""
    confidence: float = Field(default=0.8, ge=0, le=1)


class AbstractProposal(BaseModel):
    summary: str
    scope: str = "general"
    supporting_fact_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0, le=1)


class Node(BaseModel):
    id: str
    type: NodeType
    attrs: Dict[str, Any] = Field(default_factory=dict)


class Edge(BaseModel):
    source: str
    target: str
    type: EdgeType
    attrs: Dict[str, Any] = Field(default_factory=dict)
