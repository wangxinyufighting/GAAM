from __future__ import annotations

from typing import Any


ONTOLOGY_VERSION = "gaam_oracle_ontology_v1"


ALLOWED_TARGET_QUESTION_TYPES = {
    "single_session_user",
    "single_session_assistant",
    "single_session_preference",
    "multi_session",
    "temporal_reasoning",
    "knowledge_update",
}


ALLOWED_TOPICS = {
    "health_fitness",
    "sleep_wellness",
    "education",
    "career_work",
    "commute_transportation",
    "finance_housing",
    "food_dining",
    "travel",
    "arts_culture",
    "entertainment_media",
    "music_events",
    "sports_hobbies",
    "craft_collectibles",
    "photography_video",
    "shopping_products",
    "clothing_fashion",
    "coupons_deals",
    "pets",
    "family_relationships",
    "home_living",
    "language_learning",
    "work_project",
    "software_code",
    "research_paper",
    "benchmark_dataset",
    "model_training",
    "memory_system",
    "schedule_routine",
    "personal_fact",
    "preference",
    "instruction_requirement",
    "problem_solution",
    "update_correction",
    "qa_evidence",
    "assistant_recommendation",
    "time_date",
    "location_place",
    "quantity_count",
    "other",
}


ALLOWED_EVENT_CATEGORIES = {
    "preference",
    "personal_fact",
    "instruction",
    "requirement",
    "update",
    "correction",
    "question_answer",
    "assistant_recommendation",
    "project_setting",
    "decision",
    "problem",
    "solution",
    "evidence",
    "attendance",
    "purchase",
    "creation",
    "relocation",
    "schedule_assignment",
    "measurement",
    "general",
    "other",
}


ALLOWED_FACT_TYPES = {
    "preference",
    "stable_personal_fact",
    "episodic_event",
    "project_requirement",
    "project_decision",
    "implementation_detail",
    "problem",
    "solution",
    "update",
    "correction",
    "qa_evidence",
    "assistant_recommendation",
    "recommendation_constraint",
    "location",
    "date_time",
    "duration",
    "frequency",
    "quantity_count",
    "measurement",
    "financial_amount",
    "schedule_assignment",
    "sequence_order",
    "temporal_interval",
    "named_entity",
    "attribute",
    "relationship",
    "summary_evidence",
    "general",
    "other",
}


ALLOWED_TEMPORAL_SCOPES = {
    "active",
    "stale",
    "historical",
    "future_intent",
    "recurring",
    "relative_time",
    "time_interval",
    "unknown",
}


ALLOWED_ABSTRACT_SCOPES = {
    "communication_preference",
    "project_design",
    "stable_user_fact",
    "task_context",
    "behavior_pattern",
    "preference_summary",
    "temporal_summary",
    "multi_session_summary",
    "update_summary",
    "recommendation_profile",
    "general",
}


def ontology_prompt_block() -> str:
    """Return a compact JSON-like ontology block for prompts."""
    data: dict[str, Any] = {
        "ontology_version": ONTOLOGY_VERSION,
        "target_question_types_to_cover": sorted(ALLOWED_TARGET_QUESTION_TYPES),
        "allowed_topics": sorted(ALLOWED_TOPICS),
        "allowed_event_categories": sorted(ALLOWED_EVENT_CATEGORIES),
        "allowed_fact_types": sorted(ALLOWED_FACT_TYPES),
        "allowed_temporal_scopes": sorted(ALLOWED_TEMPORAL_SCOPES),
        "allowed_abstract_scopes": sorted(ALLOWED_ABSTRACT_SCOPES),
    }
    import json

    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def normalize_ontology_value(value: str | None, allowed: set[str], fallback: str) -> str:
    if not value:
        return fallback
    normalized = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    return normalized if normalized in allowed else fallback
