"""
Tests for MemorySchema.

Run with: pytest gaam_memory_training/tests/test_memory_schema.py
"""

import pytest

from gaam_graph.memory_schema import (
    FORBIDDEN_MEMORY_KEYS,
    CurrentMemory,
    MemoryEdgeType,
    MemoryNodeType,
    MemoryStatus,
    assert_no_memory_leakage,
    empty_current_memory,
    memory_to_dict,
    validate_current_memory,
)


def test_empty_current_memory():
    """Test empty current memory creation."""
    memory = empty_current_memory("test_record")

    assert memory["record_id"] == "test_record"
    assert memory["build_step"] == 0
    assert memory["memory_graph"]["nodes"] == []
    assert memory["memory_graph"]["edges"] == []
    assert memory["memory_summaries"]["user_profile"] == ""

    # Should validate successfully
    validated = validate_current_memory(memory)
    assert validated.record_id == "test_record"


def test_valid_graph_summary_abstract_memory():
    """Test validation of memory with graph, summary, and abstract nodes."""
    memory = {
        "record_id": "test_record",
        "build_step": 1,
        "memory_graph": {
            "nodes": [
                {
                    "id": "mem_fact_001",
                    "type": "fact",
                    "content": "The user graduated with a degree in Business Administration.",
                    "status": "active",
                    "confidence": 0.9,
                    "abstraction_level": 1,
                    "source_session_ids": ["sess_0012"],
                    "source_event_ids": ["evt_x"]
                },
                {
                    "id": "mem_abs_001",
                    "type": "abstract",
                    "content": "The user has an education background in Business Administration.",
                    "status": "active",
                    "confidence": 0.85,
                    "abstraction_level": 2,
                    "source_session_ids": ["sess_0012"],
                    "source_event_ids": ["evt_x"]
                }
            ],
            "edges": [
                {
                    "source": "mem_abs_001",
                    "target": "mem_fact_001",
                    "type": "ABSTRACTS",
                    "confidence": 0.9
                }
            ]
        },
        "memory_summaries": {
            "user_profile": "The user has a background in Business Administration.",
            "stable_preferences": "",
            "active_plans": "",
            "recent_changes": "",
            "cross_session_abstractions": ""
        },
        "metadata": {}
    }

    validated = validate_current_memory(memory, strict=True)

    assert validated.record_id == "test_record"
    assert validated.build_step == 1
    assert len(validated.memory_graph.nodes) == 2
    assert len(validated.memory_graph.edges) == 1
    assert validated.memory_summaries.user_profile != ""


def test_reject_missing_provenance():
    """Test that active nodes without provenance are rejected in strict mode."""
    memory = {
        "record_id": "test_record",
        "build_step": 1,
        "memory_graph": {
            "nodes": [
                {
                    "id": "mem_fact_001",
                    "type": "fact",
                    "content": "Some fact without provenance.",
                    "status": "active",
                    "confidence": 0.9,
                    "abstraction_level": 1,
                    # Missing all provenance fields
                    "source_session_ids": [],
                    "source_turn_ids": [],
                    "source_event_ids": []
                }
            ],
            "edges": []
        },
        "memory_summaries": {
            "user_profile": "",
            "stable_preferences": "",
            "active_plans": "",
            "recent_changes": "",
            "cross_session_abstractions": ""
        },
        "metadata": {}
    }

    with pytest.raises(ValueError) as exc_info:
        validate_current_memory(memory, strict=True)

    assert "provenance" in str(exc_info.value).lower()


def test_reject_memory_leakage_top_level():
    """Test rejection of forbidden keys at top level."""
    memory = {
        "record_id": "test_record",
        "build_step": 0,
        "oracle_node_ids": ["fact_x"],  # Forbidden key
        "memory_graph": {"nodes": [], "edges": []},
        "memory_summaries": {
            "user_profile": "",
            "stable_preferences": "",
            "active_plans": "",
            "recent_changes": "",
            "cross_session_abstractions": ""
        },
        "metadata": {}
    }

    with pytest.raises(ValueError) as exc_info:
        assert_no_memory_leakage(memory)

    assert "oracle_node_ids" in str(exc_info.value)


def test_reject_memory_leakage_in_metadata():
    """Test rejection of forbidden keys in metadata."""
    memory = {
        "record_id": "test_record",
        "build_step": 0,
        "memory_graph": {"nodes": [], "edges": []},
        "memory_summaries": {
            "user_profile": "",
            "stable_preferences": "",
            "active_plans": "",
            "recent_changes": "",
            "cross_session_abstractions": ""
        },
        "metadata": {
            "gold_answer": "leaked answer"  # Forbidden in metadata
        }
    }

    with pytest.raises(ValueError) as exc_info:
        validate_current_memory(memory, strict=True)

    assert "gold_answer" in str(exc_info.value)


def test_target_allowed_only_for_edge_endpoints():
    """Test target is allowed for graph edges but rejected elsewhere."""
    memory = {
        "record_id": "test_record",
        "build_step": 1,
        "memory_graph": {
            "nodes": [
                {
                    "id": "mem_fact_001",
                    "type": "fact",
                    "content": "A fact.",
                    "source_session_ids": ["sess_1"],
                },
                {
                    "id": "mem_event_001",
                    "type": "event",
                    "content": "An event.",
                    "source_session_ids": ["sess_1"],
                },
            ],
            "edges": [
                {
                    "source": "mem_fact_001",
                    "target": "mem_event_001",
                    "type": "DERIVED_FROM",
                }
            ],
        },
        "memory_summaries": {
            "user_profile": "",
            "stable_preferences": "",
            "active_plans": "",
            "recent_changes": "",
            "cross_session_abstractions": "",
        },
        "metadata": {},
    }

    validate_current_memory(memory, strict=True)

    memory_with_top_target = dict(memory)
    memory_with_top_target["target"] = "leaked benchmark target"
    with pytest.raises(ValueError) as exc_info:
        validate_current_memory(memory_with_top_target, strict=True)
    assert "target" in str(exc_info.value)

    memory_with_node_target = {
        **memory,
        "memory_graph": {
            "nodes": [
                {
                    "id": "mem_fact_001",
                    "type": "fact",
                    "content": "A fact.",
                    "target": "leaked benchmark target",
                    "source_session_ids": ["sess_1"],
                }
            ],
            "edges": [],
        },
    }
    with pytest.raises(ValueError) as exc_info:
        validate_current_memory(memory_with_node_target, strict=True)
    assert "memory_graph.nodes[0].target" in str(exc_info.value)

    memory_with_metadata_target = {
        **memory,
        "metadata": {"target": "leaked benchmark target"},
    }
    with pytest.raises(ValueError) as exc_info:
        validate_current_memory(memory_with_metadata_target, strict=True)
    assert "metadata.target" in str(exc_info.value)


def test_reject_query_and_haystack_question_type_leakage():
    """Test benchmark query aliases are forbidden in current memory."""
    for forbidden_key in ["query", "haystack_question_type"]:
        memory = empty_current_memory("test_record")
        memory["metadata"][forbidden_key] = "leaked eval metadata"

        with pytest.raises(ValueError) as exc_info:
            validate_current_memory(memory, strict=True)

        assert forbidden_key in str(exc_info.value)


def test_duplicate_node_ids():
    """Test that duplicate node IDs are rejected."""
    memory = {
        "record_id": "test_record",
        "build_step": 1,
        "memory_graph": {
            "nodes": [
                {
                    "id": "mem_001",
                    "type": "fact",
                    "content": "First fact.",
                    "source_session_ids": ["sess_1"]
                },
                {
                    "id": "mem_001",  # Duplicate ID
                    "type": "fact",
                    "content": "Second fact.",
                    "source_session_ids": ["sess_1"]
                }
            ],
            "edges": []
        },
        "memory_summaries": {
            "user_profile": "",
            "stable_preferences": "",
            "active_plans": "",
            "recent_changes": "",
            "cross_session_abstractions": ""
        },
        "metadata": {}
    }

    with pytest.raises(ValueError) as exc_info:
        validate_current_memory(memory, strict=True)

    assert "Duplicate node ID" in str(exc_info.value)


def test_edge_references_nonexistent_node():
    """Test that edges must reference existing nodes."""
    memory = {
        "record_id": "test_record",
        "build_step": 1,
        "memory_graph": {
            "nodes": [
                {
                    "id": "mem_001",
                    "type": "fact",
                    "content": "A fact.",
                    "source_session_ids": ["sess_1"]
                }
            ],
            "edges": [
                {
                    "source": "mem_001",
                    "target": "mem_999",  # Non-existent
                    "type": "SUPPORTS"
                }
            ]
        },
        "memory_summaries": {
            "user_profile": "",
            "stable_preferences": "",
            "active_plans": "",
            "recent_changes": "",
            "cross_session_abstractions": ""
        },
        "metadata": {}
    }

    with pytest.raises(ValueError) as exc_info:
        validate_current_memory(memory, strict=True)

    assert "mem_999" in str(exc_info.value)


def test_abstract_node_abstraction_level():
    """Test that abstract nodes should have abstraction_level >= 2."""
    memory = {
        "record_id": "test_record",
        "build_step": 1,
        "memory_graph": {
            "nodes": [
                {
                    "id": "mem_abs_001",
                    "type": "abstract",
                    "content": "An abstract memory.",
                    "abstraction_level": 1,  # Should be >= 2
                    "source_session_ids": ["sess_1"]
                }
            ],
            "edges": []
        },
        "memory_summaries": {
            "user_profile": "",
            "stable_preferences": "",
            "active_plans": "",
            "recent_changes": "",
            "cross_session_abstractions": ""
        },
        "metadata": {}
    }

    with pytest.raises(ValueError) as exc_info:
        validate_current_memory(memory, strict=True)

    assert "abstraction_level" in str(exc_info.value)


def test_memory_to_dict():
    """Test conversion from CurrentMemory model to dict."""
    memory_dict = {
        "record_id": "test_record",
        "build_step": 1,
        "memory_graph": {
            "nodes": [
                {
                    "id": "mem_001",
                    "type": "fact",
                    "content": "A fact.",
                    "source_session_ids": ["sess_1"]
                }
            ],
            "edges": []
        },
        "memory_summaries": {
            "user_profile": "Test user",
            "stable_preferences": "",
            "active_plans": "",
            "recent_changes": "",
            "cross_session_abstractions": ""
        },
        "metadata": {}
    }

    validated = validate_current_memory(memory_dict, strict=True)
    result_dict = memory_to_dict(validated)

    assert result_dict["record_id"] == "test_record"
    assert result_dict["build_step"] == 1
    assert len(result_dict["memory_graph"]["nodes"]) == 1


def test_confidence_bounds():
    """Test that confidence values must be between 0 and 1."""
    # Invalid confidence > 1
    memory = {
        "record_id": "test_record",
        "build_step": 1,
        "memory_graph": {
            "nodes": [
                {
                    "id": "mem_001",
                    "type": "fact",
                    "content": "A fact.",
                    "confidence": 1.5,  # Invalid
                    "source_session_ids": ["sess_1"]
                }
            ],
            "edges": []
        },
        "memory_summaries": {
            "user_profile": "",
            "stable_preferences": "",
            "active_plans": "",
            "recent_changes": "",
            "cross_session_abstractions": ""
        },
        "metadata": {}
    }

    with pytest.raises(ValueError):
        validate_current_memory(memory, strict=True)


def test_archived_node_provenance_lenient():
    """Test that archived nodes can have lenient provenance requirements."""
    memory = {
        "record_id": "test_record",
        "build_step": 1,
        "memory_graph": {
            "nodes": [
                {
                    "id": "mem_001",
                    "type": "fact",
                    "content": "An archived fact.",
                    "status": "archived",
                    "source_session_ids": [],  # Empty is OK for archived
                    "source_turn_ids": [],
                    "source_event_ids": []
                }
            ],
            "edges": []
        },
        "memory_summaries": {
            "user_profile": "",
            "stable_preferences": "",
            "active_plans": "",
            "recent_changes": "",
            "cross_session_abstractions": ""
        },
        "metadata": {}
    }

    # Should not raise in non-strict mode
    validated = validate_current_memory(memory, strict=False)
    assert validated.memory_graph.nodes[0].status == MemoryStatus.ARCHIVED


def test_summary_node_provenance_lenient():
    """Test that summary nodes can have lenient provenance (session-only)."""
    memory = {
        "record_id": "test_record",
        "build_step": 1,
        "memory_graph": {
            "nodes": [
                {
                    "id": "mem_summary_001",
                    "type": "summary",
                    "content": "A summary node.",
                    "status": "active",
                    "source_session_ids": ["sess_1"],
                    # No source_event_ids is OK for summary
                    "source_turn_ids": [],
                    "source_event_ids": []
                }
            ],
            "edges": []
        },
        "memory_summaries": {
            "user_profile": "",
            "stable_preferences": "",
            "active_plans": "",
            "recent_changes": "",
            "cross_session_abstractions": ""
        },
        "metadata": {}
    }

    validated = validate_current_memory(memory, strict=True)
    assert validated.memory_graph.nodes[0].type == MemoryNodeType.SUMMARY


def test_all_memory_node_types():
    """Test that all memory node types are accepted."""
    for node_type in MemoryNodeType:
        memory = {
            "record_id": "test_record",
            "build_step": 1,
            "memory_graph": {
                "nodes": [
                    {
                        "id": f"mem_{node_type.value}_001",
                        "type": node_type.value,
                        "content": f"A {node_type.value} node.",
                        "source_session_ids": ["sess_1"],
                        "abstraction_level": 2 if node_type == MemoryNodeType.ABSTRACT else 1
                    }
                ],
                "edges": []
            },
            "memory_summaries": {
                "user_profile": "",
                "stable_preferences": "",
                "active_plans": "",
                "recent_changes": "",
                "cross_session_abstractions": ""
            },
            "metadata": {}
        }

        validated = validate_current_memory(memory, strict=True)
        assert validated.memory_graph.nodes[0].type == node_type


def test_all_memory_edge_types():
    """Test that all memory edge types are accepted."""
    for edge_type in MemoryEdgeType:
        memory = {
            "record_id": "test_record",
            "build_step": 1,
            "memory_graph": {
                "nodes": [
                    {
                        "id": "mem_001",
                        "type": "fact",
                        "content": "Source fact.",
                        "source_session_ids": ["sess_1"]
                    },
                    {
                        "id": "mem_002",
                        "type": "fact",
                        "content": "Target fact.",
                        "source_session_ids": ["sess_1"]
                    }
                ],
                "edges": [
                    {
                        "source": "mem_001",
                        "target": "mem_002",
                        "type": edge_type.value
                    }
                ]
            },
            "memory_summaries": {
                "user_profile": "",
                "stable_preferences": "",
                "active_plans": "",
                "recent_changes": "",
                "cross_session_abstractions": ""
            },
            "metadata": {}
        }

        validated = validate_current_memory(memory, strict=True)
        assert validated.memory_graph.edges[0].type == edge_type


def test_complex_memory_graph():
    """Test a more complex memory graph with multiple nodes and edges."""
    memory = {
        "record_id": "test_record",
        "build_step": 5,
        "memory_graph": {
            "nodes": [
                {
                    "id": "mem_fact_001",
                    "type": "fact",
                    "content": "User likes Python.",
                    "abstraction_level": 1,
                    "source_session_ids": ["sess_1"],
                    "source_event_ids": ["evt_1"]
                },
                {
                    "id": "mem_fact_002",
                    "type": "fact",
                    "content": "User dislikes Java.",
                    "abstraction_level": 1,
                    "source_session_ids": ["sess_2"],
                    "source_event_ids": ["evt_2"]
                },
                {
                    "id": "mem_abs_001",
                    "type": "abstract",
                    "content": "User has programming language preferences.",
                    "abstraction_level": 2,
                    "source_session_ids": ["sess_1", "sess_2"],
                    "source_event_ids": ["evt_1", "evt_2"]
                },
                {
                    "id": "mem_pref_001",
                    "type": "preference",
                    "content": "Prefers Python for scripting.",
                    "abstraction_level": 1,
                    "source_session_ids": ["sess_1"]
                }
            ],
            "edges": [
                {
                    "source": "mem_abs_001",
                    "target": "mem_fact_001",
                    "type": "ABSTRACTS",
                    "confidence": 0.9
                },
                {
                    "source": "mem_abs_001",
                    "target": "mem_fact_002",
                    "type": "ABSTRACTS",
                    "confidence": 0.9
                },
                {
                    "source": "mem_pref_001",
                    "target": "mem_fact_001",
                    "type": "DERIVED_FROM",
                    "confidence": 0.85
                }
            ]
        },
        "memory_summaries": {
            "user_profile": "A programmer with language preferences.",
            "stable_preferences": "Prefers Python over Java.",
            "active_plans": "",
            "recent_changes": "Recently discussed language preferences.",
            "cross_session_abstractions": "Consistent preference pattern across sessions."
        },
        "metadata": {
            "last_updated": "2024-01-01T10:00:00"
        }
    }

    validated = validate_current_memory(memory, strict=True)

    assert validated.record_id == "test_record"
    assert validated.build_step == 5
    assert len(validated.memory_graph.nodes) == 4
    assert len(validated.memory_graph.edges) == 3
    assert validated.memory_summaries.user_profile != ""
    assert validated.memory_summaries.stable_preferences != ""
