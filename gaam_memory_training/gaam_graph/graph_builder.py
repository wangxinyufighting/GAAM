from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import networkx as nx
from pydantic import ValidationError

from .llm import LLMError, OpenAICompatibleLLM
from .lme_loader import LMEEvent, LMERecord
from .ontology import (
    ALLOWED_ABSTRACT_SCOPES,
    ALLOWED_EVENT_CATEGORIES,
    ALLOWED_FACT_TYPES,
    ALLOWED_TEMPORAL_SCOPES,
    ALLOWED_TOPICS,
    ONTOLOGY_VERSION,
    normalize_ontology_value,
    ontology_prompt_block,
)
from .schema import (
    ALLOWED_ENTITY_TYPES,
    ALLOWED_FACT_RELATIONS,
    ALLOWED_PREDICATES,
    AbstractProposal,
    EdgeType,
    EntityProposal,
    EventAnalysis,
    FactProposal,
    FactRelation,
    NodeType,
)
from .utils import canonicalize_name, normalize_text, stable_id


ProgressCallback = Callable[[str, Dict[str, Any]], None]


class HistoryGraphBuilder:
    def __init__(
        self,
        llm: Optional[OpenAICompatibleLLM] = None,
        use_llm: bool = True,
        *,
        event_batch_size: int = 20,
        relation_mode: str = "heuristic",
        abstract_mode: str = "heuristic",
    ):
        self.llm = llm
        self.use_llm = use_llm
        self.prompt_dir = Path(__file__).parent / "prompts"
        self.event_batch_size = max(1, int(event_batch_size))
        self.relation_mode = relation_mode
        self.abstract_mode = abstract_mode
        self.llm_stats: Dict[str, Any] = self._new_llm_stats()
        self._llm_disabled_error = ""

    def build_record_graph(self, record: LMERecord, progress_callback: Optional[ProgressCallback] = None) -> nx.MultiDiGraph:
        self.llm_stats = self._new_llm_stats()
        self._llm_disabled_error = ""
        g = nx.MultiDiGraph(record_id=record.record_id)
        self._add_record_root(g, record)
        prev_event_id: Optional[str] = None
        facts_by_subject: Dict[str, List[str]] = defaultdict(list)

        self._emit_progress(progress_callback, "record_start", record_id=record.record_id, total_events=len(record.events))
        total_events = len(record.events)
        for chunk_start in range(0, total_events, self.event_batch_size):
            chunk = record.events[chunk_start : chunk_start + self.event_batch_size]
            analyses = self._analyze_events_batch(chunk, record)

            for offset, ev in enumerate(chunk, start=1):
                event_index = chunk_start + offset
                self._emit_progress(
                    progress_callback,
                    "event_start",
                    record_id=record.record_id,
                    event_id=ev.event_id,
                    event_index=event_index,
                    total_events=total_events,
                    speaker=ev.speaker,
                )
                self._add_session_and_event(g, record, ev, prev_event_id)
                prev_event_id = ev.event_id

                analysis = analyses.get(ev.event_id) or self._heuristic_event_analysis(ev, record)
                self._add_event_analysis_to_graph(
                    g,
                    record,
                    ev,
                    analysis,
                    facts_by_subject,
                    progress_callback=progress_callback,
                    event_index=event_index,
                    total_events=total_events,
                )

        self._induce_abstract_memories(g, record, progress_callback=progress_callback)
        g.graph["llm_stats"] = dict(self.llm_stats)
        self._emit_progress(progress_callback, "record_done", record_id=record.record_id, total_events=len(record.events))
        return g

    def _new_llm_stats(self) -> Dict[str, Any]:
        return {
            "event_batch_attempts": 0,
            "event_batch_successes": 0,
            "event_batch_failures": 0,
            "event_fallback_events": 0,
            "event_missing_analyses": 0,
            "llm_disabled": False,
            "last_error": "",
        }

    def evidence_pack(self, g: nx.MultiDiGraph) -> Dict:
        nodes = []
        edges = []
        for nid, attrs in g.nodes(data=True):
            if attrs.get("type") in {NodeType.FACT.value, NodeType.EVENT.value, NodeType.ABSTRACT.value, NodeType.TOPIC.value}:
                nodes.append({"id": nid, **attrs})
        for u, v, key, attrs in g.edges(keys=True, data=True):
            if attrs.get("type") in {EdgeType.EXTRACTED_AS.value, EdgeType.SUPPORTS.value, EdgeType.SUPERSEDES.value, EdgeType.BELONGS_TO_TOPIC.value, EdgeType.NEXT.value}:
                edges.append({"source": u, "target": v, **attrs})
        return {
            "record_id": g.graph.get("record_id"),
            "nodes": nodes,
            "edges": edges,
        }

    def _add_record_root(self, g: nx.MultiDiGraph, record: LMERecord) -> None:
        root_id = stable_id("topic", record.record_id, "root")
        g.add_node(root_id, type=NodeType.TOPIC.value, name=f"record:{record.record_id}", description="Root topic for this LongMemEval record")
        g.graph["root_topic_id"] = root_id

    def _add_session_and_event(self, g: nx.MultiDiGraph, record: LMERecord, ev: LMEEvent, prev_event_id: Optional[str]) -> None:
        sid = stable_id("sess", record.record_id, ev.session_id)
        if sid not in g:
            g.add_node(sid, type=NodeType.SESSION.value, session_id=ev.session_id, timestamp=ev.timestamp or "")
        g.add_node(
            ev.event_id,
            type=NodeType.EVENT.value,
            session_id=ev.session_id,
            turn_id=ev.turn_id,
            speaker=ev.speaker,
            text=ev.text,
            timestamp=ev.timestamp or "",
        )
        self._add_edge(g, ev.event_id, sid, EdgeType.BELONGS_TO)
        if prev_event_id:
            self._add_edge(g, prev_event_id, ev.event_id, EdgeType.NEXT)

    def _analyze_event(self, ev: LMEEvent, record: LMERecord) -> EventAnalysis:
        if not self.use_llm:
            return self._heuristic_event_analysis(ev, record)
        system = (
            (self.prompt_dir / "event_analysis.txt")
            .read_text(encoding="utf-8")
            .replace("{entity_types}", json.dumps(sorted(ALLOWED_ENTITY_TYPES), ensure_ascii=False))
            .replace("{predicates}", json.dumps(sorted(ALLOWED_PREDICATES), ensure_ascii=False))
            .replace("{ontology}", ontology_prompt_block())
        )
        user = json.dumps({
            "record_id": record.record_id,
            "event": {
                "event_id": ev.event_id,
                "speaker": ev.speaker,
                "timestamp": ev.timestamp,
                "text": ev.text,
            },
        }, ensure_ascii=False, indent=2)
        data = self.llm.chat_json(system=system, user=user)
        try:
            analysis = EventAnalysis.model_validate(data)
        except ValidationError:
            analysis = EventAnalysis.model_validate(self._repair_event_analysis(data))
        return self._sanitize_event_analysis(analysis)

    def _analyze_events_batch(self, events: List[LMEEvent], record: LMERecord) -> Dict[str, EventAnalysis]:
        if not events:
            return {}
        if not self.use_llm:
            return {ev.event_id: self._heuristic_event_analysis(ev, record) for ev in events}
        if self._llm_disabled_error:
            self.llm_stats["event_fallback_events"] += len(events)
            return {ev.event_id: self._heuristic_event_analysis(ev, record) for ev in events}
        if self.event_batch_size <= 1 or len(events) == 1:
            try:
                self.llm_stats["event_batch_attempts"] += 1
                analysis = self._analyze_event(events[0], record)
                self.llm_stats["event_batch_successes"] += 1
                return {events[0].event_id: analysis}
            except Exception as exc:
                self.llm_stats["event_batch_failures"] += 1
                self.llm_stats["event_fallback_events"] += 1
                self.llm_stats["last_error"] = f"{type(exc).__name__}: {exc}"
                self._disable_llm_if_non_retryable(exc)
                return {events[0].event_id: self._heuristic_event_analysis(events[0], record)}

        system = (
            (self.prompt_dir / "event_batch_analysis.txt")
            .read_text(encoding="utf-8")
            .replace("{entity_types}", json.dumps(sorted(ALLOWED_ENTITY_TYPES), ensure_ascii=False))
            .replace("{predicates}", json.dumps(sorted(ALLOWED_PREDICATES), ensure_ascii=False))
            .replace("{ontology}", ontology_prompt_block())
        )
        user = json.dumps(
            {
                "record_id": record.record_id,
                "events": [
                    {
                        "event_id": ev.event_id,
                        "speaker": ev.speaker,
                        "timestamp": ev.timestamp,
                        "text": ev.text,
                    }
                    for ev in events
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            self.llm_stats["event_batch_attempts"] += 1
            data = self.llm.chat_json(system=system, user=user)
            self.llm_stats["event_batch_successes"] += 1
        except Exception as exc:
            self.llm_stats["event_batch_failures"] += 1
            self.llm_stats["last_error"] = f"{type(exc).__name__}: {exc}"
            if self._disable_llm_if_non_retryable(exc):
                self.llm_stats["event_fallback_events"] += len(events)
                return {ev.event_id: self._heuristic_event_analysis(ev, record) for ev in events}
            midpoint = len(events) // 2
            if midpoint <= 0:
                self.llm_stats["event_fallback_events"] += 1
                return {events[0].event_id: self._heuristic_event_analysis(events[0], record)}
            left = self._analyze_events_batch(events[:midpoint], record)
            right = self._analyze_events_batch(events[midpoint:], record)
            return {**left, **right}
        raw_items = data.get("analyses", [])
        by_id: Dict[str, EventAnalysis] = {}
        if isinstance(raw_items, list):
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                event_id = str(item.get("event_id") or "")
                if event_id not in {ev.event_id for ev in events}:
                    continue
                item = dict(item)
                item.pop("event_id", None)
                try:
                    analysis = EventAnalysis.model_validate(item)
                except ValidationError:
                    try:
                        analysis = EventAnalysis.model_validate(self._repair_event_analysis(item))
                    except ValidationError as exc:
                        self.llm_stats["last_error"] = f"{type(exc).__name__}: {exc}"
                        continue
                by_id[event_id] = self._sanitize_event_analysis(analysis)

        # Missing items fall back to a local heuristic instead of triggering more LLM calls.
        for ev in events:
            if ev.event_id not in by_id:
                self.llm_stats["event_missing_analyses"] += 1
                self.llm_stats["event_fallback_events"] += 1
                by_id[ev.event_id] = self._heuristic_event_analysis(ev, record)
        return by_id

    def _disable_llm_if_non_retryable(self, exc: Exception) -> bool:
        message = str(exc).lower()
        non_retryable = isinstance(exc, LLMError) and (
            "authentication" in message
            or "api key" in message
            or "401" in message
        )
        if non_retryable:
            self._llm_disabled_error = f"{type(exc).__name__}: {exc}"
            self.llm_stats["llm_disabled"] = True
        return non_retryable

    def _add_event_analysis_to_graph(
        self,
        g: nx.MultiDiGraph,
        record: LMERecord,
        ev: LMEEvent,
        analysis: EventAnalysis,
        facts_by_subject: Dict[str, List[str]],
        *,
        progress_callback: Optional[ProgressCallback],
        event_index: int,
        total_events: int,
    ) -> None:
        self._emit_progress(
            progress_callback,
            "event_analyzed",
            record_id=record.record_id,
            event_id=ev.event_id,
            event_index=event_index,
            total_events=total_events,
            memory_worthy=analysis.memory_worthy,
            num_entities=len(analysis.entities),
            num_facts=len(analysis.facts),
            topic=analysis.topic,
        )
        g.nodes[ev.event_id]["memory_worthy"] = analysis.memory_worthy
        g.nodes[ev.event_id]["event_category"] = analysis.event_category
        g.nodes[ev.event_id]["importance"] = analysis.importance

        topic_id = None
        if analysis.topic:
            topic_id = self._upsert_topic(g, record.record_id, analysis.topic)
            self._add_edge(g, ev.event_id, topic_id, EdgeType.BELONGS_TO_TOPIC, proposer="event_analysis")

        for ent in analysis.entities:
            eid = self._upsert_entity(g, ent)
            self._add_edge(g, ev.event_id, eid, EdgeType.MENTIONS, surface=ent.surface, confidence=ent.confidence)

        for fact in analysis.facts:
            fid = self._add_fact(g, record.record_id, ev, fact)
            self._add_edge(g, ev.event_id, fid, EdgeType.EXTRACTED_AS, confidence=fact.confidence)
            subj_id = self._upsert_entity(g, EntityProposal(surface=fact.subject, canonical_name=fact.subject, entity_type="other", confidence=fact.confidence))
            obj_id = self._upsert_entity(g, EntityProposal(surface=fact.object, canonical_name=fact.object, entity_type="other", confidence=fact.confidence))
            self._add_edge(g, fid, subj_id, EdgeType.FACT_SUBJECT)
            self._add_edge(g, fid, obj_id, EdgeType.FACT_OBJECT)
            if topic_id:
                self._add_edge(g, fid, topic_id, EdgeType.BELONGS_TO_TOPIC)
            self._detect_and_add_fact_relations(g, fid, fact, facts_by_subject[fact.subject])
            facts_by_subject[fact.subject].append(fid)
        self._emit_progress(
            progress_callback,
            "event_done",
            record_id=record.record_id,
            event_id=ev.event_id,
            event_index=event_index,
            total_events=total_events,
            graph_nodes=g.number_of_nodes(),
            graph_edges=g.number_of_edges(),
        )

    def _heuristic_event_analysis(self, ev: LMEEvent, record: LMERecord) -> EventAnalysis:
        text = normalize_text(ev.text)
        # Dry-run fallback: preserves event evidence and creates a generic said fact.
        entity = "user" if ev.speaker.lower() in {"user", "human"} else ev.speaker or "speaker"
        fact = FactProposal(
            subject=entity,
            predicate="said",
            object=text,
            text=f"{entity} said: {text}",
            fact_type="general",
            temporal_scope="historical",
            confidence=0.5,
        )
        topic = "other"
        return EventAnalysis(
            memory_worthy=len(text) > 0,
            event_category="general",
            importance=3,
            topic=topic,
            entities=[EntityProposal(surface=entity, canonical_name=entity, entity_type="person", confidence=0.5)],
            facts=[fact],
            reason="Heuristic dry-run extraction.",
        )

    def _sanitize_event_analysis(self, a: EventAnalysis) -> EventAnalysis:
        event_category = normalize_ontology_value(
            a.event_category,
            ALLOWED_EVENT_CATEGORIES,
            "other",
        )
        topic = None
        if a.topic:
            topic = normalize_ontology_value(a.topic, ALLOWED_TOPICS, "other")
        clean_entities = []
        for e in a.entities:
            et = e.entity_type if e.entity_type in ALLOWED_ENTITY_TYPES else "other"
            clean_entities.append(e.model_copy(update={"entity_type": et, "surface": normalize_text(e.surface), "canonical_name": normalize_text(e.canonical_name)}))
        clean_facts = []
        for f in a.facts:
            pred = f.predicate if f.predicate in ALLOWED_PREDICATES else "other"
            if f.subject and f.object and f.text:
                fact_type = normalize_ontology_value(f.fact_type, ALLOWED_FACT_TYPES, "other")
                temporal_scope = normalize_ontology_value(
                    f.temporal_scope,
                    ALLOWED_TEMPORAL_SCOPES,
                    "unknown",
                )
                clean_facts.append(
                    f.model_copy(
                        update={
                            "predicate": pred,
                            "text": normalize_text(f.text),
                            "fact_type": fact_type,
                            "temporal_scope": temporal_scope,
                        }
                    )
                )
        return a.model_copy(
            update={
                "event_category": event_category,
                "topic": topic,
                "entities": clean_entities,
                "facts": clean_facts,
            }
        )

    def _repair_event_analysis(self, data: Dict) -> Dict:
        data = dict(data or {})
        data.setdefault("memory_worthy", True)
        data.setdefault("event_category", "general")
        data["importance"] = self._bounded_int(data.get("importance", 3), 1, 5, 3)
        data.setdefault("topic", None)
        data["entities"] = self._repair_entities(data.get("entities", []))
        data["facts"] = self._repair_facts(data.get("facts", []))
        data.setdefault("reason", "")
        return data

    def _repair_entities(self, entities: Any) -> List[Dict[str, Any]]:
        if not isinstance(entities, list):
            return []
        repaired = []
        for ent in entities:
            if not isinstance(ent, dict):
                continue
            item = dict(ent)
            surface = normalize_text(str(item.get("surface") or item.get("canonical_name") or ""))
            canonical_name = normalize_text(str(item.get("canonical_name") or surface))
            if not surface or not canonical_name:
                continue
            item["surface"] = surface
            item["canonical_name"] = canonical_name
            item["entity_type"] = str(item.get("entity_type") or "other")
            item["confidence"] = self._bounded_float(item.get("confidence", 0.8), 0.0, 1.0, 0.8)
            repaired.append(item)
        return repaired

    def _repair_facts(self, facts: Any) -> List[Dict[str, Any]]:
        if not isinstance(facts, list):
            return []
        repaired = []
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            item = dict(fact)
            subject = normalize_text(str(item.get("subject") or ""))
            predicate = normalize_text(str(item.get("predicate") or "other"))
            obj = normalize_text(str(item.get("object") or ""))
            text = normalize_text(str(item.get("text") or ""))
            if not text and subject and predicate and obj:
                text = f"{subject} {predicate} {obj}."
            if not subject or not obj or not text:
                continue
            item["subject"] = subject
            item["predicate"] = predicate
            item["object"] = obj
            item["text"] = text
            item["fact_type"] = str(item.get("fact_type") or "general")
            item["temporal_scope"] = str(item.get("temporal_scope") or "unknown")
            item["confidence"] = self._bounded_float(item.get("confidence", 0.8), 0.0, 1.0, 0.8)
            repaired.append(item)
        return repaired

    @staticmethod
    def _bounded_int(value: Any, low: int, high: int, default: int) -> int:
        try:
            return max(low, min(high, int(value)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _bounded_float(value: Any, low: float, high: float, default: float) -> float:
        try:
            return max(low, min(high, float(value)))
        except (TypeError, ValueError):
            return default

    def _upsert_topic(self, g: nx.MultiDiGraph, record_id: str, topic: str) -> str:
        tid = stable_id("topic", record_id, canonicalize_name(topic))
        if tid not in g:
            g.add_node(
                tid,
                type=NodeType.TOPIC.value,
                name=topic,
                description="Ontology-controlled topic",
                ontology_version=ONTOLOGY_VERSION,
            )
        return tid

    def _upsert_entity(self, g: nx.MultiDiGraph, ent: EntityProposal) -> str:
        canonical = normalize_text(ent.canonical_name or ent.surface)
        eid = stable_id("ent", canonicalize_name(canonical))
        if eid not in g:
            g.add_node(eid, type=NodeType.ENTITY.value, name=canonical, entity_type=ent.entity_type, aliases=[ent.surface], confidence=ent.confidence)
        else:
            aliases = set(g.nodes[eid].get("aliases", []))
            aliases.add(ent.surface)
            g.nodes[eid]["aliases"] = sorted(aliases)
            g.nodes[eid]["confidence"] = max(float(g.nodes[eid].get("confidence", 0)), ent.confidence)
        return eid

    def _add_fact(self, g: nx.MultiDiGraph, record_id: str, ev: LMEEvent, fact: FactProposal) -> str:
        fid = stable_id("fact", record_id, ev.event_id, fact.subject, fact.predicate, fact.object, fact.text)
        g.add_node(
            fid,
            type=NodeType.FACT.value,
            subject=fact.subject,
            predicate=fact.predicate,
            object=fact.object,
            text=fact.text,
            fact_type=fact.fact_type,
            temporal_scope=fact.temporal_scope,
            status="active",
            confidence=fact.confidence,
            source_event_id=ev.event_id,
            timestamp=ev.timestamp or "",
        )
        return fid

    def _detect_and_add_fact_relations(self, g: nx.MultiDiGraph, new_fid: str, new_fact: FactProposal, candidate_fids: List[str]) -> None:
        # Candidate pruning: same subject, recent last 8 facts.
        for old_fid in candidate_fids[-8:]:
            old = g.nodes[old_fid]
            if not self._possibly_related(new_fact, old):
                continue
            rel = self._judge_fact_relation(new_fact, old)
            if rel.relation == "unrelated":
                continue
            edge_type = {
                "duplicate": EdgeType.DUPLICATE_OF,
                "complement": EdgeType.COMPLEMENTS,
                "refine": EdgeType.COMPLEMENTS,
                "contradict": EdgeType.CONTRADICTS,
                "supersede": EdgeType.SUPERSEDES,
            }[rel.relation]
            self._add_edge(g, new_fid, old_fid, edge_type, reason=rel.reason, confidence=rel.confidence)
            if rel.relation == "supersede":
                g.nodes[old_fid]["status"] = "stale"

    def _possibly_related(self, new_fact: FactProposal, old_attrs: Dict) -> bool:
        return (
            canonicalize_name(new_fact.subject) == canonicalize_name(str(old_attrs.get("subject", "")))
            or canonicalize_name(new_fact.object) == canonicalize_name(str(old_attrs.get("object", "")))
            or new_fact.predicate == old_attrs.get("predicate")
        )

    def _judge_fact_relation(self, new_fact: FactProposal, old_attrs: Dict) -> FactRelation:
        if not self.use_llm or self.relation_mode != "llm":
            return self._heuristic_fact_relation(new_fact, old_attrs)
        system = (self.prompt_dir / "fact_relation.txt").read_text(encoding="utf-8")
        user = json.dumps({
            "new_fact": new_fact.model_dump(),
            "old_fact": {k: old_attrs.get(k) for k in ["subject", "predicate", "object", "text", "temporal_scope", "status", "timestamp"]},
        }, ensure_ascii=False, indent=2)
        data = self.llm.chat_json(system=system, user=user)
        try:
            rel = FactRelation.model_validate(data)
        except ValidationError:
            rel = FactRelation(relation="unrelated", reason="Invalid LLM relation output.", confidence=0.0)
        if rel.relation not in ALLOWED_FACT_RELATIONS:
            rel = rel.model_copy(update={"relation": "unrelated"})
        return rel

    def _heuristic_fact_relation(self, new_fact: FactProposal, old_attrs: Dict) -> FactRelation:
        if self.relation_mode == "off":
            return FactRelation(relation="unrelated", reason="Relation detection disabled.", confidence=0.0)
        same_subject = canonicalize_name(new_fact.subject) == canonicalize_name(str(old_attrs.get("subject", "")))
        same_predicate = new_fact.predicate == old_attrs.get("predicate")
        same_object = normalize_text(new_fact.object).lower() == normalize_text(str(old_attrs.get("object", ""))).lower()
        if same_subject and same_predicate and same_object:
            return FactRelation(relation="duplicate", reason="Same subject, predicate, and object.", confidence=0.75)
        old_type = str(old_attrs.get("fact_type", ""))
        update_like = {
            "update",
            "correction",
        }
        new_text = normalize_text(new_fact.text).lower()
        object_text = normalize_text(new_fact.object).lower()
        explicit_replacement = any(
            marker in f" {new_text} {object_text} "
            for marker in [
                " no longer ",
                " instead ",
                " changed ",
                " updated ",
                " replaced ",
                " now ",
            ]
        )
        if (
            same_subject
            and same_predicate
            and (
                new_fact.predicate == "updates_previous_choice"
                or new_fact.fact_type in update_like
                or old_type in update_like
            )
            and (explicit_replacement or new_fact.fact_type in update_like or old_type in update_like)
        ):
            return FactRelation(relation="supersede", reason="Explicit update/correction evidence.", confidence=0.65)
        if same_subject and same_predicate:
            return FactRelation(relation="unrelated", reason="Same subject and predicate without replacement evidence.", confidence=0.5)
        return FactRelation(relation="unrelated", reason="No high-confidence heuristic relation.", confidence=0.5)

    def _induce_abstract_memories(self, g: nx.MultiDiGraph, record: LMERecord, progress_callback: Optional[ProgressCallback] = None) -> None:
        topic_to_facts: Dict[str, List[str]] = defaultdict(list)
        for u, v, attrs in g.edges(data=True):
            if attrs.get("type") == EdgeType.BELONGS_TO_TOPIC.value and g.nodes[u].get("type") == NodeType.FACT.value:
                topic_to_facts[v].append(u)
        abstract_topics = {topic_id: fact_ids for topic_id, fact_ids in topic_to_facts.items() if len(fact_ids) >= 2}
        self._emit_progress(progress_callback, "abstract_start", record_id=record.record_id, total_topics=len(abstract_topics))
        for topic_index, (topic_id, fact_ids) in enumerate(abstract_topics.items(), start=1):
            self._emit_progress(
                progress_callback,
                "abstract_topic_start",
                record_id=record.record_id,
                topic_id=topic_id,
                topic_index=topic_index,
                total_topics=len(abstract_topics),
                num_facts=len(fact_ids),
            )
            proposals = self._propose_abstracts(g, topic_id, fact_ids[:50])
            for prop in proposals:
                aid = stable_id("abs", record.record_id, topic_id, prop.summary)
                g.add_node(aid, type=NodeType.ABSTRACT.value, summary=prop.summary, scope=prop.scope, confidence=prop.confidence, status="active")
                self._add_edge(g, aid, topic_id, EdgeType.ABSTRACTS)
                for fid in prop.supporting_fact_ids:
                    if fid in g:
                        self._add_edge(g, fid, aid, EdgeType.SUPPORTS, confidence=prop.confidence)
            self._emit_progress(
                progress_callback,
                "abstract_topic_done",
                record_id=record.record_id,
                topic_id=topic_id,
                topic_index=topic_index,
                total_topics=len(abstract_topics),
                num_proposals=len(proposals),
            )
        self._emit_progress(progress_callback, "abstract_done", record_id=record.record_id, total_topics=len(abstract_topics))

    def _propose_abstracts(self, g: nx.MultiDiGraph, topic_id: str, fact_ids: List[str]) -> List[AbstractProposal]:
        if self.abstract_mode == "off":
            return []
        if not self.use_llm or self.abstract_mode != "llm":
            if len(fact_ids) >= 2:
                topic_name = g.nodes[topic_id].get("name", "topic")
                return [AbstractProposal(summary=f"This topic contains {len(fact_ids)} memory-worthy facts about {topic_name}.", scope="task_context", supporting_fact_ids=fact_ids[:5], confidence=0.5)]
            return []
        system = (
            (self.prompt_dir / "abstract_memory.txt")
            .read_text(encoding="utf-8")
            .replace("{ontology}", ontology_prompt_block())
        )
        facts = [{"fact_id": fid, **{k: g.nodes[fid].get(k) for k in ["subject", "predicate", "object", "text", "status", "confidence"]}} for fid in fact_ids]
        user = json.dumps({"topic": g.nodes[topic_id], "facts": facts}, ensure_ascii=False, indent=2)
        data = self.llm.chat_json(system=system, user=user)
        out = []
        for item in data.get("abstract_memories", [])[:3]:
            try:
                prop = AbstractProposal.model_validate(item)
                prop.scope = normalize_ontology_value(
                    prop.scope,
                    ALLOWED_ABSTRACT_SCOPES,
                    "general",
                )
                prop.supporting_fact_ids = [fid for fid in prop.supporting_fact_ids if fid in fact_ids]
                if prop.supporting_fact_ids:
                    out.append(prop)
            except ValidationError:
                continue
        return out

    def _add_edge(self, g: nx.MultiDiGraph, source: str, target: str, edge_type: EdgeType, **attrs) -> None:
        g.add_edge(source, target, type=edge_type.value, **attrs)

    def _emit_progress(self, callback: Optional[ProgressCallback], stage: str, **payload: Any) -> None:
        if callback is not None:
            callback(stage, payload)
