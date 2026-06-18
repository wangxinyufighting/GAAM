from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import networkx as nx
from pydantic import ValidationError

from .llm import OpenAICompatibleLLM
from .lme_loader import LMEEvent, LMERecord
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
    def __init__(self, llm: Optional[OpenAICompatibleLLM] = None, use_llm: bool = True):
        self.llm = llm
        self.use_llm = use_llm
        self.prompt_dir = Path(__file__).parent / "prompts"

    def build_record_graph(self, record: LMERecord, progress_callback: Optional[ProgressCallback] = None) -> nx.MultiDiGraph:
        g = nx.MultiDiGraph(record_id=record.record_id)
        self._add_record_root(g, record)
        prev_event_id: Optional[str] = None
        facts_by_subject: Dict[str, List[str]] = defaultdict(list)

        self._emit_progress(progress_callback, "record_start", record_id=record.record_id, total_events=len(record.events))
        for event_index, ev in enumerate(record.events, start=1):
            self._emit_progress(
                progress_callback,
                "event_start",
                record_id=record.record_id,
                event_id=ev.event_id,
                event_index=event_index,
                total_events=len(record.events),
                speaker=ev.speaker,
            )
            self._add_session_and_event(g, record, ev, prev_event_id)
            prev_event_id = ev.event_id

            analysis = self._analyze_event(ev, record)
            self._emit_progress(
                progress_callback,
                "event_analyzed",
                record_id=record.record_id,
                event_id=ev.event_id,
                event_index=event_index,
                total_events=len(record.events),
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

            ent_map: Dict[str, str] = {}
            for ent in analysis.entities:
                eid = self._upsert_entity(g, ent)
                ent_map[ent.canonical_name] = eid
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
                total_events=len(record.events),
                graph_nodes=g.number_of_nodes(),
                graph_edges=g.number_of_edges(),
            )

        self._induce_abstract_memories(g, record, progress_callback=progress_callback)
        self._emit_progress(progress_callback, "record_done", record_id=record.record_id, total_events=len(record.events))
        return g

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

    def _heuristic_event_analysis(self, ev: LMEEvent, record: LMERecord) -> EventAnalysis:
        text = normalize_text(ev.text)
        # Dry-run fallback: preserves event evidence and creates a generic said fact.
        entity = "user" if ev.speaker.lower() in {"user", "human"} else ev.speaker or "speaker"
        fact = FactProposal(
            subject=entity,
            predicate="said",
            object=text[:160],
            text=f"{entity} said: {text[:220]}",
            fact_type="raw_event_statement",
            temporal_scope="historical",
            confidence=0.5,
        )
        topic = "general_conversation"
        return EventAnalysis(
            memory_worthy=len(text) > 0,
            event_category="raw_event",
            importance=3,
            topic=topic,
            entities=[EntityProposal(surface=entity, canonical_name=entity, entity_type="person", confidence=0.5)],
            facts=[fact],
            reason="Heuristic dry-run extraction.",
        )

    def _sanitize_event_analysis(self, a: EventAnalysis) -> EventAnalysis:
        clean_entities = []
        for e in a.entities:
            et = e.entity_type if e.entity_type in ALLOWED_ENTITY_TYPES else "other"
            clean_entities.append(e.model_copy(update={"entity_type": et, "surface": normalize_text(e.surface), "canonical_name": normalize_text(e.canonical_name)}))
        clean_facts = []
        for f in a.facts:
            pred = f.predicate if f.predicate in ALLOWED_PREDICATES else "other"
            if f.subject and f.object and f.text:
                clean_facts.append(f.model_copy(update={"predicate": pred, "text": normalize_text(f.text)}))
        return a.model_copy(update={"entities": clean_entities, "facts": clean_facts})

    def _repair_event_analysis(self, data: Dict) -> Dict:
        data.setdefault("memory_worthy", True)
        data.setdefault("event_category", "general")
        data.setdefault("importance", 3)
        data.setdefault("topic", None)
        data.setdefault("entities", [])
        data.setdefault("facts", [])
        data.setdefault("reason", "")
        return data

    def _upsert_topic(self, g: nx.MultiDiGraph, record_id: str, topic: str) -> str:
        tid = stable_id("topic", record_id, canonicalize_name(topic))
        if tid not in g:
            g.add_node(tid, type=NodeType.TOPIC.value, name=topic, description="LLM-proposed topic")
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
        if not self.use_llm:
            if new_fact.predicate == old_attrs.get("predicate") and normalize_text(new_fact.object).lower() == normalize_text(str(old_attrs.get("object", ""))).lower():
                return FactRelation(relation="duplicate", reason="Same predicate and object in heuristic mode.", confidence=0.6)
            return FactRelation(relation="unrelated", reason="Heuristic dry-run default.", confidence=0.5)
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
        if not self.use_llm:
            if len(fact_ids) >= 2:
                topic_name = g.nodes[topic_id].get("name", "topic")
                return [AbstractProposal(summary=f"This topic contains {len(fact_ids)} memory-worthy facts about {topic_name}.", scope="task_context", supporting_fact_ids=fact_ids[:5], confidence=0.5)]
            return []
        system = (self.prompt_dir / "abstract_memory.txt").read_text(encoding="utf-8")
        facts = [{"fact_id": fid, **{k: g.nodes[fid].get(k) for k in ["subject", "predicate", "object", "text", "status", "confidence"]}} for fid in fact_ids]
        user = json.dumps({"topic": g.nodes[topic_id], "facts": facts}, ensure_ascii=False, indent=2)
        data = self.llm.chat_json(system=system, user=user)
        out = []
        for item in data.get("abstract_memories", [])[:3]:
            try:
                prop = AbstractProposal.model_validate(item)
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
