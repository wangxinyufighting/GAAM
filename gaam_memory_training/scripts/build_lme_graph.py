#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from gaam_graph.exporters import export_graph
from gaam_graph.graph_builder import HistoryGraphBuilder
from gaam_graph.llm import OpenAICompatibleLLM
from gaam_graph.lme_loader import LongMemEvalLoader
from gaam_graph.utils import append_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GAAM oracle history graphs for LongMemEval-style data.")
    parser.add_argument("--input", required=True, help="Path to LongMemEval JSON file.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--llm", dest="use_llm", action="store_true", help="Use LLM semantic extraction.")
    parser.add_argument("--no-llm", dest="use_llm", action="store_false", help="Dry-run heuristic extraction.")
    parser.set_defaults(use_llm=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    records = LongMemEvalLoader(args.input).load()
    if args.max_records:
        records = records[: args.max_records]

    llm = OpenAICompatibleLLM() if args.use_llm else None
    builder = HistoryGraphBuilder(llm=llm, use_llm=args.use_llm)

    summaries = []
    progress_enabled = sys.stderr.isatty()
    record_bar = tqdm(records, desc="Building graphs", unit="record", dynamic_ncols=True, position=0, disable=not progress_enabled)
    for rec in record_bar:
        record_bar.set_postfix(record_id=rec.record_id[:12], events=len(rec.events))
        step_total = len(rec.events) + 2
        with tqdm(total=step_total, desc=f"Case {rec.record_id[:12]}", unit="step", leave=False, dynamic_ncols=True, position=1, disable=not progress_enabled) as case_bar:
            def update_case_progress(stage: str, payload: dict) -> None:
                if stage == "event_start":
                    case_bar.set_postfix(
                        stage=f"event {payload['event_index']}/{payload['total_events']}",
                        speaker=str(payload.get("speaker") or "")[:16],
                    )
                elif stage == "event_analyzed":
                    topic = str(payload.get("topic") or "")
                    if len(topic) > 24:
                        topic = topic[:21] + "..."
                    case_bar.set_postfix(
                        stage="analyzed",
                        facts=payload.get("num_facts", 0),
                        entities=payload.get("num_entities", 0),
                        topic=topic,
                    )
                elif stage == "event_done":
                    case_bar.update(1)
                    case_bar.set_postfix(
                        stage=f"done {payload['event_index']}/{payload['total_events']}",
                        nodes=payload.get("graph_nodes", 0),
                        edges=payload.get("graph_edges", 0),
                    )
                elif stage == "abstract_start":
                    case_bar.set_postfix(stage=f"abstracts 0/{payload.get('total_topics', 0)}")
                elif stage == "abstract_topic_start":
                    case_bar.set_postfix(
                        stage=f"abstracts {payload['topic_index']}/{payload['total_topics']}",
                        facts=payload.get("num_facts", 0),
                    )
                elif stage == "abstract_topic_done":
                    case_bar.set_postfix(
                        stage=f"abstracted {payload['topic_index']}/{payload['total_topics']}",
                        proposals=payload.get("num_proposals", 0),
                    )
                elif stage == "abstract_done":
                    case_bar.update(1)
                    case_bar.set_postfix(stage=f"abstracts done ({payload.get('total_topics', 0)})")

            g = builder.build_record_graph(rec, progress_callback=update_case_progress)
            pack = builder.evidence_pack(g)
            prefix = out_dir / rec.record_id
            case_bar.set_postfix(stage="exporting")
            export_graph(g, prefix, pack)
            case_bar.update(1)
        summary = {
            "record_id": rec.record_id,
            "num_events": sum(1 for _, a in g.nodes(data=True) if a.get("type") == "event"),
            "num_facts": sum(1 for _, a in g.nodes(data=True) if a.get("type") == "fact"),
            "num_entities": sum(1 for _, a in g.nodes(data=True) if a.get("type") == "entity"),
            "num_abstracts": sum(1 for _, a in g.nodes(data=True) if a.get("type") == "abstract_memory"),
            "num_edges": g.number_of_edges(),
            "graph_json": str(prefix) + ".graph.json",
            "evidence_pack": str(prefix) + ".evidence_pack.json",
            "graphml": str(prefix) + ".graphml",
        }
        summaries.append(summary)
        append_jsonl(manifest_path, summary)

    write_json(out_dir / "summary.json", summaries)
    print(f"Wrote {len(summaries)} graphs to {out_dir}")


if __name__ == "__main__":
    main()
