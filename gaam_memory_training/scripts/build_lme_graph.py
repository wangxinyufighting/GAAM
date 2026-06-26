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
    parser.add_argument("--record_id", action="append", help="Build only specific record ID(s). Can be repeated.")
    parser.add_argument("--llm", dest="use_llm", action="store_true", help="Use LLM semantic extraction.")
    parser.add_argument("--no-llm", dest="use_llm", action="store_false", help="Dry-run heuristic extraction.")
    parser.add_argument("--event_batch_size", type=int, default=20, help="Number of events analyzed per LLM request.")
    parser.add_argument(
        "--relation_mode",
        choices=["heuristic", "llm", "off"],
        default="heuristic",
        help="Fact relation detection mode. Use 'llm' only for small/high-quality runs.",
    )
    parser.add_argument(
        "--abstract_mode",
        choices=["heuristic", "llm", "off"],
        default="heuristic",
        help="Abstract memory induction mode. Use 'llm' only for small/high-quality runs.",
    )
    parser.add_argument("--skip_existing", action="store_true", help="Skip records whose graph JSON already exists.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing manifest and graph outputs.")
    parser.set_defaults(use_llm=True)
    args = parser.parse_args()
    if args.overwrite:
        args.skip_existing = False

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    if manifest_path.exists() and args.overwrite:
        manifest_path.unlink()

    records = LongMemEvalLoader(args.input).load()
    if args.record_id:
        wanted = set(args.record_id)
        records = [rec for rec in records if rec.record_id in wanted]
    if args.max_records:
        records = records[: args.max_records]

    llm = OpenAICompatibleLLM() if args.use_llm else None
    builder = HistoryGraphBuilder(
        llm=llm,
        use_llm=args.use_llm,
        event_batch_size=args.event_batch_size,
        relation_mode=args.relation_mode,
        abstract_mode=args.abstract_mode,
    )

    summaries = []
    estimated_event_llm_calls = (
        sum((len(rec.events) + max(1, args.event_batch_size) - 1) // max(1, args.event_batch_size) for rec in records)
        if args.use_llm
        else 0
    )
    print(
        "Build config: "
        f"records={len(records)}, use_llm={args.use_llm}, event_batch_size={args.event_batch_size}, "
        f"estimated_event_llm_calls={estimated_event_llm_calls}, "
        f"relation_mode={args.relation_mode}, abstract_mode={args.abstract_mode}, "
        f"overwrite={args.overwrite}, skip_existing={args.skip_existing}"
    )
    progress_enabled = sys.stderr.isatty()
    record_bar = tqdm(records, desc="Building graphs", unit="record", dynamic_ncols=True, position=0, disable=not progress_enabled)
    for rec in record_bar:
        prefix = out_dir / rec.record_id
        graph_path = Path(str(prefix) + ".graph.json")
        if args.skip_existing and graph_path.exists():
            continue
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
            "llm_stats": g.graph.get("llm_stats", {}),
            "graph_json": str(prefix) + ".graph.json",
            "evidence_pack": str(prefix) + ".evidence_pack.json",
            "graphml": str(prefix) + ".graphml",
        }
        llm_stats = summary["llm_stats"]
        if args.use_llm and llm_stats.get("event_fallback_events"):
            print(
                "Warning: LLM event extraction fell back for "
                f"{llm_stats.get('event_fallback_events')} event(s). "
                f"batch_attempts={llm_stats.get('event_batch_attempts')}, "
                f"batch_successes={llm_stats.get('event_batch_successes')}, "
                f"batch_failures={llm_stats.get('event_batch_failures')}, "
                f"missing_analyses={llm_stats.get('event_missing_analyses')}, "
                f"last_error={llm_stats.get('last_error')}"
            )
        summaries.append(summary)
        append_jsonl(manifest_path, summary)

    write_json(out_dir / "summary.json", summaries)
    print(f"Wrote {len(summaries)} graphs to {out_dir}")


if __name__ == "__main__":
    main()
