# GAAM

GAAM contains the research code and design documents for Graph-based Agentic Active Memory.

## Repository Layout

```text
GAAM/
  gaam_memory_training/   Phase 1 memory-training pipeline and tests
  design_docs/            Method, milestone, and implementation design docs
  Code-A1/                Reference Code-A1 codebase
```

## Main Codebase

The active implementation now lives in:

```text
gaam_memory_training/
```

This directory contains:

- oracle graph construction and inspection;
- raw-history CurrentMemory building;
- oracle-valid question generation;
- frozen answering from CurrentMemory;
- reward and weakness-book computation;
- one-round Phase 1 training loop;
- tests and milestone completion reports.

See:

- [gaam_memory_training/README.md](gaam_memory_training/README.md)
- [gaam_memory_training/PHASE1_INTEGRATION_COMPLETE.md](gaam_memory_training/PHASE1_INTEGRATION_COMPLETE.md)

## Quick Verification

```bash
cd gaam_memory_training
python -m pytest tests -q
```

Phase 1 no-LLM integration smoke:

```bash
cd gaam_memory_training
python scripts/run_adversarial_memory_training.py \
  --input data/longmemeval/longmemeval_s_cleaned.json \
  --graph_dir outputs/no_leak_smoke_test \
  --output_dir outputs/adversarial_training/round_000 \
  --max_records 1 \
  --round_id 0 \
  --no_llm_question \
  --no_llm_answer \
  --allow_placeholder_questions \
  --overwrite
```
# GAAM
# GAAM
