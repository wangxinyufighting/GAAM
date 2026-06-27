# Phase 4 Benchmark Suite Troubleshooting Guide

## Problem: "No successful checkpoints found in trainer step"

This error occurs when the GRPO trainer step fails to produce any successful checkpoints during training. The trainer step runs for each actor (Memory Builder and Question Agent) and must succeed for at least one actor.

## Diagnostic Steps

### 1. Check the trainer step logs

On your remote server, navigate to the trainer step output directory:

```bash
cd outputs/phase4_benchmark_suites/full_verl/runs/seed_1029/round_000/trainer_step
```

Look for these files:
- `trainer_step_manifest.json` - Overall trainer status
- `memory_builder/trainer_report.json` - Memory Builder actor report
- `question_agent/trainer_report.json` - Question Agent actor report

Check the status and errors:

```bash
# Check overall status
jq '.status, .actor_reports[].status, .actor_reports[].errors' trainer_step_manifest.json

# Check individual actor reports
jq '.status, .errors, .warnings' memory_builder/trainer_report.json
jq '.status, .errors, .warnings' question_agent/trainer_report.json
```

### 2. Common Root Causes

#### A. Model Loading Failure

**Symptom**: Error like `FileNotFoundError`, `OSError: Unable to load weights`

**Fix**: Verify model paths exist and are accessible:

```bash
ls -lh /mnt/local2/wxy/models/Qwen3-0.6B/
# Should contain: config.json, pytorch_model.bin (or model.safetensors), tokenizer files
```

#### B. CUDA/GPU Issues

**Symptom**: Error like `CUDA out of memory`, `RuntimeError: No CUDA GPUs are available`

**Fix 1 - Check GPU availability**:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

**Fix 2 - Reduce memory usage** by adjusting batch size in your suite config:
```json
{
  "trainer_config": {
    "mini_batch_size": 1,  // reduce from default
    "gradient_accumulation_steps": 4  // increase to compensate
  }
}
```

**Fix 3 - Set CUDA device** explicitly:
```bash
export CUDA_VISIBLE_DEVICES=0  # or 1,2,3 for specific GPUs
```

#### C. LocalHFPolicyClient Import/Initialization Failure

**Symptom**: Error like `ImportError`, `ModuleNotFoundError`, `AttributeError`

**Fix**: Verify dependencies are installed:
```bash
cd gaam_memory_training
python -c "from gaam_graph.local_policy_clients import LocalHFPolicyClient"
python -c "import transformers; print(transformers.__version__)"
python -c "import torch; print(torch.__version__)"
```

If import fails, reinstall:
```bash
pip install -r requirements.txt
pip install -r requirements-local-training.txt
```

#### D. VERL Handoff Failure

**Symptom**: Error in `_write_phase4_verl_handoff` function

**Fix**: Check VERL dependencies:
```bash
python -c "import verl; print(verl.__file__)"
python -c "from gaam_graph.verl_trainer_adapter import build_trainer_ready_batch"
```

#### E. Batch Data Issues

**Symptom**: Error like `ValueError: Full checkpoint update received zero items`

**Fix**: Check if rollout produced valid training data:
```bash
# Check train rollout produced data
ls -lh outputs/phase4_benchmark_suites/full_verl/runs/seed_1029/round_000/train_rollout/

# Check batch files
jq '.status, .succeeded_record_ids, .metrics' outputs/phase4_benchmark_suites/full_verl/runs/seed_1029/round_000/train_rollout/multi_case_rollout_manifest.json
```

### 3. Debug with Simpler Backend

If VERL backend is failing, try with dry_run first to isolate the issue:

```bash
# Test with dry_run backend (no actual model updates)
MEMORY_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
QUESTION_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
ANSWERER_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
TRAINER_BACKEND=dry_run \
SUITE_CONFIG=configs/phase4/full_verl_suite.json \
OUTPUT_DIR=outputs/phase4_benchmark_suites/full_verl_dryrun \
SEEDS="1029" \
PREPARE_FULL_SPLIT=1 \
bash scripts/run_remote_phase4_benchmark_suite.sh
```

If dry_run succeeds, the issue is specifically in the GRPO training logic.

### 4. Enable Detailed Logging

Add debugging to see where exactly the failure occurs:

```bash
export PYTHONUNBUFFERED=1
export GAAM_DEBUG=1  # if supported

# Run with verbose output
bash scripts/run_remote_phase4_benchmark_suite.sh 2>&1 | tee phase4_debug.log
```

Check the log for the first exception traceback.

### 5. Test Trainer Step in Isolation

Run just the trainer step manually to get full error output:

```bash
cd gaam_memory_training

python scripts/run_phase4_grpo_trainer_step.py \
  --multi_case_run_dir outputs/phase4_benchmark_suites/full_verl/runs/seed_1029/round_000/train_rollout \
  --output_dir outputs/phase4_test_trainer \
  --backend verl \
  --round_id 0 \
  --step_id 0 \
  --memory_builder_model_path /mnt/local2/wxy/models/Qwen3-0.6B \
  --question_agent_model_path /mnt/local2/wxy/models/Qwen3-0.6B \
  --actors memory_builder question_agent \
  --overwrite
```

This will show the full Python traceback.

### 6. Verify Model is Compatible

Test that your model can be loaded and run:

```bash
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "/mnt/local2/wxy/models/Qwen3-0.6B"
print(f"Loading model from {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print(f"Model loaded: {model.__class__.__name__}")
print(f"Model device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")
print("Model loading test: PASSED")
PY
```

## Expected Successful Output

When the trainer step succeeds, you should see:

1. Trainer step manifest with `status: "succeeded"` or `"partial"`
2. Actor reports with at least one `status: "succeeded"`
3. Written checkpoints in `memory_builder/checkpoint/` and/or `question_agent/checkpoint/`
4. Checkpoint metadata files with `checkpoint_id` fields

Example successful status:
```bash
$ jq '.status, .actor_reports[].status' trainer_step_manifest.json
"succeeded"
"succeeded"
"succeeded"
```

## Quick Workaround

If you just want to test the full pipeline flow without actual model training, use `dry_run` backend:

```bash
TRAINER_BACKEND=dry_run bash scripts/run_remote_phase4_benchmark_suite.sh
```

This validates all data flow and file I/O without running expensive GPU training.

## Still Stuck?

1. Share the full error from `trainer_step_manifest.json` or actor reports
2. Share output of `nvidia-smi` and Python/CUDA check from the script
3. Share the first Python exception traceback from the logs
4. Check disk space: `df -h` (OOM can also be disk space exhaustion when saving checkpoints)
