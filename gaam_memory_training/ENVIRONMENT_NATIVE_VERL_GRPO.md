# GAAM Native Code-A1/VERL GRPO Environment

This file describes the recommended remote GPU environment for:

- GAAM Phase 4 native VERL GRPO training
- `GAAMDualRayTrainer`
- `scripts/run_native_verl_grpo_training.sh`
- `scripts/run_native_verl_dual_cotraining.sh`
- Full-parameter training, not LoRA

The target server used during development was an NVIDIA RTX A6000 machine with
CUDA available through PyTorch.

---

## 1. Required System Environment

Recommended:

| Component | Requirement |
|---|---|
| OS | Linux |
| GPU | NVIDIA Ampere or newer, e.g. RTX A6000 / A100 / H100 |
| GPU memory | 24 GB minimum for Qwen3-0.6B smoke runs; 48 GB+ recommended |
| NVIDIA driver | Must support CUDA 12.4 or the CUDA build you install |
| CUDA runtime | Use the runtime bundled with PyTorch wheels unless your cluster requires module CUDA |
| Python | 3.10 or 3.11 |
| PyTorch | `2.6.0+cu124` recommended for the current server |
| Training backend | Code-A1 vendored VERL under `Code-A1/Code-A1/verl` |

Check the driver first:

```bash
nvidia-smi
```

For PyTorch `cu124`, the driver must be new enough for CUDA 12.4. If your driver
is older, install the PyTorch build matching your driver-supported CUDA version.

---

## 2. Recommended Conda Environment

Create a clean environment:

```bash
conda create -n gaam-verl python=3.11 -y
conda activate gaam-verl
python -m pip install --upgrade pip setuptools wheel packaging ninja
```

Use Python 3.11 if possible. Python 3.10 is also acceptable.

GAAM also provides a sourceable production environment template:

```bash
cd /mnt/local2/wxy/GAAM/gaam_memory_training
cp production.env.example production.env
# edit production.env and fill API keys / paths
set -a
source production.env
set +a
bash scripts/run_production_e2e_training_and_evaluation.sh
```

On a remote GPU server, the preferred launcher is:

```bash
bash scripts/run_remote_production_training.sh
```

It loads `production.env`, writes remote diagnostics first, and then launches
the production E2E script. Use the lower-level E2E script directly only when you
already sourced the environment yourself.

For a fresh remote environment, install dependencies through the dedicated
installer first:

```bash
bash scripts/install_remote_production_deps.sh
```

Or let the launcher run it once before diagnostics:

```bash
INSTALL_DEPS_BEFORE_TRAINING=1 bash scripts/run_remote_production_training.sh
```

Keep `INSTALL_DEPS_BEFORE_TRAINING=0` after the environment is prepared. The
installer intentionally keeps `INSTALL_FLASH_ATTN=0` by default because
flash-attn must match the active PyTorch/CUDA/C++ ABI; install it only after
the PyTorch check in this document is clean.

Do not commit `production.env`; keep real API keys in your server environment or
secret manager.

---

## 3. Install PyTorch First

Install PyTorch before installing `flash-attn` or VERL dependencies.

Recommended for your current A6000 server:

```bash
python -m pip install \
  torch==2.6.0 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

Expected example:

```text
torch: 2.6.0+cu124
cuda: 12.4
cuda_available: True
device_count: >=1
device: NVIDIA RTX A6000
```

---

## 4. Install GAAM Runtime Dependencies

From the GAAM training directory:

```bash
cd /mnt/local2/wxy/GAAM/gaam_memory_training
python -m pip install -r requirements.txt
```

This installs the GAAM-side dependencies:

- `networkx`
- `pydantic`
- `python-dotenv`
- `openai`
- `tqdm`
- `orjson`
- `PyYAML`
- `pandas`
- `pyarrow`

---

## 5. Install Code-A1 / VERL Dependencies

Code-A1 vendored VERL requirements live at:

```text
/mnt/local2/wxy/GAAM/Code-A1/Code-A1/verl/requirements.txt
```

Install all VERL dependencies except `flash-attn` first:

```bash
cd /mnt/local2/wxy/GAAM

python - <<'PY'
from pathlib import Path

src = Path("Code-A1/Code-A1/verl/requirements.txt")
dst = Path("/tmp/verl.requirements.no_flash_attn.txt")

kept = []
for line in src.read_text(encoding="utf-8").splitlines():
    normalized = line.strip().split("#", 1)[0].strip().lower().replace("_", "-")
    if normalized == "flash-attn" or normalized.startswith("flash-attn"):
        continue
    kept.append(line)

dst.write_text("\n".join(kept) + "\n", encoding="utf-8")
print(dst)
PY

python -m pip install -r /tmp/verl.requirements.no_flash_attn.txt
```

Recommended explicit packages if the unpinned VERL install gives conflicts:

```bash
python -m pip install \
  accelerate \
  codetiming \
  datasets \
  dill \
  hydra-core \
  liger-kernel \
  numpy \
  pandas \
  "pyarrow>=19.0.0" \
  pybind11 \
  pylatexenc \
  "ray[default]" \
  "tensordict<=0.6.2" \
  torchdata \
  "transformers>=4.51.0,<5.0" \
  wandb \
  uvicorn \
  fastapi \
  packaging
```

---

## 6. Install vLLM

For native VERL rollout, install vLLM. Code-A1's requirements mention vLLM
`0.8.4`.

Recommended:

```bash
python -m pip install vllm==0.8.4
```

If pip prints dependency-conflict warnings like `opentelemetry-sdk` or
`protobuf`, the vLLM install may still have succeeded. For `vllm==0.8.4`,
prefer the vLLM-compatible OpenTelemetry stack:

```bash
python -m pip uninstall -y opentelemetry-exporter-prometheus google-api-core

python -m pip install --force-reinstall \
  "opentelemetry-api>=1.26.0,<1.27.0" \
  "opentelemetry-sdk>=1.26.0,<1.27.0" \
  "opentelemetry-semantic-conventions==0.47b0" \
  "opentelemetry-proto>=1.26.0,<1.27.0" \
  "opentelemetry-exporter-otlp-proto-http>=1.26.0,<1.27.0" \
  "opentelemetry-exporter-otlp-proto-grpc>=1.26.0,<1.27.0" \
  "protobuf>=3.20.3,<5.0"

python -m pip check
```

Then verify vLLM:

```bash
python - <<'PY'
import vllm
print("vllm:", vllm.__version__)
PY
```

If this conflicts with your installed PyTorch/CUDA stack, use the vLLM version
compatible with your PyTorch build, then run the verification commands below.

---

## 7. Install flash-attn Last

Install `flash-attn` only after PyTorch is installed and importable.

Recommended:

```bash
python -m pip install flash-attn --no-build-isolation
```

If build fails, first verify:

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
PY
```

The common failure:

```text
ModuleNotFoundError: No module named 'torch'
```

means `flash-attn` was installed before PyTorch was visible to the build
environment. Reinstall after PyTorch with `--no-build-isolation`.

---

## 8. Required Environment Variables

For native VERL training:

```bash
cd /mnt/local2/wxy/GAAM/gaam_memory_training

export CODE_A1_ROOT=/mnt/local2/wxy/GAAM/Code-A1/Code-A1
export VERL_ROOT=${CODE_A1_ROOT}/verl
export PYTHONPATH=/mnt/local2/wxy/GAAM/gaam_memory_training:${CODE_A1_ROOT}:${VERL_ROOT}:${PYTHONPATH}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export KMP_DUPLICATE_LIB_OK=TRUE
export KMP_INIT_AT_FORK=FALSE
```

Model paths:

```bash
export MEMORY_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B
export QUESTION_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B
export ANSWERER_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B
```

If your oracle graph directory contains only a subset of LongMemEval cases,
create the split from existing graphs instead of splitting the full raw dataset:

```bash
python scripts/create_split_from_existing_graphs.py \
  --input data/longmemeval/longmemeval_s_cleaned.json \
  --oracle_graph_dir outputs/longmemeval_s_graph \
  --output outputs/splits/longmemeval_s.existing_graphs.seed1029.json \
  --train_ratio 0.8 \
  --dev_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 1029 \
  --allow_empty_split \
  --overwrite
```

Use `--allow_empty_split` only when the number of available oracle graphs is too
small for a non-empty train/dev/test split. For real experiments, build enough
oracle graphs to keep train, dev, and test non-empty.

---

## 9. Verify Imports

Run:

```bash
cd /mnt/local2/wxy/GAAM/gaam_memory_training

python - <<'PY'
import sys
import torch
import transformers
import ray
import pandas
import pyarrow
import vllm
import verl

print("python:", sys.version)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0:", torch.cuda.get_device_name(0))
print("transformers:", transformers.__version__)
print("ray:", ray.__version__)
print("pandas:", pandas.__version__)
print("pyarrow:", pyarrow.__version__)
print("vllm:", vllm.__version__)
print("verl:", verl.__file__)
PY
```

Also verify GAAM imports:

```bash
PYTHONPATH=/mnt/local2/wxy/GAAM/gaam_memory_training python - <<'PY'
from gaam_graph.native_verl_dual_trainer import GAAMDualRayTrainer
from gaam_graph.verl_gaam_reward import compute_score
print("GAAM native VERL imports: OK")
print(compute_score("gaam_question_agent", '{"questions":[]}', {"actor_role": "question_agent"}))
PY
```

---

## 10. Verify Dataset Export Before Training

```bash
cd /mnt/local2/wxy/GAAM/gaam_memory_training

python scripts/export_native_verl_grpo_dataset.py \
  --input data/longmemeval/longmemeval_s_cleaned.json \
  --oracle_graph_dir outputs/longmemeval_s_graph \
  --output_dir outputs/native_verl_grpo/datasets/memory_builder \
  --actor_role memory_builder \
  --split_manifest outputs/splits/longmemeval_s.phase4_split.seed0.json \
  --memory_input_mode incremental \
  --memory_session_chunk_size 4 \
  --max_memory_chunk_chars 12000 \
  --max_previous_memory_chars 6000 \
  --allow_static_incremental_scaffold \
  --max_records_per_split 1
```

Expected output:

```text
== Native VERL GRPO dataset exported ==
actor_role: memory_builder
train_path: ...
val_path: ...
num_train_rows: >=1
num_val_rows: >=1
```

---

## 11. Smoke Test Native Dual Training

First run a dry-run scheduler test:

```bash
cd /mnt/local2/wxy/GAAM/gaam_memory_training

DUAL_DRY_RUN=1 \
MEMORY_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
QUESTION_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
SPLIT_MANIFEST=outputs/splits/longmemeval_s.phase4_split.seed0.json \
bash scripts/run_native_verl_dual_cotraining.sh
```

Then run a real one-step full-parameter GRPO smoke test:

```bash
cd /mnt/local2/wxy/GAAM/gaam_memory_training

MEMORY_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
QUESTION_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
SPLIT_MANIFEST=outputs/splits/longmemeval_s.phase4_split.seed0.json \
MEMORY_NUM_GPUS=2 \
QUESTION_NUM_GPUS=2 \
ROUNDS=1 \
QUESTIONS_PER_CASE=8 \
GAAM_REWARD_JUDGE_ENABLED=1 \
GAAM_REWARD_JUDGE_BASE_URL=https://api.deepseek.com \
GAAM_REWARD_JUDGE_API_KEY="${DEEPSEEK_API_KEY}" \
GAAM_REWARD_JUDGE_MODEL=deepseek-v4-flash \
TOTAL_TRAINING_STEPS_PER_ACTOR_STEP=1 \
SAVE_FREQ=1 \
SAVE_HF_MODEL=1 \
REQUIRE_HF_CHECKPOINT=1 \
bash scripts/run_native_verl_dual_cotraining.sh
```

For production runs, the simplest end-to-end entrypoint is:

```bash
cd /mnt/local2/wxy/GAAM/gaam_memory_training

MEMORY_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
QUESTION_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
AUTO_CREATE_SPLIT=1 \
SPLIT_SEED=1029 \
ALLOW_EMPTY_SPLIT=0 \
TRAINING_OUTPUT_DIR=outputs/production_native_verl_dual_cotraining \
EVAL_OUTPUT_DIR=outputs/production_post_training_case_evaluation \
MEMORY_NUM_GPUS=2 \
QUESTION_NUM_GPUS=2 \
ROUNDS=1 \
QUESTIONS_PER_CASE=8 \
GAAM_REWARD_JUDGE_ENABLED=1 \
GAAM_REWARD_JUDGE_BASE_URL=https://api.deepseek.com \
GAAM_REWARD_JUDGE_API_KEY="${DEEPSEEK_API_KEY}" \
GAAM_REWARD_JUDGE_MODEL=deepseek-v4-flash \
ANSWER_BACKEND=api \
ANSWER_API_KEY="${DEEPSEEK_API_KEY}" \
JUDGE_BACKEND=api \
JUDGE_API_KEY="${DEEPSEEK_API_KEY}" \
SAVE_HF_MODEL=1 \
REQUIRE_HF_CHECKPOINT=1 \
VERIFY_TRAINING_OUTPUT=1 \
VERIFY_EVALUATION_OUTPUT=1 \
bash scripts/run_production_e2e_training_and_evaluation.sh
```

This script can create a graph-backed split, run native dual GRPO training,
verify full Hugging Face checkpoints, run post-training evaluation, and verify
evaluation artifacts. By default, post-training evaluation uses
`EVALUATION_SPLIT=test` from the generated split manifest. Use
`EVALUATION_SPLIT=dev` for model-selection checks, `EVALUATION_SPLIT=train` for
debugging, or `EVALUATION_SPLIT=all` for all split-manifest records.

Before the expensive native VERL job starts, the E2E script runs a production
preflight by default:

```bash
RUN_PREFLIGHT=1
REQUIRE_CUDA=1
CHECK_IMPORTS=1
REQUIRE_VERL_IMPORT=1
```

The preflight checks raw data, graph-backed split records, missing oracle
graphs, train/test split availability, Memory Builder and Question Agent model
directories, Code-A1/VERL files, Python package imports, CUDA device count, and
required API keys for reward/evaluation backends. For a cheap environment-only
check, run it directly:

```bash
python scripts/check_production_e2e_preflight.py \
  --input data/longmemeval/longmemeval_s_cleaned.json \
  --oracle_graph_dir outputs/longmemeval_s_graph \
  --split_manifest outputs/splits/longmemeval_s.existing_graphs.seed1029.json \
  --memory_model_path /mnt/local2/wxy/models/Qwen3-0.6B \
  --question_model_path /mnt/local2/wxy/models/Qwen3-0.6B \
  --code_a1_root /mnt/local2/wxy/GAAM/Code-A1/Code-A1 \
  --training_output_dir outputs/production_native_verl_dual_cotraining \
  --eval_output_dir outputs/production_post_training_case_evaluation \
  --require_cuda 1 \
  --check_imports 1 \
  --require_verl_import 1
```

Set `REQUIRE_CUDA=0`, `CHECK_IMPORTS=0`, or `REQUIRE_VERL_IMPORT=0` only for
local shell smoke tests; keep them enabled on the remote GPU server.

For a richer remote diagnostics report before starting training, run:

```bash
python scripts/collect_remote_training_diagnostics.py \
  --output outputs/remote_training_diagnostics.json
```

This writes Python/package versions, PyTorch CUDA details, optional `nvidia-smi`
output, a redacted environment snapshot, and the full production preflight
report. It is the first command to run when a remote server behaves differently
from the local tests.

At the end, the script also writes a final audit report:

```text
outputs/production_post_training_case_evaluation/production_run_audit.json
```

The audit passes only when native training verification passed, final
Memory Builder / Question Agent checkpoint paths exist, case evaluation used the
expected split, all selected records were judged, and evaluation leakage
verification passed.

The E2E script also exports a compact evidence bundle by default:

```bash
EXPORT_ARTIFACT_BUNDLE=1
ARTIFACT_BUNDLE_DIR=outputs/production_artifact_bundle
VERIFY_ARTIFACT_BUNDLE=1
VERIFY_PRODUCTION_COMPLETION=1
RUN_ID=remote_a6000_seed1029
```

The bundle intentionally does not copy large model checkpoints. It copies small
evidence artifacts only: training manifest, training verification, split
manifest, case evaluation manifest, evaluation verification, evaluation
summary, production audit, checksums, and a redacted environment snapshot. You
can export it manually after a completed run:

```bash
python scripts/export_production_artifact_bundle.py \
  --training_output_dir outputs/production_native_verl_dual_cotraining \
  --evaluation_output_dir outputs/production_post_training_case_evaluation \
  --split_manifest outputs/splits/longmemeval_s.existing_graphs.seed1029.json \
  --bundle_output_dir outputs/production_artifact_bundle \
  --run_id remote_a6000_seed1029
```

Verify the bundle:

```bash
python scripts/verify_production_artifact_bundle.py \
  --bundle_output_dir outputs/production_artifact_bundle
```

The verifier writes:

```text
outputs/production_artifact_bundle/production_artifact_bundle_verification.json
```

A remote production run is not considered fully evidenced until all of these
files report success:

```text
outputs/production_native_verl_dual_cotraining/native_verl_training_verification.json
outputs/production_post_training_case_evaluation/case_evaluation_verification.json
outputs/production_post_training_case_evaluation/production_run_audit.json
outputs/production_artifact_bundle/production_artifact_bundle_verification.json
outputs/production_post_training_case_evaluation/production_e2e_completion_verification.json
```

You can run the final completion verifier manually:

```bash
python scripts/verify_production_e2e_completion.py \
  --training_output_dir outputs/production_native_verl_dual_cotraining \
  --evaluation_output_dir outputs/production_post_training_case_evaluation \
  --artifact_bundle_dir outputs/production_artifact_bundle \
  --expected_evaluation_split test \
  --split_manifest outputs/splits/longmemeval_s.existing_graphs.seed1029.json
```

If you want to run training without evaluation, use the training wrapper below.
It performs readiness checks before training and artifact verification
afterward:

```bash
cd /mnt/local2/wxy/GAAM/gaam_memory_training

MEMORY_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
QUESTION_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
SPLIT_MANIFEST=outputs/splits/longmemeval_s.phase4_split.seed0.json \
OUTPUT_DIR=outputs/production_native_verl_dual_cotraining \
MEMORY_NUM_GPUS=2 \
QUESTION_NUM_GPUS=2 \
ROUNDS=1 \
QUESTIONS_PER_CASE=8 \
GAAM_REWARD_JUDGE_ENABLED=1 \
GAAM_REWARD_JUDGE_BASE_URL=https://api.deepseek.com \
GAAM_REWARD_JUDGE_API_KEY="${DEEPSEEK_API_KEY}" \
GAAM_REWARD_JUDGE_MODEL=deepseek-v4-flash \
SAVE_HF_MODEL=1 \
REQUIRE_HF_CHECKPOINT=1 \
VERIFY_AFTER_TRAINING=1 \
bash scripts/run_production_gaam_training.sh
```

Expected logs should include:

```text
GAAMDualRayTrainer round 1/1
question_agent: launching native VERL GRPO
== Launching native VERL GRPO ==
python -m verl.trainer.main_ppo algorithm.adv_estimator=grpo ...
memory_builder: launching native VERL GRPO
== Launching native VERL GRPO ==
python -m verl.trainer.main_ppo algorithm.adv_estimator=grpo ...
```

Do not enable `MEMORY_INPUT_MODE=incremental` in native parquet training unless
you also set `ALLOW_STATIC_INCREMENTAL_SCAFFOLD=1` for a smoke test. Static
incremental parquet rows are independent samples and are not a real stateful
memory-building rollout.

For a real stateful incremental Memory Builder rollout, use an API endpoint
(DeepSeek, OpenAI-compatible service, or a local vLLM server):

```bash
cd /mnt/local2/wxy/GAAM/gaam_memory_training

GAAM_MEMORY_BUILDER_BASE_URL=https://api.deepseek.com \
GAAM_MEMORY_BUILDER_API_KEY="${DEEPSEEK_API_KEY}" \
GAAM_MEMORY_BUILDER_MODEL=deepseek-v4-flash \
python scripts/run_stateful_incremental_memory_builder.py \
  --input data/longmemeval/longmemeval_s_cleaned.json \
  --output_dir outputs/stateful_incremental_memory \
  --record_id e47becba \
  --session_chunk_size 4
```

LLM-as-judge reward is configured by environment variables:

```bash
GAAM_REWARD_JUDGE_ENABLED=1
GAAM_REWARD_JUDGE_BASE_URL=https://api.deepseek.com
GAAM_REWARD_JUDGE_API_KEY="${DEEPSEEK_API_KEY}"
GAAM_REWARD_JUDGE_MODEL=deepseek-v4-flash
GAAM_REWARD_JUDGE_REQUIRED=0
GAAM_REWARD_JUDGE_TIMEOUT=60
GAAM_REWARD_JUDGE_MAX_INPUT_CHARS=20000
```

The reward judge uses an OpenAI-compatible chat-completions API. Keep the API
key out of committed files and pass it through `GAAM_REWARD_JUDGE_API_KEY` or a
secret manager on the remote server. Reward details record the configured
`base_url`, `model`, and whether an API key was present, but never write the raw
API key.

The offline/worker `RewardManager` path also supports API judging for answer
correctness and Question Agent diagnostic value. It uses
`GAAM_REWARD_MANAGER_JUDGE_BASE_URL`, `GAAM_REWARD_MANAGER_JUDGE_API_KEY`,
`GAAM_REWARD_MANAGER_JUDGE_MODEL`, `GAAM_REWARD_MANAGER_JUDGE_TIMEOUT`, and
`GAAM_REWARD_MANAGER_JUDGE_MAX_CHARS` when set; otherwise it falls back to the
`GAAM_REWARD_JUDGE_*` variables above.

P0/P1 reward safety checks are enforced before any LLM judge call:

- empty actor output receives `score=0`;
- forbidden leakage fields such as `target_question`, `benchmark_question`,
  `gold_answer`, and `oracle_answer` receive `score=0`;
- Memory Builder outputs additionally forbid generic `question` and `answer`
  keys, because Current Memory must not store benchmark target fields;
- Question Agent outputs may contain ordinary generated `question` fields, but
  not target/gold/oracle answer fields.

Outputs:

```text
outputs/native_verl_dual_cotraining/
  dual_cotraining_manifest.json
  native_verl_training_verification.json
  logs/
  rounds/round_000/question_agent/
  rounds/round_000/memory_builder/
```

To verify an existing native dual training output directory:

```bash
python scripts/verify_native_verl_training_output.py \
  --output_dir outputs/production_native_verl_dual_cotraining
```

The verification passes only when the manifest succeeded, every actor step
launched native GRPO, every actor step produced a recognizable Hugging Face
checkpoint, and the final model paths exist.

Resolve final checkpoint paths for downstream commands:

```bash
python scripts/resolve_native_verl_checkpoints.py \
  --output_dir outputs/production_native_verl_dual_cotraining \
  --format json
```

Or export shell variables directly:

```bash
eval "$(
  python scripts/resolve_native_verl_checkpoints.py \
    --output_dir outputs/production_native_verl_dual_cotraining \
    --format shell
)"
```

---

## 12. Evaluate Trained Checkpoints

After verification passes, use the produced Hugging Face checkpoint as the
Memory Builder backend for case-level evaluation:

```bash
cd /mnt/local2/wxy/GAAM/gaam_memory_training

MEMORY_BACKEND=stateful_local_hf \
MEMORY_MODEL="${MEMORY_MODEL}" \
MEMORY_DEVICE_MAP=auto \
MEMORY_TORCH_DTYPE=auto \
ANSWER_BACKEND=api \
ANSWER_MODEL=deepseek-v4-flash \
ANSWER_API_KEY="${DEEPSEEK_API_KEY}" \
JUDGE_BACKEND=api \
JUDGE_MODEL=deepseek-v4-flash \
JUDGE_API_KEY="${DEEPSEEK_API_KEY}" \
RECORD_ID=e47becba \
bash scripts/run_production_case_evaluation.sh
```

If you also have a local frozen Answerer checkpoint:

```bash
ANSWER_BACKEND=local_hf \
ANSWER_MODEL=/mnt/local2/wxy/models/Qwen3-0.6B \
ANSWER_DEVICE_MAP=auto \
ANSWER_TORCH_DTYPE=auto
```

During evaluation, Memory Builder sees only case sessions while constructing
Current Memory. The benchmark question and answer are used only after memory is
built, by the Answerer and correctness Judge.

Verify a case-evaluation output directory manually:

```bash
python scripts/verify_case_evaluation_output.py \
  --output_dir outputs/production_post_training_case_evaluation \
  --input data/longmemeval/longmemeval_s_cleaned.json
```

This verifier checks that the manifest succeeded, per-case memory/answer/judge
reports exist, `num_records` / `num_judged` / `accuracy` are consistent, and
Current Memory does not leak the exact benchmark target question. Exact answer
matches are warnings by default because the answer may be a true fact in raw
history; use `--strict_answer_leakage` if you want those to be hard failures.

Audit a completed production run:

```bash
python scripts/audit_production_run.py \
  --training_output_dir outputs/production_native_verl_dual_cotraining \
  --evaluation_output_dir outputs/production_post_training_case_evaluation \
  --split_manifest outputs/splits/longmemeval_s.existing_graphs.seed1029.json \
  --expected_evaluation_split test
```

You can also run verification, checkpoint resolution, and evaluation as one
workflow:

```bash
cd /mnt/local2/wxy/GAAM/gaam_memory_training

TRAINING_OUTPUT_DIR=outputs/production_native_verl_dual_cotraining \
SPLIT_MANIFEST=outputs/splits/longmemeval_s.existing_graphs.seed1029.json \
EVALUATION_SPLIT=test \
EVAL_OUTPUT_DIR=outputs/production_post_training_case_evaluation \
ANSWER_BACKEND=api \
ANSWER_API_KEY="${DEEPSEEK_API_KEY}" \
JUDGE_BACKEND=api \
JUDGE_API_KEY="${DEEPSEEK_API_KEY}" \
RECORD_ID=e47becba \
bash scripts/run_production_post_training_evaluation.sh
```

---

## 13. Full-Parameter Training Requirements

The native VERL script explicitly disables LoRA:

```text
actor_rollout_ref.model.lora_rank=0
actor_rollout_ref.model.lora_alpha=0
```

It also requests full checkpoint materialization:

```text
actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra']
```

This means the expected mode is full-parameter GRPO training, not LoRA. The
native scripts also default to `REQUIRE_HF_CHECKPOINT=1`; if VERL exits without
writing a `huggingface/` or `hf_model/` checkpoint directory, GAAM treats the
actor step as failed. Set `REQUIRE_HF_CHECKPOINT=0` only for smoke tests that do
not need updated model weights.

---

## 14. Common Problems

### flash-attn cannot find torch

Cause: `flash-attn` was installed before PyTorch or with build isolation.

Fix:

```bash
python -m pip install flash-attn --no-build-isolation
```

### flash-attn undefined symbol

You may see:

```text
ImportError: ... flash_attn_2_cuda...so: undefined symbol: _ZN3c105Error...
```

Cause: the installed `flash-attn` binary was built against a different
PyTorch/CUDA/C++ ABI than the currently active environment.

First verify the active PyTorch:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
PY
```

Then reinstall `flash-attn` from source against the current PyTorch:

```bash
python -m pip uninstall -y flash-attn
python -m pip cache remove flash-attn || true

MAX_JOBS=8 python -m pip install \
  --no-build-isolation \
  --no-cache-dir \
  --no-binary flash-attn \
  flash-attn
```

Verify:

```bash
python - <<'PY'
import torch
import flash_attn
print("torch:", torch.__version__, torch.version.cuda)
print("flash_attn:", flash_attn.__version__)
PY
```

If source build fails because `nvcc` is unavailable, load a CUDA toolkit module
matching your PyTorch CUDA build, or install a prebuilt `flash-attn` wheel that
matches the exact PyTorch/CUDA/Python combination.

If source build fails with a CUDA mismatch:

```text
The detected CUDA version (11.8) mismatches the version that was used to compile
PyTorch (12.4).
```

then your active `nvcc` is CUDA 11.8 while PyTorch is `+cu124`. Fix this in one
of two ways:

Option A, recommended: load CUDA 12.4 toolkit and rebuild flash-attn:

```bash
which nvcc
nvcc --version

# Example only; use your cluster's actual module name.
module avail cuda
module unload cuda || true
module load cuda/12.4

which nvcc
nvcc --version

python -m pip uninstall -y flash-attn
MAX_JOBS=8 FLASH_ATTENTION_FORCE_BUILD=TRUE python -m pip install \
  --no-build-isolation \
  --no-cache-dir \
  --no-binary flash-attn \
  flash-attn==2.8.3
```

Option B: keep CUDA 11.8 toolkit and reinstall a PyTorch build compiled with
CUDA 11.8. Only do this if vLLM and the rest of the environment support that
stack:

```bash
python -m pip uninstall -y torch torchvision torchaudio flash-attn
python -m pip install torch==2.6.0 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118
```

Do not compile `flash-attn` with CUDA 11.8 against `torch==2.6.0+cu124`.

If pip guesses a wheel URL such as `2.8.3.post1` but that release asset does not
exist, do not chase the guessed filename. Install an existing local wheel that
matches the active environment, for example:

```bash
python -m pip install --force-reinstall --no-deps \
  /path/to/flash_attn-2.8.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

This wheel name matches:

- Python `cp311`
- PyTorch `2.6`
- CUDA 12.x
- `torch._C._GLIBCXX_USE_CXX11_ABI == False`

After installing a local wheel, always verify import:

```bash
python - <<'PY'
import torch
import flash_attn
print("torch:", torch.__version__, torch.version.cuda)
print("cxx11_abi:", torch._C._GLIBCXX_USE_CXX11_ABI)
print("flash_attn:", flash_attn.__version__)
print("flash_attn file:", flash_attn.__file__)
PY
```

### pip reports dependency conflicts after installing vLLM

You may see warnings like:

```text
opentelemetry-exporter-prometheus 0.64b0 requires opentelemetry-sdk~=1.43.0,
but you have opentelemetry-sdk 1.26.0 which is incompatible.

google-api-core 2.31.0 requires protobuf<8.0.0,>=5.29.6,
but you have protobuf 4.25.9 which is incompatible.
```

This warning does not necessarily mean `vllm` failed. First check whether vLLM
imports:

```bash
python - <<'PY'
import vllm
print("vllm:", vllm.__version__)
PY
```

Then repair the conflicting packages:

```bash
python -m pip uninstall -y opentelemetry-exporter-prometheus google-api-core

python -m pip install --force-reinstall \
  "opentelemetry-api>=1.26.0,<1.27.0" \
  "opentelemetry-sdk>=1.26.0,<1.27.0" \
  "opentelemetry-semantic-conventions==0.47b0" \
  "opentelemetry-proto>=1.26.0,<1.27.0" \
  "opentelemetry-exporter-otlp-proto-http>=1.26.0,<1.27.0" \
  "opentelemetry-exporter-otlp-proto-grpc>=1.26.0,<1.27.0" \
  "protobuf>=3.20.3,<5.0"

python -m pip check
```

Why uninstall those two packages?

- `vllm==0.8.4` requires `opentelemetry-api/sdk >=1.26,<1.27`.
- `opentelemetry-exporter-prometheus 0.64b0` requires OpenTelemetry `1.43.x`,
  which conflicts with vLLM.
- `opentelemetry-proto 1.26.0` requires `protobuf<5`.
- `google-api-core 2.31.0` requires `protobuf>=5.29.6`, which conflicts with
  the vLLM-compatible OpenTelemetry stack.

If `pip check` still reports conflicts, prefer removing packages unrelated to
GAAM/VERL from this dedicated training environment.

### VERL import fails

Check:

```bash
export CODE_A1_ROOT=/mnt/local2/wxy/GAAM/Code-A1/Code-A1
export VERL_ROOT=${CODE_A1_ROOT}/verl
export PYTHONPATH=/mnt/local2/wxy/GAAM/gaam_memory_training:${CODE_A1_ROOT}:${VERL_ROOT}:${PYTHONPATH}
python - <<'PY'
import verl
print(verl.__file__)
PY
```

### transformers does not provide AutoModelForVision2Seq

You may see:

```text
ImportError: cannot import name 'AutoModelForVision2Seq' from 'transformers'
```

For GAAM's Qwen3 text-only training, `AutoModelForVision2Seq` is not needed.
Some VERL versions still import it unconditionally. The repo includes a
compatibility patch in the vendored Code-A1/VERL files so text-only models fall
back to `AutoModelForCausalLM` when this symbol is missing.

If your checkout does not include that patch, either sync the updated repo or
upgrade `transformers` to a version that provides `AutoModelForVision2Seq` and is
still compatible with your vLLM version.

### Qwen2Tokenizer has no all_special_tokens_extended

You may see:

```text
AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended
```

This usually means `transformers` was upgraded to a 5.x version that is not
compatible with the current vLLM/VERL stack. Pin `transformers` back to 4.x:

```bash
python -m pip uninstall -y transformers tokenizers
python -m pip install --force-reinstall \
  "transformers>=4.51.0,<5.0" \
  "tokenizers>=0.21,<0.22"

python - <<'PY'
import transformers
from transformers import AutoTokenizer
print("transformers:", transformers.__version__)
tok = AutoTokenizer.from_pretrained("/mnt/local2/wxy/models/Qwen3-0.6B", trust_remote_code=True)
print(type(tok))
print("all_special_tokens_extended:", hasattr(tok, "all_special_tokens_extended"))
PY
```

Avoid reinstalling Code-A1/VERL requirements with an unpinned `transformers`
entry afterward, because it may upgrade back to 5.x.

### CUDA unavailable

Check:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
PY
```

If `nvidia-smi` works but `torch.cuda.is_available()` is false, reinstall a CUDA
PyTorch wheel compatible with the driver.

### Dataset split file missing

Create the split first:

```bash
cd /mnt/local2/wxy/GAAM/gaam_memory_training

python scripts/create_dataset_split.py \
  --input data/longmemeval/longmemeval_s_cleaned.json \
  --oracle_graph_dir outputs/longmemeval_s_graph \
  --output outputs/splits/longmemeval_s.phase4_split.seed0.json \
  --train_ratio 0.8 \
  --dev_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 0 \
  --allow_empty_split \
  --overwrite
```

### Out of memory

Reduce:

```bash
MEMORY_NUM_GPUS=1
QUESTION_NUM_GPUS=1
TRAIN_BATCH_SIZE=2
PPO_MINI_BATCH_SIZE=1
PPO_MICRO_BATCH_SIZE_PER_GPU=1
ROLLOUT_N=2
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=1024
GPU_MEMORY_UTILIZATION=0.5
```

For real training, increase these gradually after the one-step smoke test passes.
