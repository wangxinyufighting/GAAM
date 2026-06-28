# GAAM Reward 设计说明

本文档说明 GAAM 中 `Memory Builder` 和 `Question Agent` 的 reward 设计、信号来源、是否使用 rule-based 或 LLM-as-judge，以及对应代码位置。

## 1. 总体原则

GAAM 的训练目标不是让模型记住某一个 benchmark question，而是让模型学会从 raw history 构建一个有用、压缩、抽象、无泄露的 Current Memory，并让 Question Agent 持续产生覆盖面广、有诊断价值的问题。

真实训练入口使用 native Code-A1/VERL GRPO：

- 一键端到端入口：`scripts/run_production_e2e_training_and_evaluation.sh`
- 训练脚本：`scripts/run_production_gaam_training.sh`
- 双智能体调度：`gaam_graph/native_verl_dual_trainer.py`
- VERL 数据导出：`gaam_graph/native_verl_grpo.py`
- VERL reward 函数：`gaam_graph/verl_gaam_reward.py`

LLM-as-judge 采用 OpenAI-compatible API，配置项如下：

```bash
GAAM_REWARD_JUDGE_ENABLED=1
GAAM_REWARD_JUDGE_BASE_URL=https://api.deepseek.com
GAAM_REWARD_JUDGE_API_KEY="${DEEPSEEK_API_KEY}"
GAAM_REWARD_JUDGE_MODEL=deepseek-v4-flash
GAAM_REWARD_JUDGE_REQUIRED=0
GAAM_REWARD_JUDGE_TIMEOUT=60
GAAM_REWARD_JUDGE_MAX_INPUT_CHARS=20000
```

其中：

- `GAAM_REWARD_JUDGE_ENABLED=1`：启用 API judge。
- `GAAM_REWARD_JUDGE_REQUIRED=1`：如果 judge 调用失败，则 reward 置为 0；默认 `0` 表示 judge 失败时保留 rule-based reward，并在 reward details 中记录失败原因。
- `GAAM_REWARD_JUDGE_BASE_URL`：OpenAI-compatible endpoint，可使用 DeepSeek、OpenAI 或本地 vLLM。
- `GAAM_REWARD_JUDGE_API_KEY`：API key。
- `GAAM_REWARD_JUDGE_MODEL`：judge 模型名。
- `GAAM_REWARD_JUDGE_TIMEOUT`：API judge 超时时间。
- `GAAM_REWARD_JUDGE_MAX_INPUT_CHARS`：单次 reward judge 输入的最大字符数，避免把过长 oracle/current memory 全量送入 judge。

旧版/离线 `RewardManager` 路径也支持 API judge。如果设置了下面的变量，则优先使用它们；否则自动回退到 `GAAM_REWARD_JUDGE_*`：

```bash
GAAM_REWARD_MANAGER_JUDGE_BASE_URL=https://api.deepseek.com
GAAM_REWARD_MANAGER_JUDGE_API_KEY="${DEEPSEEK_API_KEY}"
GAAM_REWARD_MANAGER_JUDGE_MODEL=deepseek-v4-flash
GAAM_REWARD_MANAGER_JUDGE_REQUIRED=0
GAAM_REWARD_MANAGER_JUDGE_TIMEOUT=60
GAAM_REWARD_MANAGER_JUDGE_MAX_CHARS=20000
```

P0/P1 安全约束：

- 空输出直接 `score=0`。
- Memory Builder 输出中禁止出现 `question`、`target_question`、`benchmark_question`、`answer`、`gold_answer` 等字段。
- Question Agent 合法输出可以包含普通 `question` 字段，但禁止包含 `target_question`、`benchmark_question`、`gold_answer`、`oracle_answer` 等泄露字段。
- 如果 `ground_truth` 中显式带有 `target_question` 或 `benchmark_question`，reward 会检查生成内容是否复制了 exact target question；命中则直接 `score=0`。
- API judge 的 `base_url`、`model`、`api_key_configured` 会写入 reward details；API key 本身不会写入日志。
- 如果 `correctness_mode="llm_judge"`，`RewardManager` 的答案正确性判断和 Question Agent 的 diagnostic value 会通过 OpenAI-compatible API 调用，而不是使用固定 placeholder。

## 2. Memory Builder Reward

目标：评估 Current Memory 是否能够支持后续回答、是否覆盖 oracle graph 中的重要信息、是否有抽象总结、是否不过度冗余或过度压缩、是否没有泄露 benchmark target。

代码位置：

- 主入口：`gaam_graph/verl_gaam_reward.py::_score_memory_builder`
- API judge：`gaam_graph/verl_gaam_reward.py::_maybe_llm_judge_memory`
- 生产训练调用：`scripts/run_production_gaam_training.sh`

生产训练默认要求每个 native VERL actor step 写出真实 HuggingFace full-model checkpoint：

```bash
SAVE_HF_MODEL=1
REQUIRE_HF_CHECKPOINT=1
```

如果 VERL 子进程返回成功但没有产生 `huggingface/` 或 `hf_model/` checkpoint，训练会被判为失败，避免“看起来训练完成、实际权重没有更新”的假成功。

训练结束后可以运行：

```bash
python scripts/verify_native_verl_training_output.py \
  --output_dir outputs/production_native_verl_dual_cotraining
```

该检查会验证 `dual_cotraining_manifest.json`、每个 actor step 的状态、native GRPO 日志标记、HF checkpoint 目录、以及最终模型路径。

| Reward 组件 | 中文名称 | 信号来源 | 设计目的 | 代码位置 |
|---|---|---|---|---|
| `parse_score` | 格式有效性 | rule-based | 要求输出可解析 JSON，并包含 `memory_graph`、`memory_summaries` | `verl_gaam_reward.py::_score_memory_builder` |
| `coverage_score` | Oracle 覆盖度 | rule-based + oracle digest | 检查 Current Memory 是否覆盖 oracle graph digest 中抽取的 required terms | `verl_gaam_reward.py::_coverage_score` |
| `abstraction_score` | 抽象/总结质量 | rule-based | 通过 summary、abstraction、preference、update、session、evidence、provenance 等关键词粗略检查是否包含抽象总结意识 | `verl_gaam_reward.py::_keyword_fraction` |
| `compression_score` | 压缩平衡 | rule-based | 惩罚过短和过长输出，避免过度压缩或冗余堆积 | `verl_gaam_reward.py::_length_band_score` |
| `leakage_penalty` | 泄露惩罚 | rule-based hard penalty | 惩罚 benchmark question、target question、gold answer、oracle answer 等泄露词 | `verl_gaam_reward.py::_leakage_penalty` |
| `hard_failure` | 硬失败 | rule-based hard filter | 空输出、非法字段、exact target question 泄露会直接 `score=0` | `verl_gaam_reward.py::_hard_failure` |
| `answer_correctness` | 答案正确性 | rule-based 或 LLM-as-judge API | 在离线/worker reward 中判断 Answerer 对训练问题的回答是否正确；`correctness_mode="llm_judge"` 时走 API | `reward_manager.py::_score_answer_correctness_with_mode` |
| `llm_judge.oracle_coverage` | LLM 判断的 oracle 覆盖度 | LLM-as-judge API | 判断 Current Memory 是否保留了足够关键事实、事件、偏好、更新和跨会话信息 | `verl_gaam_reward.py::_maybe_llm_judge_memory` |
| `llm_judge.groundedness` | 依据充分性 | LLM-as-judge API | 判断 memory 内容是否能由 oracle digest 支撑，降低幻觉 | `verl_gaam_reward.py::_maybe_llm_judge_memory` |
| `llm_judge.non_redundancy` | 非冗余性 | LLM-as-judge API | 判断 memory 是否重复堆叠同一信息 | `verl_gaam_reward.py::_maybe_llm_judge_memory` |
| `llm_judge.compression_balance` | 压缩不过度 | LLM-as-judge API | 判断 memory 是否既压缩又保留必要细节 | `verl_gaam_reward.py::_maybe_llm_judge_memory` |
| `llm_judge.abstraction_quality` | 抽象质量 | LLM-as-judge API | 判断 summary/abstract 是否有概括价值，而不是复制 raw turns | `verl_gaam_reward.py::_maybe_llm_judge_memory` |
| `llm_judge.leakage_safety` | 泄露安全性 | LLM-as-judge API | 判断输出是否包含 benchmark target 或 gold answer 泄露风险 | `verl_gaam_reward.py::_maybe_llm_judge_memory` |

当前聚合方式：

```text
如果不开 API judge:
  final_score = heuristic_score

如果开启 API judge 且成功:
  final_score = 0.45 * heuristic_score + 0.55 * llm_judge_score - leakage_penalty

如果开启 API judge 且失败:
  GAAM_REWARD_JUDGE_REQUIRED=0 -> 保留 heuristic_score
  GAAM_REWARD_JUDGE_REQUIRED=1 -> final_score = 0
```

注意：native VERL reward 中不能直接运行完整 Answerer rollout，因此当前 `verl_gaam_reward.py` 中的 Memory Builder reward 是“训练时即时 reward”。完整 answer-based evaluation 在 case evaluation 代码中执行：

- `gaam_graph/case_evaluation.py`
- `scripts/run_case_evaluation.py`

## 3. Question Agent Reward

目标：鼓励 Question Agent 产生 oracle-valid、answerable、多样、有难度、覆盖弱点、无重复、无泄露的问题。问题不能复制或推断 benchmark target question。

代码位置：

- 主入口：`gaam_graph/verl_gaam_reward.py::_score_question_agent`
- API judge：`gaam_graph/verl_gaam_reward.py::_maybe_llm_judge_questions`
- 问题数量控制：`gaam_graph/native_verl_grpo.py::_question_agent_prompt`

| Reward 组件 | 中文名称 | 信号来源 | 设计目的 | 代码位置 |
|---|---|---|---|---|
| `question_count_score` | 问题数量匹配 | rule-based | 要求输出问题数等于 `QUESTIONS_PER_CASE` | `verl_gaam_reward.py::_exact_count_score` |
| `diversity_score` | 类型多样性 | rule-based | 鼓励覆盖 single-hop、multi-hop、multi-session、temporal、preference、contradiction/update、summary/abstraction | `verl_gaam_reward.py::_question_diversity_score` |
| `coverage_score` | Oracle 覆盖度 | rule-based + oracle digest | 检查问题是否覆盖 oracle digest required terms | `verl_gaam_reward.py::_coverage_score` |
| `difficulty_score` | 难度/推理性 | rule-based | 鼓励多跳、跨会话、时间、偏好、更新、总结类问题 | `verl_gaam_reward.py::_keyword_fraction` |
| `redundancy_penalty` | 重复惩罚 | rule-based | 惩罚重复或高度相似的问题 | `verl_gaam_reward.py::_question_redundancy_penalty` |
| `leakage_penalty` | 泄露惩罚 | rule-based hard penalty | 惩罚 benchmark question、target question、gold answer、oracle answer 等泄露词 | `verl_gaam_reward.py::_leakage_penalty` |
| `hard_failure` | 硬失败 | rule-based hard filter | 空输出、非法泄露字段、exact target question 泄露会直接 `score=0`；普通 `question` 字段不算泄露 | `verl_gaam_reward.py::_hard_failure` |
| `diagnostic_value` | 诊断价值 | rule-based 或 LLM-as-judge API | 在离线/worker reward 中判断问题集是否能暴露 Memory Builder 的弱点；`correctness_mode="llm_judge"` 时走 API | `reward_manager.py::_score_question_diagnostic_value` |
| `llm_judge.oracle_validity` | Oracle 有效性 | LLM-as-judge API | 判断问题是否可以由 oracle graph digest 支撑 | `verl_gaam_reward.py::_maybe_llm_judge_questions` |
| `llm_judge.answerability` | 可回答性 | LLM-as-judge API | 判断问题是否能从 oracle graph 中回答，而不是开放式胡问 | `verl_gaam_reward.py::_maybe_llm_judge_questions` |
| `llm_judge.diversity` | 多样性 | LLM-as-judge API | 判断问题集合是否覆盖多种记忆能力 | `verl_gaam_reward.py::_maybe_llm_judge_questions` |
| `llm_judge.difficulty` | 难度 | LLM-as-judge API | 判断是否有足够挑战性，能够暴露 Memory Builder 缺陷 | `verl_gaam_reward.py::_maybe_llm_judge_questions` |
| `llm_judge.weakness_targeting` | 弱点定位 | LLM-as-judge API | 判断问题是否有诊断价值，是否能测试多会话、更新、偏好、抽象等弱点 | `verl_gaam_reward.py::_maybe_llm_judge_questions` |
| `llm_judge.non_redundancy` | 非冗余性 | LLM-as-judge API | 判断问题集合内部是否重复 | `verl_gaam_reward.py::_maybe_llm_judge_questions` |
| `llm_judge.leakage_safety` | 泄露安全性 | LLM-as-judge API | 判断是否存在 benchmark target question 泄露风险 | `verl_gaam_reward.py::_maybe_llm_judge_questions` |

当前聚合方式：

```text
如果不开 API judge:
  final_score = heuristic_score

如果开启 API judge 且成功:
  final_score = 0.40 * heuristic_score + 0.60 * llm_judge_score - leakage_penalty

如果开启 API judge 且失败:
  GAAM_REWARD_JUDGE_REQUIRED=0 -> 保留 heuristic_score
  GAAM_REWARD_JUDGE_REQUIRED=1 -> final_score = 0
```

## 4. 与 Evaluation 的关系

训练时 reward 不应直接使用 benchmark target question。Question Agent 使用 oracle graph digest 生成训练问题；Memory Builder 使用 raw history 构建 Current Memory。

最终 evaluation 可以使用 benchmark target question，因为此时 memory 已经构建完毕，不会泄露给 Memory Builder。

Evaluation 入口：

- `gaam_graph/case_evaluation.py`
- `gaam_graph/case_evaluation_verifier.py`
- `scripts/run_case_evaluation.py`
- `scripts/run_production_case_evaluation.sh`
- `scripts/verify_case_evaluation_output.py`

Evaluation 流程：

```text
case sessions
  -> Memory Builder 构建 Current Memory
  -> Answerer 读取 Current Memory 和 benchmark question
  -> Judge 比较 prediction 和 benchmark answer/rubric
  -> 输出 accuracy 与逐 case 报告
  -> verifier 检查报告完整性、accuracy 一致性、Current Memory 无 target question 泄露
```

Evaluation backend：

| 阶段 | 可选 backend | 用途 |
|---|---|---|
| Memory Builder | `baseline` | 非 LLM baseline，用于 smoke test |
| Memory Builder | `stateful_api` | 通过 OpenAI-compatible API 顺序构建 Current Memory |
| Memory Builder | `stateful_local_hf` | 加载训练后的本地 HuggingFace Memory Builder checkpoint，顺序构建 Current Memory |
| Answerer | `api` | 通过 OpenAI-compatible API 回答 |
| Answerer | `local_hf` | 加载本地 HuggingFace Answerer checkpoint 回答 |
| Answerer | `no_llm` | 非 LLM 检索 baseline，用于 smoke test |
| Judge | `api` | LLM-as-judge 判断答案正确性 |
| Judge | `heuristic` | exact/containment/token-F1 轻量判断 |

使用训练后的 Memory Builder checkpoint 进行 evaluation 示例：

```bash
MEMORY_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
QUESTION_MODEL_PATH=/mnt/local2/wxy/models/Qwen3-0.6B \
AUTO_CREATE_SPLIT=1 \
TRAINING_OUTPUT_DIR=outputs/production_native_verl_dual_cotraining \
EVAL_OUTPUT_DIR=outputs/production_post_training_case_evaluation \
EVALUATION_SPLIT=test \
GAAM_REWARD_JUDGE_ENABLED=1 \
GAAM_REWARD_JUDGE_API_KEY="${DEEPSEEK_API_KEY}" \
ANSWER_BACKEND=api \
ANSWER_API_KEY="${DEEPSEEK_API_KEY}" \
JUDGE_BACKEND=api \
JUDGE_API_KEY="${DEEPSEEK_API_KEY}" \
bash scripts/run_production_e2e_training_and_evaluation.sh
```

默认情况下，端到端脚本会把 `SPLIT_MANIFEST` 传给 evaluation，并用 `EVALUATION_SPLIT=test` 做最终 held-out evaluation。可以将 `EVALUATION_SPLIT` 改成 `dev` 用于模型选择检查，改成 `train` 用于调试，或改成 `all` 评估 split manifest 中所有 records。

端到端脚本最后还会生成：

```text
outputs/production_post_training_case_evaluation/production_run_audit.json
```

该 audit 会汇总检查：native VERL 训练是否成功、是否产出 full HuggingFace checkpoint、evaluation 是否使用预期 split、是否所有 selected records 都被 judge、evaluation leakage verifier 是否通过。

如果已经完成训练，只做 evaluation：

```bash
TRAINING_OUTPUT_DIR=outputs/production_native_verl_dual_cotraining \
SPLIT_MANIFEST=outputs/splits/longmemeval_s.existing_graphs.seed1029.json \
EVALUATION_SPLIT=test \
EVAL_OUTPUT_DIR=outputs/production_post_training_case_evaluation \
ANSWER_BACKEND=api \
JUDGE_BACKEND=api \
bash scripts/run_production_post_training_evaluation.sh
```

`scripts/run_production_post_training_evaluation.sh` 默认会在 evaluation 后运行：

```bash
python scripts/verify_case_evaluation_output.py \
  --output_dir outputs/production_post_training_case_evaluation \
  --input data/longmemeval/longmemeval_s_cleaned.json
```

该 verifier 会检查：

- `case_evaluation_manifest.json` 存在且状态为 `succeeded`。
- 每个 case 的 `current_memory.json`、`answer_report.json`、`judge_report.json` 都存在。
- `num_records`、`num_judged`、`accuracy` 与逐 case 报告一致。
- `current_memory.json` 不包含 benchmark target question 或非法泄露字段。
- 默认只把 exact benchmark answer/rubric 出现在 memory 中记为 warning，因为 answer 本身可能是 raw history 中的真实个人事实；如需把它也作为 hard failure，可使用 `--strict_answer_leakage`。

如果需要手动拆分步骤，也可以先解析 checkpoint：

```bash
eval "$(
  python scripts/resolve_native_verl_checkpoints.py \
    --output_dir outputs/production_native_verl_dual_cotraining \
    --format shell
)"

MEMORY_BACKEND=stateful_local_hf \
MEMORY_MODEL="${MEMORY_MODEL}" \
ANSWER_BACKEND=api \
JUDGE_BACKEND=api \
bash scripts/run_production_case_evaluation.sh
```

## 5. 当前实现边界

1. native VERL parquet 训练中的 `MEMORY_INPUT_MODE=incremental` 默认禁用，因为它只是 independent rows + static scaffold，不是真正 stateful rollout。
2. 真正 stateful incremental memory 构建通过 API 入口实现：`scripts/run_stateful_incremental_memory_builder.py`。
3. 如果后续要让 native VERL 训练本身也完全 stateful，需要进一步实现自定义 rollout worker，使 chunk `k` 的 prompt 使用 chunk `k-1` 的模型输出。
