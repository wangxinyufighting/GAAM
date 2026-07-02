# GAAM Memory Refactoring Policy 正式训练说明

本文档说明当前正式训练版本的代码功能、数据格式、训练流程和启动方式。

## 1. 目标

本版本训练的对象是 **Memory Refactoring Policy**，不是 Answer Agent，也不是 Retriever。

策略输入为：

- 当前攻击问题 `question`
- 黄金证据 `gold_evidence`
- 黄金答案 `gold_answer`
- 当前 memory snapshot
- 检索到的候选 chunks
- local / near / anchor 测试 ID

策略输出为一个 JSON 格式的 `MemoryPatch`：

- `ADD`
- `REFACTOR`
- `REVISE`
- `SPLIT`
- `LINK_ONLY`
- `NO_OP`
- `DEPRECATE`

其中 `REFACTOR` 是 many-to-many 操作，允许多个旧 chunk 和新证据生成多个新 chunk、多个更新 chunk、若干 deprecated chunk，以及若干 weighted question edges。

## 2. 代码结构

### 2.1 核心 schema

文件：`gaam_graph/memory_refactor_schema.py`

定义正式训练使用的数据结构：

- `MemoryChunk`
- `AtomicFact`
- `QuestionEdge`
- `MemoryPatch`
- `PatchEvalResult`
- `CommitGateConfig`
- `WeightedRewardConfig`

关键设计：

- 用 `question_edges` 替代简单的 `linked_questions`
- 每条 question edge 包含 `role`、`weight`、`pass_count`、`fail_count`
- chunk 保留 `parent_chunks`、`provenance`、`validity_scope`
- `LINK_ONLY` 只能改边，不能改 chunk
- `NO_OP` 不能产生任何状态修改
- `DEPRECATE` 通过状态标记完成，不物理删除 chunk

### 2.2 Sandbox 与 reward

文件：`gaam_graph/memory_refactor_env.py`

提供训练 reward 所需的无副作用 sandbox：

- `MemoryEnv.load_snapshot`
- `MemoryEnv.retrieve`
- `MemoryEnv.apply_patch_sandbox`
- `MemoryEnv.evaluate_patch`
- `compute_weighted_reward`
- `commit_gate_allows`
- `CommitManager`

训练阶段只在 snapshot 副本上测试 patch，不修改主 memory store。

`reward` 与 `commit gate` 分离：

- reward 给 GRPO 使用，允许提供连续训练信号
- commit gate 保护主记忆库，只允许硬条件全部通过的 patch 提交

当前 gate 条件包括：

- JSON 与 schema 合法
- 当前攻击题修复成功
- 当前答案有证据支持
- local regression 达标
- anchor 达标
- 无 unsupported fact
- 无 conflict
- 无 over-compression

### 2.3 VERL reward 入口

文件：`gaam_graph/verl_gaam_reward.py`

新增数据源：

```text
adversarial_memory_refactor
```

VERL 调用：

```python
compute_score(data_source, solution_str, ground_truth, extra_info)
```

当 `data_source == "adversarial_memory_refactor"` 时：

1. 解析模型输出的 `MemoryPatch`
2. 从 `memory_snapshot_id` 加载 snapshot
3. 在 sandbox 中应用 patch
4. 运行 current / local / near / anchor 测试
5. 返回 reward 和 diagnostics

reward 函数不提交主库，不写训练状态，不产生跨样本副作用。

### 2.4 数据集导出

文件：`gaam_graph/memory_refactor_training.py`

正式训练数据导出入口：

```python
export_memory_refactor_grpo_dataset(config)
```

输出：

- `memory_refactor.train.parquet`
- `memory_refactor.val.parquet`
- `memory_refactor.dataset_manifest.json`
- `snapshots/*.snapshot.json`

parquet 行结构兼容 native VERL：

```json
{
  "data_source": "adversarial_memory_refactor",
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "ability": "memory_refactor",
  "reward_model": {
    "style": "rule",
    "ground_truth": {
      "question": "...",
      "gold_answer": "...",
      "gold_evidence": "...",
      "memory_snapshot_id": "..."
    }
  },
  "extra_info": {
    "attack_id": "...",
    "memory_snapshot_id": "...",
    "retrieved_chunk_ids": ["..."],
    "local_test_ids": ["..."],
    "near_test_ids": ["..."],
    "anchor_test_ids": ["..."]
  }
}
```

### 2.5 Prompt 构造

文件：`gaam_graph/memory_refactor_dataset.py`

负责构造 Memory Refactoring Policy 的单轮 JSON patch prompt。

prompt 明确要求：

- 不回答问题，只输出 `MemoryPatch`
- 不强行把独立事实压成一个 chunk
- 新增或修订的 atomic facts 必须由 `gold_evidence` 支持
- 保留时间范围、条件、例外、provenance、parent chunks
- 使用带权 `question_edge_updates`

## 3. 训练样本格式

输入文件支持 `.json` 或 `.jsonl`。

JSON 文件可以是数组：

```json
[
  {
    "attack_id": "atk_0001",
    "split": "train",
    "question": "What code editor does the user primarily use now?",
    "gold_answer": "Cursor",
    "gold_evidence": "The user now primarily uses Cursor as their code editor.",
    "memory_snapshot": {
      "chunks": [
        {
          "id": "mem_editor_old",
          "content": "The user previously used VSCode as a code editor.",
          "atomic_facts": [
            {
              "fact_id": "fact_old_editor",
              "text": "The user previously used VSCode as a code editor.",
              "confidence": 0.9
            }
          ],
          "question_edges": [
            {
              "question_id": "q_editor_history",
              "role": "core",
              "weight": 0.92,
              "pass_count": 3
            }
          ]
        }
      ],
      "historical_questions": {
        "q_editor_history": "What code editor did the user use previously?"
      }
    },
    "relation_hints": ["temporal_update"],
    "local_test_ids": ["q_editor_history"],
    "near_test_ids": [],
    "anchor_test_ids": []
  }
]
```

也可以是对象：

```json
{
  "samples": [...]
}
```

如果 snapshot 已经单独落盘，可以使用：

```json
{
  "memory_snapshot_path": "/absolute/path/to/snapshot.json"
}
```

## 4. 数据集导出命令

```bash
cd gaam_memory_training

python scripts/export_memory_refactor_grpo_dataset.py \
  --input data/memory_refactor/train_samples.jsonl \
  --output_dir outputs/memory_refactor_grpo/dataset
```

可选参数：

```bash
--max_records_per_split 100
```

导出成功后会生成：

```text
outputs/memory_refactor_grpo/dataset/
  memory_refactor.train.parquet
  memory_refactor.val.parquet
  memory_refactor.dataset_manifest.json
  snapshots/
```

## 5. 正式训练启动脚本

文件：`scripts/run_memory_refactor_grpo_training.sh`

最小启动方式：

```bash
cd gaam_memory_training

INPUT_PATH=data/memory_refactor/train_samples.jsonl \
MODEL_PATH=/path/to/hf_model \
CODE_A1_ROOT=/path/to/Code-A1/Code-A1 \
bash scripts/run_memory_refactor_grpo_training.sh
```

常用参数：

```bash
NUM_GPUS=2
ROLLOUT_N=8
TRAIN_BATCH_SIZE=8
PPO_MINI_BATCH_SIZE=4
TOTAL_EPOCHS=1
TOTAL_TRAINING_STEPS=100
OUTPUT_ROOT=outputs/memory_refactor_grpo
```

脚本流程：

1. 调用 `export_memory_refactor_grpo_dataset.py`
2. 生成 train / val parquet
3. 设置 `custom_reward_function.path=gaam_graph/verl_gaam_reward.py`
4. 设置 `custom_reward_function.name=compute_score`
5. 用 `algorithm.adv_estimator=grpo` 启动 `verl.trainer.main_ppo`
6. 将 checkpoint 写入 `OUTPUT_ROOT/checkpoints`

## 6. 训练与提交的边界

正式 GRPO 训练只更新 Memory Refactoring Policy 模型参数。

训练时：

- 模型对同一 prompt 采样多个 patch
- reward function 分别 sandbox 评分
- GRPO 使用组内相对 reward 更新 policy
- reward 阶段不提交主 memory store

提交时：

- 由外层在线 commit loop 选择候选 patch
- 只提交通过 hard gate 的最佳 patch
- 未通过 gate 的攻击样本应进入 high-priority buffer

本版本已经实现训练侧所需的数据、prompt、reward 和启动入口；在线主库事务提交可以在后续接入真实 MemoryStore 时扩展。

## 7. 验证命令

建议改动后运行：

```bash
cd gaam_memory_training
pytest -q tests/test_memory_refactor_policy.py tests/test_memory_refactor_training_export.py
```

完整包级测试：

```bash
cd gaam_memory_training
pytest -q tests
```

