---
name: training-qat
description: 量化感知训练（QAT）。当 PTQ（inference-quantization）在最优档位仍达不到调用方的精度容忍、闭环只剩「同方案下把精度损失压下去」这一条路时使用。也用于：验收一个**已存在**的量化 checkpoint、在指定 scheme 下做 QAT、以及对 QAT 产出做跨插件对账。
---

# training-qat

**QAT 是「主模型保持全精度」的受控例外**（`training-qat-contract` §5.1）：
只有本插件输出量化模型；SFT / pretrain / RL 一律全精度。
它**不改变主模型** —— 源模型只读挂载，产出落在独立目录。

## 什么时候用它

触发路径（契约 §5.2）：

```
inference_quantization_quantize 精度不达标（dropPct > 阈值，且测量成立）
   → inference_quantization_qat_plan        （只出计划，不执行）
   → training_qat_prepare / train           （本插件：QAT + 真量化）
   → inference_quantization_validate        （验收已有 checkpoint：跨插件对账）
   → training_qat_validate                  （serving 仪器上的精度恢复判定）
```

## 六个工具

| 工具 | 做什么 | 风险 |
|---|---|---|
| `training_qat_prepare` | 语料去重 + 确定性切分 + **O1 双轨去泄漏**（哈希交集为空 **且** val 文本不是 train 行子串） | write |
| `training_qat_train` | 容器内 **torchao QATConfig** 假量化（STE）训练 → 对 QAT 后的权重施加**真实 INT8 量化** → compressed-tensors 产出 | write / container / 需确认 |
| `training_qat_validate` | serving 上部署「QAT 产出 + 基座」→ 配对测量 → `dualGate` + O2/O3/O5 → **跨插件对账** | write / container / 需确认 |
| `training_qat_rollback` | 标记 rolled-back；`purge=true` 只删本 run 登记的路径 | write |
| `training_qat_get_report` | 读 registry + 审计链 + eval-result | read |
| `training_qat_health` | 自检（沙箱/镜像/GPU/serving/量化入口/判据参考实现/审计链） | read |

## 三条不可省的纪律

1. **产物级断言**：`training_qat_train` 断言「QAT 后的权重指纹 ≠ 源模型权重指纹」。
   loss 下降是**过程量**，一个什么都没写盘的 run 也能打印下降的 loss。
   指纹相同 ⇒ `E_NO_OP_TRAINING`；指纹缺失 ⇒ 按**不通过**处理。
2. **方案诚实（M3）**：输出 `config.json` 里的 `quantization_config` 是**从产出读回**的，
   不是声明的。请求 8bit 而产出不是 8bit ⇒ `E_SCHEME_DRIFT`。
   `qat.mjs:67` 写的 W4A16 **本插件不产出、也不声明**（本机实测无可用 W4A16 QAT 路径）。
3. **阈值外部给**：`training_qat_validate` 的 `maxAccuracyDropPct` **必须显式传入**。
   本插件不猜默认值 —— "自己定阈值、自己签收精度"是明确禁止的。

## 典型调用

```jsonc
// 1) 语料（O1 闸门；不过就拒绝训练）
training_qat_prepare { "train": 128, "val": 64 }

// 2) QAT（CPU 路径不占 GPU；返回里 scheme 是从产出读回的）
training_qat_train {
  "datasetDir": "…/qat__wikitext__s20260929__train128-val64",
  "device": "cpu", "maxSteps": 60, "maxSeqLen": 256, "lr": 1e-5,
  "confirm": "<E_CONFIRMATION_REQUIRED 给的 confirmToken>"
}

// 3) 验收（serving 仪器 + 跨插件对账）
training_qat_validate { "runId": "tq_…", "thresholds": { "maxAccuracyDropPct": 2 } }
```

## 硬上限与诚实记账

`maxSteps ≤ 500`、`maxGpuHours ≤ 2`、并发 1、参数 ≤ 0.5B。
超限**暂停并报告**，不静默降级。CPU 路径上 `gpuHours` **恒为 0**（不占 GPU 就不记 GPU 账）。

## 会失败的地方（按设计）

- `E_DATASET_LEAK`：O1 任一轨非空。
- `E_NO_OP_TRAINING` / `E_ARTIFACT_FINGERPRINT_MISSING`：权重没变，或无法断言。
- `E_SCHEME_DRIFT` / `E_SCHEME_UNSPECIFIED`：产出与请求的 scheme 不符，或产出没有 `quantization_config`。
- `E_THRESHOLD_NOT_PROVIDED`：没给 `maxAccuracyDropPct`。
- `E_SANDBOX_UNAVAILABLE`：没有容器沙箱（train/validate 需要容器层）。
- `E_MEASUREMENT_INVALID`（validity 未过）：**先判测量成立、再判数值**，两类失败必须可区分。
