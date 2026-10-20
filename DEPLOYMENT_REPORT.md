# DEPLOYMENT_REPORT.md — `training-qat` 交付与验收报告

> 交付物：`inference-training/training-qat/` v0.1.0（长任务 4 · 阶段 3）
> 契约：`training-qat-contract.md` **frozen 1.0.0**；`qat-verification-contract.md` **frozen 1.0.0**
> 证据等级：`[verified]` 本机一手实测 / `[claimed]` 二手 / `[inferred]` 推论。
>
> **撰写说明**：本文件由队长（captain）在归属子代理结束后、依据已复核的现场证据补齐
> （该子代理交付了 `ACCEPTANCE_MATRIX.md` 与全部代码/证据，但未产出本文件）。
> 凡未验证项一律标 ❌ 或"未验证"。

---

## 1. 一句话

**接口层全部验证通过**（serving 可加载 + 真实前向 + scheme 从产出读回 + 产物级断言 + 跨插件验收入口）；
**精度门禁未达标，差 0.5684 个百分点**（+3.5684% vs 调用方给的 3%）；
吞吐 +45.08%，但**未声明 perf 目标**故该维度不予判定。
契约按任务书指定的**三条接口条件**冻结；精度未达标被**显式登记为"结果"**（契约 §4B），不是静默放过。

---

## 2. 框架与方案（**由实测选定，不是假定**）

| 候选 | 实测判定 | 证据 |
|---|---|---|
| `prepare_qat_fx`（原计划的兜底路径） | ❌ 在真实 HF CausalLM 上失败：`ValueError: code: co_varnames is too small` | `evidence/probe-fxqat.json` |
| eager `torch.ao.quantization.prepare_qat` | ⚠️ 能插模块，但 `convert` 产出不可序列化的 `quantized.Linear` GraphModule | 同上 |
| torchao `Int4WeightOnlyConfig` | ❌ `Requires mslk>=1.0.0`；而**装 mslk 会让 `import torchao` 在 GPU 上直接失败**（ABI 不兼容）⇒ 明确不装 | `out/qat-probe/step3b`, `step5-gpu` |
| torchao `Intx*` convert | ⚠️ 能转，但**不产 `quantization_config`** 且 `save_pretrained` 报错 ⇒ **无可加载产物** | `out/qat-probe/step3d` |
| **torchao `QATConfig(step="prepare")`** | ✅ 真实 0.5B 上插入 169 个 `FakeQuantizedLinear`，真实 fwd+bwd+AdamW | `evidence/probe-qat-engine.json` |
| **llmcompressor `QuantizationModifier`** | ✅ 产出 compressed-tensors；但 **INT8 在本机不可服务** | `out/probe-vllm-int8-sm121.log` |

**最终方案**：QAT 训练 = torchao STE 伪量化；QAT 产出 = 镜像自带的 llmcompressor 真量化；
**默认 scheme = `fp8-dynamic`**（因 `Int8 not supported on SM121 … Use FP8 quantization instead`），
`int8-w8a8` 保留并如实标 `servableOnThisHost: false`。

## 3. 装机与运行

| 项 | 值 |
|---|---|
| 工具链镜像 | `training-qat/toolchain:0.1.0`（`FROM training-sft/toolchain:0.1.0` + torchao 0.18.0） |
| 宿主 | NVIDIA GB10 / aarch64 / 驱动 580.126.09 / CUDA 13.0 / sm_121 |
| 容器约定 | `--user 1000:1000` + `HOME` + **`USER`/`LOGNAME`**（非 root uid 必需，否则 `getpass` 失败） |
| 共享状态 | `~/.ai-infra/state/qat-registry.yaml` |
| 审计链 | `~/.ai-infra/logs/audit/training-qat/{date}.jsonl` |
| 产出目录 | `~/.ai-infra/quantized/<model>-qat-<runId>/` |

## 4. 工具面（恰好 6 个，命名空间 `training_qat_*`）

`training_qat_prepare` · `training_qat_train` · `training_qat_validate` · `training_qat_rollback` · `training_qat_get_report` · `training_qat_health`

逐个实调通过（见 `out/lt4-tool-invocation.json`）：无损 JSON、无 `undefined` 键、错误码路径具体
（`E_CONFIRMATION_REQUIRED` / `E_RUN_NOT_FOUND` / `E_SCHEME_DRIFT` / `E_LICENSE_NOT_TRAINABLE`）。
**边界**：经插件自身的 `mcp/server.mjs` 实调；**未在 DSH 内注册后实调**（同 RL，需重启进程，见 §7）。

## 5. 真机报告

### 5.1 训练 run `tq_20260929154102_mf06`

| 项 | 值 |
|---|---|
| 数据 | `yahma/alpaca-cleaned`（`cc-by-4.0`，permissive；**曾误用 wikitext，已回滚并加 fail-closed 闸门**） |
| 设备 / scheme | GPU / `fp8-dynamic` |
| 步数 / 墙钟 / GPU·h | **200 / 271.5 s / 0.075** |
| final loss | 2.183 |
| 起点权重只读 | ✅ `sha256:9f55e03f…` 运行前后**逐位相同** |

### 5.2 产物级断言

✅ **PASS**：源模型 `sha256:9f55e03f…` ≠ QAT 产出权重；`assertArtifactChanged` 通过；
源模型**未被改动**。（"只断言过程抓不到空操作，必须断言权重"。）

### 5.3 serving 加载 + 真实前向（冻结条件 2）

✅ **通过**：vLLM 加载成功（`usabilityOk=true`，image digest `…be5ff841…`）；
validity **5/5 ok** —— `qat-forward` / `base-forward` 各取得逐文本 NLL，两侧 `not-uniform` 均 ok，
`o1-leak-gate` ok（hash 0 / substring 0）。

### 5.4 scheme 从产出**读回**（不是声明）

✅ **`8bit/channel`**，`compressed-tensors`，`float-quantized`，weights `num_bits 8 / channel / symmetric / type float`。
⇒ **M3 语义漂移已解决**：`qat.mjs` 曾声明的 `W4A16` 与实际产出**不一致**，现在以**读回值**为准。

### 5.5 精度门禁 —— ❌ **未达标**

| 指标 | QAT | 基线 | 变化 | 阈值 | 判定 |
|---|---|---|---|---|---|
| 困惑度（**serving 仪器**） | **6.562785** | 6.336668 | **+3.5684%** | 3% | ❌ **超出 0.5684pp** |
| `mean_nll_val` | 1.850532 | 1.815444 | +1.93% | — | — |
| 解码吞吐 | **242.42** tok/s | 167.10 | **+45.08%** | **未声明** | ⚠️ 不予判定 |

> **注意区分**：`gate.pass=false` 的原因是 **"数值未达标"**，而 `validityOk=true`
> ⇒ **测量成立、数值差 0.57pp**。两者不可混读。

### 5.6 跨插件验收（M2）

✅ **已跑通**：`inference_quantization_validate`（真实产物）→
`scheme=8bit/channel`、`schemeRecognized=true`、`measurement.valid=true`、
**`parity.dropPct=1.294%`**、`decision=sign`；
记录 `~/.ai-infra/run/inference-quantization/validate-existing/qv_5bf38e2ac8e9.json`，`auditSeq=88`，**未写矩阵**。

> ⚠️ **两台仪器分歧 2.3×**：量化仪器 **1.294%**（wikitext-400）vs serving 仪器 **3.5684%**（Alpaca-val-48）。
> 按 `training-eval-contract` §5.1，**训练侧判定以 serving 为准** ⇒ **未达标**。
> **没有签收那个好看的数字。** 该分歧的性质未被解释（测试集不同 + 仪器不同），列为未验证项。

## 6. 真机过程中修掉的三个真实缺陷

| # | 缺陷 | 根因 | 影响 |
|---|---|---|---|
| 1 | **产出无 `LICENSE`，serving 拒绝部署** | 量化模型是衍生作品，但 provenance 未传递 | 每个 QAT 产物都会撞同一堵墙；已改为**从源继承**写入，而非用 `confirm` 令牌绕过 |
| 2 | **llmcompressor 写成 1,629 字节的 stub `tokenizer.json`** | 真实文件 7,031,635 B | 所有 prompt 分词为 `[]` ⇒ vLLM 报 "decoder prompt cannot be empty"；转换器现会带上基座 tokenizer/chat-template 并**断言分词非空** |
| 3 | **INT8 W8A8 在本机不可服务** | `Int8 not supported on SM121` | 默认 scheme 改为 `fp8-dynamic`（torchao float8 伪量化，使**模拟方案 = 实际产出方案**） |

另有 **license 越界自纠**：首版 QAT 曾用 wikitext 作训练语料，违反 `license-whitelist` §2.1
（share-alike 只可作校准/评估）⇒ 已**回滚并 purge** 该 run，加入 fail-closed 的
`E_LICENSE_NOT_TRAINABLE` / `E_LICENSE_UNRESOLVED` 闸门，改用 `cc-by-4.0` 数据重跑。

## 7. 已知问题与处置

| # | 问题 | 处置 |
|---|---|---|
| 1 | **QAT vs PTQ 同 scheme 未对比** | **未测量 ⇒ 不主张 QAT 优于 PTQ** |
| 2 | 仪器分歧 2.3× 未解释 | 需按 `training-eval-contract` §5.1 做双路径校准并落盘 |
| 3 | **单种子** | `E_SEEDS_LT_3` ⇒ 只能表述"**该模型**"，不得表述"该流程" |
| 4 | 量化镜像未做 scan+signature 验证 | 跨插件调用用了插件自身**已审计**的 `allowUnverifiedImage` 开关，并记入输出 |
| 5 | `qat.mjs:67/68` 的 `W4A16` 措辞与错误 `frameworkHint` | **只报不改**（属 `inference-quantization` 源码，范围纪律） |
| 6 | 3 个 INT8 产物留在盘上作为"本机不可服务"的证据 | 保留 |
| 7 | **RL/QAT 未在 DSH 内运行时装载** | 根因：长驻 host 缓存了 `training-core` 升级**前**的模块实例 ⇒ 需**重启 DSH**；属**系统级变更**，须授权 |

## 8. 复核命令

```bash
cd inference-training/training-qat && npm test          # 5 spec / 178 断言
cd inference-quantization && npm test                   # 146/146（含新增 7 集成 + 7 单测）
node inference-contracts/checks/run-all.mjs             # 117/117（含 C21a–C21v）
```

## 9. 回滚

- **不签收**：量化矩阵不产生 `verified` 记录（本次跨插件验收**未写矩阵**）。
- **删产出**：只删本插件登记的产出目录，**源模型不受影响**（已核：源权重逐位未变）。
- **拆端点**：`validate` 自带 `teardown`，已确认 `ok=true`。
- 已 purge 的 wikitext run 的语料与产物目录均已删除。

## 10. 结论

| 项 | 判定 |
|---|---|
| 接口层（M1/M2/M3 + serving + 契约） | ✅ **全部通过** |
| 精度 | ❌ **未达标，差 0.5684pp** |
| 交付形态 | **完整交付**（包 + 双契约 frozen + 真机 QAT + 跨插件验收 + 22 条 CI 校验），**附带一条未达标的精度结论** |
