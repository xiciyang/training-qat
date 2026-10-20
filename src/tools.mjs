/**
 * tools —— 6 个工具的规格（**MCP 与 DSH 两条通道共用同一份**，不存在行为分叉）。
 *
 * 命名空间纪律：`training_qat_*`，与既有四类命名空间
 * （`inference_serving_*` / `inference_quantization_*` / `inference_tuning_*` /
 *  `training_sft_*` / `training_pretrain_*`）**均无交集**
 * —— 由 `tests/unit/namespace.spec.mjs` 去兄弟包源码实时抓名并断言交集为空。
 *
 * DSH 工具契约（**踩过的坑**）：`ctx.tools.register` 要求 `output { schema, render }`
 * **同时**给出；缺 render 会让**整条 entry 被拒**。故每个工具都带 `output`，
 * 且 `src/index.mjs` 里用"像 DSH 一样挑刺"的 mock ctx 跑过。
 */
import { CONFIRM_PARAM, ROOT_PARAM, passthroughOutput } from './output.mjs'
import * as api from './core.mjs'

export const ALL_TOOL_SPECS = [
  {
    name: 'training_qat_prepare',
    description: 'Prepare and validate a QAT corpus. TWO independent gates, both fail-closed: (1) license gate — QAT updates weights, so a share-alike corpus (license-whitelist.md section 2.1) is REFUSED for training (E_LICENSE_NOT_TRAINABLE) and an unresolvable license is refused too (E_LICENSE_UNRESOLVED); the license is INHERITED from the source lineage, never invented here. (2) the training-eval-contract O1 leak gate on TWO tracks — whole-line content-hash intersection must be empty AND no val text may appear as a substring inside any train row (E_DATASET_LEAK). Default corpus is the permissive yahma/alpaca-cleaned; content-hash dedup + deterministic split; rows are never rewritten.',
    parameters: {
      type: 'object', additionalProperties: false, required: [],
      properties: {
        sourceDir: { type: 'string', description: '语料源目录（需含 train.jsonl+heldout.jsonl 或 calib.jsonl+test.jsonl，且 lineage.json 带 license.spdx）。缺省自动发现状态目录里最近的可训练（非 share-alike）语料。' },
        seed: { type: 'integer', description: '切分种子（同 seed 同输入 ⇒ 逐条相同的切分）。默认 20260929。' },
        train: { type: 'integer', description: '训练行数。默认 128。' },
        val: { type: 'integer', description: '留出行数。默认 64。' },
        outDir: { type: 'string', description: '输出目录；缺省落在 <root>/state/datasets/qat__…' },
        force: { type: 'boolean', description: '强制重新生成。默认 false。' },
        root: ROOT_PARAM,
      },
    },
    output: passthroughOutput(),
    handler: (a) => api.prepare({ root: a?.root, sourceDir: a?.sourceDir ?? null, seed: a?.seed, train: a?.train, val: a?.val, outDir: a?.outDir ?? null, force: a?.force === true }),
  },
  {
    name: 'training_qat_train',
    description: 'Run quantization-aware training through the shared monolithic loop (@xiciyang/training-core) inside a container sandbox: torchao QATConfig fake-quantization (STE) on real text, then the convert step applies genuine INT8 real-quantization to the QAT-trained weights via the repository\'s own llmcompressor INT8 scheme, producing a compressed-tensors checkpoint that inference-serving can load. Hard budget enforcement (maxSteps <=500, maxGpuHours <=2, concurrency 1, params <=0.5B). A run whose weights are byte-identical to the input FAILS (E_NO_OP_TRAINING) — the artifact fingerprint is asserted, not the loss curve. The scheme written to the output is READ BACK from the artifact, never declared.',
    parameters: {
      type: 'object', additionalProperties: false, required: ['datasetDir'],
      properties: {
        datasetDir: { type: 'string', description: 'training_qat_prepare 产出的语料目录（需含 train.jsonl 与 allClear=true 的 lineage.json）。' },
        baseModel: { type: 'string', description: '基座模型本地目录。默认 <BASE_MODEL>（默认取环境变量 DSH_BASE_MODEL）。' },
        device: { type: 'string', enum: ['cpu', 'cuda'], description: 'QAT 设备。默认 cpu。注意：scheme=fp8-dynamic（默认）**需要 cuda**（torchao float8 假量化在 CPU 上不可用），两者不一致会在起容器前被拒。' },
        scheme: { type: 'string', enum: ['fp8-dynamic', 'int8-w8a8'], description: '量化方案。默认 fp8-dynamic —— 本机实测**只有 FP8 能被 vLLM 服务**（GB10/sm_121 上 INT8 cutlass 内核直接 RuntimeError "Int8 not supported on SM121"）。int8-w8a8 保留给支持 INT8 的机器；在本机选它会产出真实 INT8 产物但 validate 阶段会被 serving 拒绝。**产出的方案一律从产出 config.json 读回，不由本参数声明。**' },
        lr: { type: 'number', description: '学习率。默认 1e-5（QAT 惯例：比 SFT 更小）。' },
        batchSize: { type: 'integer', description: '每微批样本数。默认 1。' },
        gradAccum: { type: 'integer', description: '梯度累积步数。默认 1。' },
        maxSeqLen: { type: 'integer', description: '最大序列长度。默认 256。' },
        maxSteps: { type: 'integer', description: '优化器步数上限（<=500）。默认 60。' },
        seed: { type: 'integer', description: '训练随机种子。默认 20260929。' },
        calibrationSamples: { type: 'integer', description: 'convert 步的校准样本数（>=16）。默认 64。' },
        checkpointEvery: { type: 'integer', description: '每 N 步存一个 checkpoint。默认 30。' },
        confirm: CONFIRM_PARAM,
        root: ROOT_PARAM,
      },
    },
    output: passthroughOutput(),
    handler: (a) => api.train({ ...a, root: a?.root }),
  },
  {
    name: 'training_qat_validate',
    description: 'Close the vertical loop for a QAT run with the serving path as the ONLY judging instrument: deploy the QAT quantized artifact AND the original base model through inference-serving, measure both (uniform signature, per-text mean NLL on the QAT train split AND on the held-out val split, decode throughput), then judge with @xiciyang/testing-core dualGate using caller-supplied thresholds (no built-in defaults), and run the training-eval-contract O2/O3/O5 overfit criterion by importing its single reference implementation (never re-implemented here). Also feeds the real artifact through inference-quantization\'s new inference_quantization_validate entry to prove the cross-plugin acceptance path. Deploys only its own leased endpoints and tears them down.',
    parameters: {
      type: 'object', additionalProperties: false, required: ['runId', 'thresholds'],
      properties: {
        runId: { type: 'string', description: 'training_qat_train 返回的 run id。' },
        nTrain: { type: 'integer', description: '训练切分上取多少条文本做配对测量。默认 32。' },
        nVal: { type: 'integer', description: '留出切分上取多少条文本做配对测量。默认 32。' },
        thresholds: { type: 'object', description: '{maxAccuracyDropPct} 必填；{minPerfGainPct} 可选（给了才纳入性能门禁）。**不内置默认值**。' },
        crossPluginValidate: { type: 'boolean', description: '是否调 inference-quantization 的新验收入口做跨插件对账。默认 true。' },
        dtypeConversion: {
          type: ['object', 'null'],
          description: '{from,to,declaredBy,reason} —— **显式声明**候选产物在测量前已被转换过存储 dtype。'
            + '默认 null（不声明）⇒ 基线与候选的存储 dtype 不同时**拒绝比较**（E_DTYPE_COMPARISON_REFUSED）：'
            + '实测仅把未量化张量 F32→BF16 就使 serving 的 dropPct 变化 0.40pp（1.7205%→1.3212%），'
            + '沉默地跨 dtype 比会把 dtype 效应混进 Δ。声明必须与产物对得上（to 必须等于实测 dtype），否则 E_DTYPE_CONVERSION_UNVERIFIED。',
        },
        teardown: { type: 'boolean', description: '是否拆解自己租用的端点。默认 true。' },
        confirm: CONFIRM_PARAM,
        root: ROOT_PARAM,
      },
    },
    output: passthroughOutput(),
    handler: (a) => api.validate({ ...a, root: a?.root }),
  },
  {
    name: 'training_qat_rollback',
    description: 'Roll a run back to a non-promoted state. Marks it rolled-back in the shared registry and (with purge=true) deletes ONLY the paths this run registered under the training-qat run root / export root — never the base model, never a dataset, never another run.',
    parameters: {
      type: 'object', additionalProperties: false, required: ['runId'],
      properties: { runId: { type: 'string' }, purge: { type: 'boolean', description: '是否删除该 run 的产出目录。默认 false。' }, confirm: CONFIRM_PARAM, root: ROOT_PARAM },
    },
    output: passthroughOutput(),
    handler: (a) => api.rollback({ root: a?.root, runId: a?.runId, purge: a?.purge === true }),
  },
  {
    name: 'training_qat_get_report',
    description: 'Read the QAT run registry and the audit hash chain. Without runId it lists recent runs (steps, device, engine, the scheme read back from the artifact, dropPct, gate verdict); with runId it returns the full run record plus the eval-result it produced.',
    parameters: { type: 'object', additionalProperties: false, required: [], properties: { runId: { type: 'string' }, limit: { type: 'integer' }, root: ROOT_PARAM } },
    output: passthroughOutput(),
    handler: (a) => api.getReport({ root: a?.root, runId: a?.runId ?? null, limit: a?.limit ?? 10 }),
  },
  {
    name: 'training_qat_health',
    description: 'Self-check for the QAT side: behavioural sandbox tier probe (process/container/vm), the training-qat toolchain image presence, GPU, base model availability, inference-serving MCP resolution, the inference-quantization validation entry, the overfit-criterion reference implementation, audit chain integrity, and the shared registry path. Read-only.',
    parameters: { type: 'object', additionalProperties: false, required: [], properties: { deep: { type: 'boolean' }, root: ROOT_PARAM } },
    output: passthroughOutput(),
    handler: (a) => api.health({ root: a?.root, deep: a?.deep === true }),
  },
]

export function findToolSpec(name) {
  return ALL_TOOL_SPECS.find((t) => t.name === name) || null
}
