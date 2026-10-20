/**
 * train —— QAT 执行编排（**公共循环走 training-core 的单体档位**）。
 *
 * 边界（与同侧插件同一口径）：
 *   training-core 负责：循环记账、步数/墙钟/GPU 小时上限、checkpoint 编排、钩子接口、
 *                       **事后复核**（插件不守限额时如实记违规）
 *   training-qat 负责：QAT 配置、容器执行体适配、**产物级断言**、真量化(convert)编排、
 *                       指标抽取、门禁判定
 *
 * ## 两条不可省的硬判据（本文件是它们唯一的落点）
 *
 * **① 产物级断言**：`fingerprintArtifacts(源模型)` vs `fingerprintArtifacts(QAT 浮点产出)`，
 * 由 `assertArtifactChanged` 断言**必须不同**。
 * 为什么不能只看 loss：loss 是**过程量** —— 一个什么都没写盘的 run 也能打印下降的 loss。
 * 指纹相同 ⇒ `E_NO_OP_TRAINING`；指纹缺失 ⇒ `E_ARTIFACT_FINGERPRINT_MISSING`
 * （"无法断言"按**不通过**处理）。
 *
 * **② 方案诚实**：QAT 的"真量化"步产出的 `quantization_config` 由**产出自身读回**，
 * 并断言 `num_bits` 与请求一致；不一致抛 `E_SCHEME_DRIFT`。
 * 本包从不声明一个它没有产出的 scheme（M3）。
 */
import { readFileSync, existsSync, promises as fs } from 'node:fs'
import { join } from 'node:path'

import {
  createMonolithicLoop, fingerprintArtifacts, assertArtifactChanged,
  listCheckpoints, BUDGET_DEFAULTS,
} from '@xiciyang/training-core'

import { ToolError, ERROR_CODES } from './version.mjs'
import { paths, audit, digestArgs, mkdirp, contentHash } from './runtime.mjs'
import { requireContainerSandbox, sandboxProbe } from './runtime.mjs'
import { ContainerWorker, qatMounts, QAT_IMAGE } from './worker.mjs'
import { makeQatPlugin } from './qat.mjs'

/** 单次运行硬上限（长任务纪律）：≤500 步、≤2 GPU 小时、并发 1、≤0.5B 参数。 */
export const DEFAULT_LIMITS = Object.freeze({ ...BUDGET_DEFAULTS, maxSteps: 60, maxGpuHours: 2, concurrency: 1, maxParams: 0.5e9 })

/**
 * 量化方案表 —— **每一项都是"本机实测能被下游服务"的方案**，不是愿望清单。
 *
 * 本机实测（GB10 / sm_121）：
 *   - `int8-w8a8`：llmcompressor 能产出 compressed-tensors INT8，但 **vLLM 的 INT8 cutlass 内核对 SM121 直接
 *      RuntimeError: "dispatch_scaled_mm ... Int8 not supported on SM121. Use FP8 quantization instead"**
 *      ⇒ 本机**不可服务**（`[verified]`，见 TRAINING_QAT_REPORT §INT8/SM121）。
 *   - `fp8-dynamic`：llmcompressor 产出 `format: float-quantized`（weights 8bit float/channel，
 *     激活 8bit float/token/dynamic），**vLLM 在同一台机器上加载成功并做了真实前向**。
 *
 * QAT 的假量化方案必须与最终产出的方案一致（否则"训练时以为的动作"与"实际施加的量化"漂移 —— M3）。
 * 故 `fp8-dynamic` 的 QAT 用 torchao 的 float8 假量化（需 CUDA），`int8-w8a8` 用 int8 假量化（CPU 可）。
 */
export const QAT_SCHEMES = Object.freeze({
  'fp8-dynamic': Object.freeze({
    id: 'fp8-dynamic',
    llmcompressorScheme: 'FP8_DYNAMIC',
    weightBits: 8, weightType: 'float', weightStrategy: 'channel',
    activationBits: 8, activationStrategy: 'token', activationDynamic: true,
    requiresCuda: true,
    servableOnThisHost: true,
    note: 'FP8 动态量化（权重 8bit float per-channel + 激活 8bit float per-token 动态）。'
      + '本机实测 vLLM 可加载并真实前向。int8-w8a8 在本机（SM121）不可服务。'
      + 'qat.mjs:67 里写的 W4A16 **本包不产出、也不声明**（M3：不宣布一个没产出的方案）。',
  }),
  'int8-w8a8': Object.freeze({
    id: 'int8-w8a8',
    llmcompressorScheme: 'INT8',
    weightBits: 8, weightType: 'int', weightStrategy: 'channel',
    activationBits: 8, activationStrategy: 'token', activationDynamic: true,
    requiresCuda: false,
    servableOnThisHost: false,
    note: 'INT8 W8A8（权重 8bit int per-channel + 激活 8bit int per-token 动态）。'
      + '本机实测：**vLLM 的 INT8 内核不支持 SM121**（`Int8 not supported on SM121`）⇒ 产物不可被本机服务。'
      + '保留此档是为了在支持 INT8 的机器上使用；在本机选它会在 validate 阶段被 serving 拒绝。'
      + 'qat.mjs:67 里写的 W4A16 **本包不产出、也不声明**。',
  }),
})

export const DEFAULT_QAT_SCHEME = 'fp8-dynamic'

/** 当前默认方案（对外只报**从产出读回**的事实；这里只是请求口径）。 */
export const QAT_SCHEME = Object.freeze({ ...QAT_SCHEMES[DEFAULT_QAT_SCHEME] })

export function schemeSpec(id) {
  const spec = QAT_SCHEMES[id]
  if (!spec) throw new ToolError(ERROR_CODES.E_SCHEME_UNSPECIFIED, {
    hint: `未知量化方案 "${id}"；可用：${Object.keys(QAT_SCHEMES).join(', ')}`, recoverable: true,
  })
  return spec
}

/** 读回产出的实际 scheme（唯一事实来源 = 产出目录里的 config.json）。 */
export async function readActualScheme(dir) {
  const cfgPath = join(dir, 'config.json')
  if (!existsSync(cfgPath)) return { ok: false, reason: '产出缺 config.json' }
  let cfg
  try { cfg = JSON.parse(readFileSync(cfgPath, 'utf8')) } catch (e) { return { ok: false, reason: `config.json 解析失败：${e.message}` } }
  const qc = cfg.quantization_config
  if (!qc) return { ok: false, reason: 'config.json 里没有 quantization_config（说明没真的量化）' }
  const g0 = qc.config_groups?.group_0 ?? {}
  const w = g0.weights ?? {}
  let files = []
  try { files = await fs.readdir(dir) } catch { files = [] }
  const weights = files.filter((f) => /\.(safetensors|bin)$/.test(f))
  let bytes = 0
  for (const f of weights) bytes += (await fs.stat(join(dir, f))).size
  return {
    ok: weights.length > 0,
    reason: weights.length > 0 ? null : '产出目录没有权重文件',
    quantMethod: qc.quant_method ?? null,
    format: qc.format ?? null,
    numBits: w.num_bits ?? null,
    strategy: w.strategy ?? null,
    symmetric: w.symmetric ?? null,
    groupSize: w.group_size ?? null,
    type: w.type ?? null,
    inputActivations: g0.input_activations ? {
      numBits: g0.input_activations.num_bits ?? null,
      strategy: g0.input_activations.strategy ?? null,
      dynamic: g0.input_activations.dynamic ?? null,
    } : null,
    ignore: qc.ignore ?? null,
    weightFiles: weights.length,
    bytes,
    /** 与 `inference-quantization/src/quantize.mjs:71` 完全同形的 scheme 串。 */
    scheme: w.num_bits !== undefined && w.strategy !== undefined ? `${w.num_bits}bit/${w.strategy}` : null,
  }
}

/**
 * 跑一次 QAT。返回 training-core 的 run 结果 + 产物指纹 + 实际 scheme。
 */
export async function runQat({
  root, runId, runDir, baseModel, datasetDir, config,
  limits, checkpoint, onEvent = () => {}, log = () => {}, workerFactory = null,
}) {
  // 方案 ↔ 设备一致性：在**起容器之前**判掉（runQat 是真正起容器的那一层，故它也把关一次 ——
  // core.train 里还有一道更早的同判据；两处都留是有意的防御纵深，判据只有一份实现 schemeSpec）。
  const specPre = schemeSpec(config.scheme ?? DEFAULT_QAT_SCHEME)
  if (specPre.requiresCuda && (config.device ?? 'cpu') !== 'cuda') {
    throw new ToolError(ERROR_CODES.E_MISSING_PARAM, {
      hint: `量化方案 ${specPre.id} 的假量化需要 CUDA（device=${config.device ?? 'cpu'} 不满足）—— 拒绝起容器`,
      details: { scheme: specPre.id, device: config.device ?? 'cpu', requiresCuda: true }, recoverable: true,
    })
  }

  const worker = workerFactory
    ? workerFactory({ runDir, config })
    : new ContainerWorker({
      image: QAT_IMAGE,
      // CPU 路径**不带 --gpus**：不碰 GPU 就不占 GPU（本机有 GPU 串行化纪律）
      gpus: (config.device ?? 'cpu') === 'cuda',
      mounts: qatMounts({ modelDir: baseModel, datasetDir, outDir: runDir }),
      log,
    }).start()

  const plugin = makeQatPlugin({ worker, config, log })
  const events = []

  try {
    // ── 产物级断言的前一半：源模型权重指纹 ──
    const before = await fingerprintArtifacts(baseModel)
    // ⚠️ **不能用"哈希磁盘目录"**（原实现）：checkpoint 在训练**过程中**写，
    // 而此刻 `qat-float/` 还没落盘 ⇒ `fingerprintArtifacts(...).hash` 恒为 `null`
    // ⇒ payload 里 `artifactFingerprint: null`（CI `fingerprint-coverage` 实测报红的就是这个）。
    // 改成向容器要**内存指纹**（`cmd_fingerprint`，只哈希 requires_grad 参数）——任何时刻都可算。
    const fpCall = () => worker.call('fingerprint', {}, { timeoutMs: 5 * 60 * 1000 })
    Object.assign(plugin, {
      artifactFingerprint: async () => (await fpCall()).hash,
      initArtifactFingerprint: async () => (await fpCall()).initHash,
    })

    const loop = createMonolithicLoop({
      plugin,
      limits,
      checkpoint,
      onEvent: (e) => { events.push(e); onEvent(e) },
      hooks: {},
    })
    const result = await loop.run()

    // ── QAT 浮点导出（QAT 训练后的权重）──
    const floatDir = join(runDir, 'qat-float')
    const saved = await worker.call('save_float', { dir: '/out/qat-float' }, { timeoutMs: 30 * 60 * 1000 })

    // ── ① 产物级断言：QAT 训练必须真的改了权重 ──
    const after = await fingerprintArtifacts(floatDir)
    let artifactAssertion
    try {
      artifactAssertion = { ...assertArtifactChanged({ before, after, what: 'QAT 训练权重' }), ok: true }
    } catch (e) {
      artifactAssertion = { ok: false, error: String(e.message), before: before.hash, after: after.hash }
      throw new ToolError(
        String(e.message).startsWith('E_NO_OP_TRAINING') ? ERROR_CODES.E_NO_OP_TRAINING : ERROR_CODES.E_ARTIFACT_FINGERPRINT_MISSING,
        { hint: String(e.message), details: { before: before.hash, after: after.hash, floatDir }, recoverable: false })
    }

    // ── 真量化（convert）：对 QAT 训练后的权重施加 INT8 ──
    const quantDir = join(runDir, 'quantized')
    const spec = schemeSpec(config.scheme ?? DEFAULT_QAT_SCHEME)
    const conv = await worker.call('convert', {
      src: '/out/qat-float', dir: '/out/quantized',
      samples: config.calibrationSamples ?? 64, maxSeqLen: config.maxSeqLen ?? 256,
      scheme: spec.llmcompressorScheme, ignore: spec.ignore ?? ['lm_head'],
      expectBits: spec.weightBits, expectType: spec.weightType,
    }, { timeoutMs: 2 * 60 * 60 * 1000 })

    // ── 入参必须**真的生效**：convert 报回的校准样本数必须与请求一致 ──
    // 为什么值得一条硬判据：本包实测过"参数被静默丢弃"的缺陷 —— `config` 少了
    // `calibrationSamples` 这一项时，`config.calibrationSamples ?? 64` 恒定退回 64，
    // registry 记的是 256、产物却是 64 样本量的、指纹与 64 那次**逐位相同**
    // （run tq_20260929184115_ee92）。"改了参数但什么都没变"是最难发现的一类假实验，
    // 与 M3 的 scheme 漂移同族（请求 ≠ 实际）故共用 E_SCHEME_DRIFT。
    {
      const want = Number(config.calibrationSamples)
      const got = Number(conv?.calibrationSamples)
      if (Number.isFinite(want) && Number.isFinite(got) && want !== got) {
        throw new ToolError(ERROR_CODES.E_SCHEME_DRIFT, {
          hint: `convert 实际用了 ${got} 条校准样本，与请求的 ${want} 不符 —— 入参未生效，`
            + '拒绝把这次运行当作出参数的结果（否则是"改了参数、产物没变"的假实验）',
          details: { requestedCalibrationSamples: want, reportedCalibrationSamples: got }, recoverable: false,
        })
      }
    }

    // ── ② 方案诚实：从产出读回，不声明 ──
    const actual = await readActualScheme(quantDir)
    if (!actual.ok) {
      throw new ToolError(ERROR_CODES.E_SCHEME_UNSPECIFIED, {
        hint: `QAT 产出不可用作"已量化的 checkpoint"：${actual.reason}`,
        details: actual, recoverable: false,
      })
    }
    if (actual.numBits !== spec.weightBits) {
      throw new ToolError(ERROR_CODES.E_SCHEME_DRIFT, {
        hint: `产出实际 ${actual.numBits}bit，与请求的 ${spec.weightBits}bit 不符 —— 拒绝冒充`,
        details: actual, recoverable: false,
      })
    }
    if (actual.type !== spec.weightType) {
      throw new ToolError(ERROR_CODES.E_SCHEME_DRIFT, {
        hint: `产出权重 type=${actual.type}，与请求的 ${spec.weightType} 不符（int 与 float 是两种量化语义）—— 拒绝冒充`,
        details: actual, recoverable: false,
      })
    }
    const quantFingerprint = await fingerprintArtifacts(quantDir)
    const ckpts = await listCheckpoints({ dir: checkpoint?.dir ?? join(runDir, 'checkpoints') })

    // ── ③ 起点只读：源模型权重指纹在运行前后必须**逐位不变**（training-qat-contract §4 条件 4）──
    const sourceAfter = await fingerprintArtifacts(baseModel)
    const sourceUnchanged = { ok: before.hash === sourceAfter.hash, before: before.hash, after: sourceAfter.hash }

    return {
      result, events, checkpoints: ckpts, floatDir, quantDir, actual, conv,
      fingerprints: {
        source: before.hash, sourceAfter: sourceAfter.hash, sourceUnchanged,
        qatFloat: after.hash, quantized: quantFingerprint.hash, artifactAssertion,
        // 容器侧**独立**算了一遍 QAT 浮点权重的 sha256（与宿主侧 fingerprintArtifacts 是两条独立测量）。
        // 两条测量都指向"权重变了"，比单条更可信；不一致本身就是缺陷信号。
        containerQatFloatSha256: saved?.weightsSha256 ?? null,
        containerReportsWeightBytes: saved?.bytes ?? null,
      },
      workerInit: result.outcome?.init ?? null,
    }
  } finally {
    try { await worker.stop?.() } catch (e) { log(`worker 停止异常（忽略）：${e.message}`) }
  }
}

/** 少量提示：把"实际跑了什么规模"如实写进 registry（不做无声明降级的掩盖）。 */
export function runSummary({ result, actual, train }) {
  return {
    steps: result.state.step,
    wallMs: result.state.wallMs,
    gpuHours: Number((result.state.gpuHours ?? 0).toFixed(4)),
    stopReason: result.stop?.reason ?? null,
    stopChecks: result.stop?.checks ?? [],
    violations: result.violations ?? [],
    lossesTail: result.state.losses.slice(-10),
    device: train?.device ?? null,
    engine: train?.qat?.engine ?? null,
    scheme: actual.scheme,
  }
}

export { BUDGET_DEFAULTS, digestArgs, contentHash, requireContainerSandbox, sandboxProbe, paths, audit, mkdirp }
