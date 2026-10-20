/**
 * core —— training-qat 的 6 个 capability 实现（薄适配层）。
 *
 * 能力→源：
 *   prepare   → @xiciyang/data-core（去重/确定性切分/O1 双轨去泄漏/血缘）
 *   train     → @xiciyang/training-core（单体档位循环）+ 容器内 torchao QAT + llmcompressor 真量化
 *   validate  → inference-serving（部署）+ @xiciyang/testing-core（判定）
 *                + overfit-criterion.mjs（O2/O3/O5 唯一参考实现）
 *                + inference-quantization 的**新入口** inference_quantization_validate（跨插件对账）
 *   rollback / get_report / health → 状态与审计（state-core / audit-core / sandbox-core）
 */
import { readFileSync, writeFileSync, existsSync, promises as fs } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { execFileSync } from 'node:child_process'

import { BUDGET_DEFAULTS, instrumentFromDeployReport, assertInstrumentUsable, DTYPE_POLICY, readTensorDtypes, assertProductDtype, assertComparableDtype } from '@xiciyang/training-core'
import { CONTRACT_VERSION as TESTING_CORE_VERSION } from '@xiciyang/testing-core'

import { assertWeightFingerprint as sharedAssertWeightFingerprint } from '@xiciyang/training-core'

/**
 * 本包对共享守卫的**薄绑定**（只补默认 `what`）。
 *
 * 为什么留这一层而不是直接用共享函数：① 让回归测试与红演示针对**本包实际调用的入口**，
 * 而不是库里另一个东西；② 与 `training-sft` 保持同形（一个模式，两处使用）。
 */
export function assertWeightFingerprint(payload, opts = {}) {
  return sharedAssertWeightFingerprint(payload, { what: 'QAT 权重', ...opts })
}
import { ToolError, ERROR_CODES, PACKAGE_VERSION, SCHEMA_VERSION, REGISTRY_SCHEMA } from './version.mjs'
import { paths, audit, auditVerify, readRegistry, updateRegistry, sandboxProbe, requireContainerSandbox, digestArgs, mkdirp, nowId, contentHash } from './runtime.mjs'
import { runQat, DEFAULT_LIMITS, QAT_SCHEME, QAT_SCHEMES, DEFAULT_QAT_SCHEME, schemeSpec, readActualScheme } from './train.mjs'
import { QAT_IMAGE } from './worker.mjs'
import { prepareQatData, reverify } from './data/prepare.mjs'
import { evalLM, evalUniform, evalThroughput, qatGate, dropPctOf, loadCriterion, get, EVAL_PROTOCOL } from './evaluate.mjs'
import { resolveServingPath, ServingMcpClient, callWithConfirm } from './serving-client.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const PKG_ROOT = join(HERE, '..')
const WORKSPACE = process.env.DSH_WORKSPACE || join(PKG_ROOT, '..', '..')
const SCHEMA_DIR = join(process.env.INFERENCE_CONTRACTS_DIR || join(WORKSPACE, 'inference-contracts'), 'state-schema')
const SCHEMA_FILE = join(SCHEMA_DIR, 'qat-registry.schema.json')

function schemaPath() { return existsSync(SCHEMA_FILE) ? SCHEMA_FILE : null }
const P = (root) => paths(root)

async function withRun(root, runId, mutator) {
  return updateRegistry(root, (doc) => {
    const i = (doc.runs || []).findIndex((r) => r.id === runId)
    if (i < 0) throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `run 不存在：${runId}`, recoverable: true })
    const runs = [...(doc.runs || [])]
    runs[i] = mutator({ ...runs[i] })
    return { ...doc, runs }
  }, { schemaPath: schemaPath() })
}

/* ══════════════════════════════════════════════════════════ 1. prepare */

export async function prepare({ root = paths().root, sourceDir = null, sourceLicense = null, seed = 20260929, train = 128, val = 64, outDir = null, force = false, log = () => {} }) {
  const t0 = Date.now()
  await audit(root, { capability: 'prepare', action: 'prepare', phase: 'begin', ...digestArgs({ sourceDir, seed, train, val }) })
  if (!(train > 0) || !(val > 0)) throw new ToolError(ERROR_CODES.E_MISSING_PARAM, { hint: `train/val 必须为正整数（实得 train=${train} val=${val}）`, recoverable: true })
  let out
  try {
    out = await prepareQatData({ sourceDir, sourceLicense, datasetsRoot: P(root).datasetsDir, outDir, seed, train, val, force, write: true, log })
  } catch (e) {
    // license 闸门的错误码必须进**闭集**（调用方只看一套码）
    const m = /^(E_LICENSE_[A-Z_]+):/.exec(String(e.message ?? ''))
    if (m) throw new ToolError(ERROR_CODES[m[1]] ?? ERROR_CODES.E_LICENSE_UNRESOLVED, { hint: String(e.message).slice(0, 600), recoverable: true })
    throw new ToolError(ERROR_CODES.E_DATASET_INVALID, { hint: String(e.message).slice(0, 600), recoverable: true })
  }
  if (!out.allClear) {
    await audit(root, { capability: 'prepare', action: 'prepare', phase: 'end', outcome: 'leak-detected', ok: false })
    throw new ToolError(ERROR_CODES.E_DATASET_LEAK, {
      hint: `O1 去泄漏闸门未过：${JSON.stringify(out.verify.filter((v) => !v.ok))} —— 拒绝训练`,
      details: { verify: out.verify }, recoverable: false,
    })
  }
  const r = await audit(root, { capability: 'prepare', action: 'prepare', phase: 'end', outcome: 'ok', trainRows: out.trainRows, valRows: out.valRows, ms: Date.now() - t0 })
  log(`QAT 语料就绪：train=${out.trainRows} val=${out.valRows} → ${out.out}`)
  return {
    datasetDir: out.out, source: out.source, sourceFiles: out.sourceFiles, trainRows: out.trainRows, valRows: out.valRows,
    verify: out.verify, allClear: out.allClear,
    license: out.license,
    licenseGate: { spdx: out.license.spdx, class: out.license.class ?? null, trainable: out.license.class !== 'share-alike' },
    lineage: { schema: out.lineage.schema, license: out.lineage.license, allClear: out.lineage.allClear },
    audit: { seq: r.seq, hash: r.hash },
  }
}

/* ══════════════════════════════════════════════════════════════ 2. train */

export async function train({
  root = paths().root, baseModel = process.env.DSH_BASE_MODEL, datasetDir,
  device = 'cpu', scheme = DEFAULT_QAT_SCHEME, lr = 1e-5, batchSize = 1, gradAccum = 1, maxSeqLen = 256, maxSteps = 60, seed = 20260929,
  calibrationSamples = 64, limits = {}, checkpointEvery = 30, checkpointKeep = 2, confirm = null, log = () => {},
}) {
  const p = P(root)
  await mkdirp(p.runsDir)
  await mkdirp(p.exportsDir)
  const limitsResolved = { ...DEFAULT_LIMITS, ...limits, maxSteps }
  if (limitsResolved.maxSteps > 500) throw new ToolError(ERROR_CODES.E_BUDGET_EXCEEDED, { hint: `maxSteps=${limitsResolved.maxSteps} 超过单次上限 500`, recoverable: true })
  if (limitsResolved.maxGpuHours > 2) throw new ToolError(ERROR_CODES.E_BUDGET_EXCEEDED, { hint: `maxGpuHours=${limitsResolved.maxGpuHours} 超过单次上限 2`, recoverable: true })

  if (!datasetDir || !existsSync(join(datasetDir, 'train.jsonl'))) throw new ToolError(ERROR_CODES.E_DATASET_INVALID, { hint: `语料未就绪：${datasetDir}（先调 training_qat_prepare）`, recoverable: true })
  const lg = JSON.parse(readFileSync(join(datasetDir, 'lineage.json'), 'utf8'))
  if (lg.allClear !== true) throw new ToolError(ERROR_CODES.E_DATASET_LEAK, { hint: '语料血缘 allClear !== true —— 拒绝训练', recoverable: false })
  if (!existsSync(join(baseModel, 'config.json'))) throw new ToolError(ERROR_CODES.E_BASE_MODEL_NOT_FOUND, { hint: `基座模型目录不合法：${baseModel}`, recoverable: true })

  const probe = await sandboxProbe()
  requireContainerSandbox(probe)

  const runId = nowId('tq')
  const runDir = join(p.runsDir, runId)
  await mkdirp(runDir)
  const started = new Date().toISOString()
  const spec = schemeSpec(scheme)
  // 方案与设备的组合必须在**起容器之前**判掉（便宜的地方失败）：
  // torchao 的 float8 假量化在 CPU 上会 assert（requires compute capability ≥8.9）。
  if (spec.requiresCuda && device !== 'cuda') {
    throw new ToolError(ERROR_CODES.E_MISSING_PARAM, {
      hint: `量化方案 ${spec.id} 的假量化需要 CUDA（device=${device} 不满足）。`
        + '要么把 device 改成 cuda，要么改选不需要 CUDA 的方案'
        + `（可用：${Object.keys(QAT_SCHEMES).join(', ')}）。`,
      details: { scheme: spec.id, device, requiresCuda: spec.requiresCuda }, recoverable: true,
    })
  }
  const config = {
    baseModel: '/models/' + baseModel.split('/').filter(Boolean).pop(),
    trainFile: '/data/train.jsonl',
    engine: 'torchao', scheme: spec.id, weightAxis: 0,
    device, lr, weightDecay: 0.0, batchSize, gradAccum, maxSeqLen, seed, threads: null,
    // **必须显式透传**：`runQat` 的 convert 步读的是 `config.calibrationSamples`（train.mjs），
    // 少了这一行，工具入参会被静默丢弃、恒定退回 `?? 64`（缺陷：入参 no-op）。
    // 实测证据：run tq_20260929184115_ee92 传 calibrationSamples=256，registry 记为 256，
    // 但 artifacts.conversion.calibrationSamples=64 且 quantized 指纹与 64 样本的 run
    // **逐位相同**（sha256:94ce8640…）——即"改了参数但产物没变"。
    calibrationSamples,
    // 容器内的 config 是**容器的**路径；宿主侧路径另记在 registry 里
    _host: { baseModel, datasetDir, runDir },
  }
  await audit(root, { capability: 'train', action: 'train', phase: 'begin', runId, baseModel, datasetDir, device, requestedScheme: spec.id, lr, batchSize, gradAccum, maxSeqLen, maxSteps, limits: limitsResolved, image: QAT_IMAGE, declaredScheme: null })

  await updateRegistry(root, (doc) => ({
    ...doc,
    runs: [...(doc.runs || []).filter((r) => r.id !== runId), {
      id: runId, status: 'running', startedAt: started, finishedAt: null, runDir,
      baseModel: { path: baseModel, revision: readRevision(baseModel) },
      dataset: { dir: datasetDir, trainRows: lg.calib?.count ?? null, valRows: lg.test?.count ?? null, lineageHash: `sha256:${contentHash(readFileSync(join(datasetDir, 'lineage.json'), 'utf8'))}` },
      config: { device, scheme: spec.id, requestedScheme: spec.id, lr, batchSize, gradAccum, maxSeqLen, maxSteps, seed, engine: 'torchao-QATConfig', calibrationSamples },
      budget: limitsResolved,
      // **先声明"我们不知道"**：scheme 由产出读回后才填（M3：不预先宣布方案）
      scheme: null,
      auditSeqs: [],
    }],
  }), { schemaPath: schemaPath() })

  try {
    const r = await runQat({
      root, runId, runDir, baseModel, datasetDir, config, limits: limitsResolved,
      checkpoint: { dir: join(runDir, 'checkpoints'), every: checkpointEvery, keep: checkpointKeep },
      log,
    })
    const actual = r.actual
    // ── **产物级断言：权重相对起点必须真的变了**（fail-closed）──
    // 为什么必须在这里：本仓库的原始事故是 `lr=0` 让 500 步全是空操作，
    // 而**计数类断言全部为真**。qat 此前**完全没有**这条防线（payload 的
    // `artifactFingerprint` 恒为 null —— 见 train.mjs 里那段注释）。
    // 判据实现来自共享库（一个实现、三处使用），不在本包另写一份。
    if (r.checkpoints?.length) {
      const lastName = r.checkpoints[r.checkpoints.length - 1].name
      const cpPayload = JSON.parse(readFileSync(join(runDir, 'checkpoints', lastName, 'payload.json'), 'utf8'))
      assertWeightFingerprint(cpPayload, { what: 'QAT 权重' })
    }
    // 产出迁到导出根（可被 inference-quantization 直接指向的稳定路径）
    const exportDir = join(p.exportsDir, `${baseModel.split('/').filter(Boolean).pop()}-qat-${runId}`)
    await fs.rm(exportDir, { recursive: true, force: true })
    await fs.cp(r.quantDir, exportDir, { recursive: true })
    const exported = await readActualScheme(exportDir)
    // ── license 携带：派生模型必须能被下游认出 license（否则 serving 的 license 闸门会拒绝部署）──
    const lic = artifactLicense(exportDir)
    const licenseGate = {
      ...lic,
      inheritedFrom: r.conv?.license?.source ?? null,
      inheritedSpdx: r.conv?.license?.spdx ?? null,
      copiedFiles: r.conv?.license?.copiedFiles ?? [],
      note: 'license **继承自基座模型**，由容器内执行体携带到产出目录；本包不发明 license 值',
    }
    // ── 出处链（provenance）：源模型 + 其 license、训练语料 + 其 license、量化方案 ──
    const dsLineage = (() => { try { return JSON.parse(readFileSync(join(datasetDir, 'lineage.json'), 'utf8')) } catch { return null } })()
    const provenance = {
      schema: 'training-qat/provenance@1',
      generatedAt: new Date().toISOString(),
      runId,
      derivation: 'quantization-aware training (QAT): fake-quant STE training of the source model, then real INT8 quantization of the trained weights',
      source: {
        path: baseModel,
        license: { spdx: r.conv?.license?.spdx ?? null, resolved: r.conv?.license?.resolved === true, source: r.conv?.license?.source ?? null },
      },
      trainingData: {
        dir: datasetDir,
        license: dsLineage?.license
          ? { spdx: dsLineage.license.spdx, class: dsLineage.license.class ?? null, source: dsLineage.license.source ?? null, inheritedFrom: join(datasetDir, 'lineage.json') }
          : null,
        trainRows: dsLineage?.calib?.count ?? null,
        valRows: dsLineage?.test?.count ?? null,
        allClear: dsLineage?.allClear ?? null,
      },
      quantization: {
        method: 'compressed-tensors / llmcompressor QuantizationModifier(scheme="INT8")',
        scheme: exported.scheme,
        weightBits: exported.numBits, weightStrategy: exported.strategy, weightSymmetric: exported.symmetric,
        inputActivations: exported.inputActivations, format: exported.format, ignore: exported.ignore,
        toolchain: r.conv?.toolchain ?? null,
      },
      qat: { engine: r.workerInit?.qat?.engine ?? null, device, steps: r.result.state.step, seed, lr, batchSize, gradAccum, maxSeqLen },
      fingerprints: { source: r.fingerprints.source, sourceUnchanged: r.fingerprints.sourceUnchanged?.ok ?? null, qatFloat: r.fingerprints.qatFloat, quantized: r.fingerprints.quantized },
      licenseCarriedFiles: r.conv?.license?.copiedFiles ?? [],
      note: 'license 一律**继承自来源**（源模型的 LICENSE 与训练语料的 lineage.license）；本包不发明 license 值。',
    }
    writeFileSync(join(exportDir, 'provenance.json'), `${JSON.stringify(provenance, null, 2)}\n`)
    writeFileSync(join(runDir, 'provenance.json'), `${JSON.stringify(provenance, null, 2)}\n`)
    if (r.floatDir) { try { writeFileSync(join(r.floatDir, 'provenance.json'), `${JSON.stringify(provenance, null, 2)}\n`) } catch { /* 可选 */ } }

    if (!lic.discernible) {
      throw new ToolError(ERROR_CODES.E_LICENSE_UNRESOLVED, {
        hint: `QAT 产出无法被下游识别 license（${lic.source ?? '未找到 LICENSE / README frontmatter'}，spdx=${lic.spdx ?? 'null'}）`
          + '—— 这会让 inference_serving_deploy 以 E_LICENSE_UNRESOLVED 拒绝部署（本机实测）。'
          + `基座目录的 license 是否可解析：${r.conv?.license?.resolved === true ? '是（说明携带步骤失败，属本包缺陷）' : '否（**不发明 license**，如实失败）'}。`
          + 'provenance.json 已写出，出处链可查。',
        details: { quantizedDir: exportDir, convLicense: r.conv?.license ?? null, artifactLicense: lic, provenance: join(exportDir, 'provenance.json') },
        recoverable: true,
      })
    }

    const ar = await audit(root, {
      capability: 'train', action: 'train', phase: 'end', runId,
      outcome: r.result.stop?.reason ?? 'ok', steps: r.result.state.step,
      wallMs: r.result.state.wallMs, gpuHours: Number((r.result.state.gpuHours ?? 0).toFixed(4)),
      violations: (r.result.violations || []).length,
      artifactChanged: r.fingerprints.artifactAssertion.ok,
      sourceFingerprint: r.fingerprints.source, qatFloatFingerprint: r.fingerprints.qatFloat,
      scheme: exported.scheme, weightBits: exported.numBits, strategy: exported.strategy,
    })
    await withRun(root, runId, (x) => ({
      ...x, status: 'trained', finishedAt: new Date().toISOString(),
      result: {
        steps: r.result.state.step, wallMs: r.result.state.wallMs,
        gpuHours: Number((r.result.state.gpuHours ?? 0).toFixed(4)),
        stopReason: r.result.stop?.reason ?? null, stopChecks: r.result.stop?.checks ?? [],
        violations: r.result.violations ?? [], lossesTail: r.result.state.losses.slice(-10),
        device, engine: r.workerInit?.qat?.engine ?? null,
        qatInit: r.workerInit ? { params: r.workerInit.params, rows: r.workerInit.rows, qat: r.workerInit.qat, torch: r.workerInit.torch, threads: r.workerInit.threads, cudaAvailable: r.workerInit.cudaAvailable } : null,
      },
      artifacts: {
        qatFloatDir: r.floatDir, quantizedDir: exportDir, runQuantizedDir: r.quantDir,
        fingerprints: r.fingerprints,
        conversion: { seconds: r.conv.seconds, calibrationSamples: r.conv.calibrationSamples, toolchain: r.conv.toolchain },
        licenseGate,
        provenancePath: join(exportDir, 'provenance.json'),
        provenance: { source: provenance.source, trainingData: provenance.trainingData, quantization: provenance.quantization },
      },
      // **实际 scheme 从产出读回**（不是声明）
      scheme: {
        quantizationMethod: exported.quantMethod, format: exported.format,
        weightBits: exported.numBits, weightStrategy: exported.strategy, weightSymmetric: exported.symmetric,
        type: exported.type, inputActivations: exported.inputActivations, ignore: exported.ignore,
        scheme: exported.scheme, weightFiles: exported.weightFiles, bytes: exported.bytes,
        readFrom: join(exportDir, 'config.json'),
        requestedScheme: spec.id,
        note: spec.note,
      },
      checkpoint: (r.checkpoints || []).length ? { dir: join(runDir, 'checkpoints'), count: r.checkpoints.length, last: r.checkpoints[r.checkpoints.length - 1]?.name ?? null } : null,
      auditSeqs: [...(x.auditSeqs || []), ar.seq],
    }))
    return {
      runId, runDir, status: 'trained',
      steps: r.result.state.step, wallMs: r.result.state.wallMs,
      gpuHours: Number((r.result.state.gpuHours ?? 0).toFixed(4)),
      device, engine: r.workerInit?.qat?.engine ?? null,
      stopReason: r.result.stop?.reason ?? null,
      violations: r.result.violations ?? [],
      quantizedDir: exportDir, qatFloatDir: r.floatDir,
      scheme: exported.scheme, schemeDetail: { quantMethod: exported.quantMethod, weightBits: exported.numBits, weightStrategy: exported.strategy, inputActivations: exported.inputActivations },
      licenseGate,
      fingerprints: r.fingerprints,
      finalLoss: r.result.state.losses.slice(-1)[0] ?? null,
      checkpointCount: (r.checkpoints || []).length,
      audit: { seq: ar.seq, hash: ar.hash },
    }
  } catch (e) {
    const ar = await audit(root, { capability: 'train', action: 'train', phase: 'failed', runId, error: String(e.message).slice(0, 300), code: e.code ?? null }).catch(() => null)
    await withRun(root, runId, (x) => ({ ...x, status: 'failed', error: String(e.message).slice(0, 500), auditSeqs: ar ? [...(x.auditSeqs || []), ar.seq] : (x.auditSeqs || []) })).catch(() => {})
    throw e
  }
}


/**
 * 派生产物的 license 可辨识性 —— **与 inference-serving 的读取口径一致**（config.json → README
 * frontmatter → LICENSE 文本）。为什么本包要自己看一遍：本机实测，产物没有 license 时
 * `inference_serving_deploy` 会以 `E_LICENSE_UNRESOLVED` 拒绝部署 —— 那正是"下游闸门替上游发现缺陷"。
 * 上游（本包）应当在**更便宜的地方**先发现它。
 */
/** `unknown` / 空值**不算**"可辨识"—— 判据是"下游能不能据此放行"，不是"文件在不在"。 */
function isResolved(v) {
  const s = String(v ?? '').trim().toLowerCase()
  return s !== '' && s !== 'unknown' && s !== 'unknown-text' && s !== 'n/a'
}

export function artifactLicense(dir) {
  const cfgPath = join(dir, 'config.json')
  if (existsSync(cfgPath)) {
    try {
      const v = JSON.parse(readFileSync(cfgPath, 'utf8')).license
      if (typeof v === 'string' && v.trim()) return { discernible: isResolved(v), spdx: v.trim(), source: 'config.json' }
    } catch { /* 继续 */ }
  }
  const readme = join(dir, 'README.md')
  if (existsSync(readme)) {
    const m = /^---\n([\s\S]*?)\n---/.exec(readFileSync(readme, 'utf8').slice(0, 4000))
    if (m) {
      const lm = /^\s*license(?:_name)?\s*:\s*(.+)$/im.exec(m[1])
      if (lm) {
        const spdx = lm[1].trim().replace(/^['"]|['"]$/g, '')
        return { discernible: isResolved(spdx), spdx, source: 'README frontmatter' }
      }
    }
  }
  for (const f of ['LICENSE', 'LICENSE.txt', 'LICENSE.md', 'license']) {
    const p = join(dir, f)
    if (!existsSync(p)) continue
    const head = readFileSync(p, 'utf8').slice(0, 2000).toLowerCase()
    for (const [hint, spdx] of [['apache', 'apache-2.0'], ['mit license', 'mit'], ['bsd 3-clause', 'bsd-3-clause'], ['bsd 2-clause', 'bsd-2-clause']]) {
      if (head.includes(hint)) return { discernible: true, spdx, source: f }
    }
    return { discernible: false, spdx: 'unknown-text', source: f, note: 'LICENSE 文本无法归类为已知 SPDX' }
  }
  return { discernible: false, spdx: null, source: null }
}

function readRevision(modelDir) {
  try {
    const m = join(modelDir, '.cache', 'huggingface', 'download', 'config.json.metadata')
    return existsSync(m) ? readFileSync(m, 'utf8').split('\n')[0].trim() : null
  } catch { return null }
}

/* ══════════════════════════════════════════════════════════════ 3. validate */

/** 解析 inference-quantization 的 MCP 入口（**通过它的公开接口调用，不改它的源码**）。 */
export async function resolveQuantizationPath({ env = process.env, packageRoot = PKG_ROOT } = {}) {
  const tried = []
  const workspace = env.DSH_WORKSPACE || join(packageRoot, '..', '..')
  for (const cand of [env.INFERENCE_QUANTIZATION_ROOT, join(workspace, 'inference-quantization')].filter(Boolean)) {
    tried.push({ path: cand, from: cand === env.INFERENCE_QUANTIZATION_ROOT ? 'env' : 'sibling directory' })
    if (existsSync(join(cand, 'mcp', 'server.mjs'))) return { root: cand, serverPath: join(cand, 'mcp', 'server.mjs'), tried }
  }
  return { root: null, serverPath: null, tried }
}

function readJsonl(file) {
  return readFileSync(file, 'utf8').split('\n').filter((l) => l.trim()).map((l) => { try { return JSON.parse(l) } catch { return null } }).filter(Boolean)
}

async function measureAt({ endpoint, model, trainRows, valRows, nTrain, nVal, vocabSize, log }) {
  const uniform = await evalUniform({ endpoint, model, prompt: 'The quick brown fox jumps over the lazy dog. In a distant country, the population of', vocabSize })
  const train = await evalLM({ endpoint, model, rows: trainRows, limit: nTrain, log })
  const val = await evalLM({ endpoint, model, rows: valRows, limit: nVal, log })
  const perf = await evalThroughput({ endpoint, model, requests: 8, maxTokens: 32 })
  return { uniform, train, val, perf }
}

/**
 * fail-closed：**有损仪器不得用于验收**（`E_INSTRUMENT_LOSSY`）。
 *
 * 对 QAT 尤其刺眼：本包的产物**本来就已经量化过**，若 serving 还因为内存不足再叠一层
 * **有损**降级（`precision` 档：`dtype=half` + `quantization=fp8`），那测出来的 dropPct
 * 就同时包含"QAT 的精度代价"与"serving 降级的代价"两件事，却只有一个数字 —— 无法归因。
 * 判据在 `@xiciyang/training-core`（单一实现），这里只做错误码转译。
 */
function guardInstrument(instrument, runId) {
  try {
    assertInstrumentUsable(instrument, { runId })
  } catch (e) {
    if (e?.code !== ERROR_CODES.E_INSTRUMENT_LOSSY) throw e
    throw new ToolError(ERROR_CODES.E_INSTRUMENT_LOSSY, { hint: e.message, details: e.details ?? null, recoverable: false })
  }
}

const INTEGRITY_FIELDS = ['attempted', 'succeeded', 'failed', 'skipped', 'partial']
const integrityOf = (x) => {
  const out = {}
  if (!x || typeof x !== 'object') return out
  for (const k of INTEGRITY_FIELDS) if (x[k] !== undefined) out[k] = x[k]
  return out
}
const summarize = (m) => ({
  uniform: { ok: m.uniform.ok, n: m.uniform.n, criterion: m.uniform.criterion ?? null, bitIdenticalAcrossRuns: m.uniform.bitIdenticalAcrossRuns ?? null },
  // ⚠️ 手写字段表是**这个缺口反复出现的地方**（pretrain/sft/rl 都栽过）：
  // `train`/`val` 就是 `evalLM` 的产物，完整性证据必须一起搬。
  train: { ...integrityOf(m.train), n: m.train.n, nOk: m.train.nOk, texts: m.train.texts, tokens: m.train.tokens, meanNll: m.train.meanNll, ppl: m.train.ppl },
  val: { ...integrityOf(m.val), n: m.val.n, nOk: m.val.nOk, texts: m.val.texts, tokens: m.val.tokens, meanNll: m.val.meanNll, ppl: m.val.ppl },
  perf: { ...integrityOf(m.perf), requests: m.perf.requests, nOk: m.perf.nOk, outTokens: m.perf.outTokens, ms: m.perf.ms, decodeTps: m.perf.decodeTps },
})

/**
 * QAT 的精度判据 —— **纯函数**，便于无 GPU / 无容器回归测试。
 *
 * 契约 `training-qat-contract` **2.0.0 §4.1.1**（major，用户授权）：量化产物的目标是**保持**而非提升，
 * 故 O2 的"提升显著"（`improvementVal > z·SE`）对量化产物**按构造恒 false**，问的不是同一个问题。
 * 改为**单侧退化判据** `improvementVal >= -z·SE`（"退化不显著"）。
 *
 * **本函数不涉及任何阈值、不改 `z`、不改 O3。** `z` 由冻结参考实现给出，调用方原样传入。
 * 缺字段时返回 `pass: null`（**不猜**）：`null` 不等于通过。
 */
export function degradationGateFor(o2) {
  const improvementVal = o2?.improvementVal
  const zSe = o2?.zSeVal
  if (typeof improvementVal !== 'number' || typeof zSe !== 'number') {
    return { criterion: 'one-sided-non-degradation', improvementVal: improvementVal ?? null, zSe: zSe ?? null, pass: null }
  }
  return { criterion: 'one-sided-non-degradation', improvementVal, zSe, pass: improvementVal >= -zSe }
}

export async function validate({
  root = paths().root, runId, nTrain = 32, nVal = 32, thresholds = null, deploy = true, teardown = true,
  crossPluginValidate = true, dtypeConversion = null, log = () => {},
}) {
  const t0 = Date.now()
  const p = P(root)
  const reg = await readRegistry(root, schemaPath())
  const run = (reg.runs || []).find((r) => r.id === runId)
  if (!run) throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `run 不存在：${runId}`, recoverable: true })
  if (!['trained', 'validated', 'gate-failed'].includes(run.status)) {
    throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `run ${runId} 状态为 ${run.status}，不能直接 validate`, recoverable: true })
  }
  const quantDir = run.artifacts?.quantizedDir
  if (!quantDir || !existsSync(join(quantDir, 'config.json'))) {
    throw new ToolError(ERROR_CODES.E_MISSING_PARAM, { hint: `run 没有可验证的量化产出：${quantDir}`, recoverable: true })
  }
  // ① 阈值**必须由调用方显式给出**（不得内置默认 —— "自己定阈值、自己签收精度"是明确禁止的）
  const maxAccuracyDropPct = thresholds?.maxAccuracyDropPct ?? thresholds?.max_accuracy_drop_pct ?? null
  if (maxAccuracyDropPct === null || !Number.isFinite(Number(maxAccuracyDropPct))) {
    throw new ToolError(ERROR_CODES.E_THRESHOLD_NOT_PROVIDED, {
      hint: '缺少 maxAccuracyDropPct：本插件**不猜默认阈值**。请在调用里显式传 thresholds.maxAccuracyDropPct。',
      details: { received: thresholds ?? null }, recoverable: true,
    })
  }
  const minPerfGainPct = thresholds?.minPerfGainPct ?? thresholds?.min_perf_gain_pct ?? null
  const thresholdsResolved = { maxAccuracyDropPct: Number(maxAccuracyDropPct), minPerfGainPct: (minPerfGainPct === null ? null : Number(minPerfGainPct)) }

  // ② O1 现场重算（不信任 prepare 时的结论）
  const o1 = reverify(run.dataset.dir)
  if (!o1.ok) {
    await audit(root, { capability: 'validate', action: 'validate', phase: 'failed', runId, reason: 'O1-leak-gate' })
    throw new ToolError(ERROR_CODES.E_DATASET_LEAK, { hint: `O1 去泄漏闸门现场复核未过：${JSON.stringify(o1)}`, details: o1, recoverable: false })
  }

  // ③ 唯一参考实现（O2/O3/O5）—— 加载不到就 fail-closed，不自己实现
  const criterion = await loadCriterion()

  await audit(root, { capability: 'validate', action: 'validate', phase: 'begin', runId, quantizedDir: quantDir, thresholds: thresholdsResolved, nTrain, nVal })
  const trainRows = readJsonl(join(run.dataset.dir, 'train.jsonl'))
  const valRows = readJsonl(join(run.dataset.dir, 'val.jsonl'))
  if (trainRows.length === 0 || valRows.length === 0) throw new ToolError(ERROR_CODES.E_DATASET_INVALID, { hint: `语料 split 为空：${run.dataset.dir}`, recoverable: true })
  const baseCfg = JSON.parse(readFileSync(join(run.baseModel.path, 'config.json'), 'utf8'))
  const vocabSize = baseCfg.vocab_size

  if (deploy !== true) throw new ToolError(ERROR_CODES.E_MISSING_PARAM, { hint: 'validate 需要 deploy=true（判定仪器必须是 serving，training-eval-contract §5.1）', recoverable: true })

  // ── ④ 产物存储 dtype 断言（**缺陷 16**，在任何部署之前 fail-closed）──────────────
  // 为什么断言**产物**而不是流程报告：实测"仅把 QAT 产物的未量化张量 F32→BF16"就使
  // serving 测得的 dropPct 从 1.7205% 变成 1.3212%（0.40pp）。dtype 是**测量变量**，
  // 所以它必须从权重文件头读出来、必须等于策略 dtype、必须与基线可比 —— 三者缺一即拒绝。
  log(`产物 dtype 断言（策略 ${DTYPE_POLICY.evaluationProduct}）…`)
  const baseDtypes = await readTensorDtypes({ dir: run.baseModel.path })
  const candDtypes = await readTensorDtypes({ dir: quantDir })
  let dtypeAssertion = null
  try {
    dtypeAssertion = assertProductDtype({ dtypes: candDtypes, expected: DTYPE_POLICY.evaluationProduct, what: `QAT 产物 ${quantDir}` })
  } catch (e) {
    throw new ToolError(e.code === 'E_PRODUCT_DTYPE_VIOLATION' ? ERROR_CODES.E_PRODUCT_DTYPE_VIOLATION : ERROR_CODES.E_PRODUCT_DTYPE_UNKNOWN, {
      hint: String(e.message), details: { ...(e.details ?? {}), baseDtypes: baseDtypes.floatCounts }, recoverable: false,
    })
  }
  let dtypeComparability = null
  try {
    dtypeComparability = assertComparableDtype({
      baseline: { ref: run.baseModel.path, dtype: baseDtypes.floatDtype },
      candidate: { ref: quantDir, dtype: candDtypes.floatDtype },
      conversion: dtypeConversion,
    })
  } catch (e) {
    throw new ToolError(e.code === 'E_DTYPE_COMPARISON_REFUSED' ? ERROR_CODES.E_DTYPE_COMPARISON_REFUSED : ERROR_CODES.E_PRODUCT_DTYPE_UNKNOWN, {
      hint: String(e.message), details: e.details ?? {}, recoverable: false,
    })
  }
  log(`  base=${baseDtypes.floatDtype}(${baseDtypes.floatTotal} 张量) candidate=${candDtypes.floatDtype}(${candDtypes.floatTotal} 张量) ⇒ ${dtypeComparability.on}`)
  await audit(root, {
    capability: 'validate', action: 'validate', phase: 'dtype-assertion', runId,
    policy: DTYPE_POLICY.evaluationProduct, baseDtypes: baseDtypes.floatCounts, candidateDtypes: candDtypes.floatCounts,
    quantizedCounts: candDtypes.quantizedCounts, comparability: dtypeComparability, assertion: dtypeAssertion,
  })

  const sp = await resolveServingPath({ root })
  const client = await new ServingMcpClient(sp.serverPath).start()
  const results = {}
  let qatEp = null
  let baseEp = null
  // ── 仪器来源：每次用于测量的部署各留一份归一化记录（见 training-sft 同处说明）
  let candidateInstrument = null
  let baseInstrument = null
  try {
    log(`部署 QAT 量化产物 ${quantDir} …`)
    const d = await callWithConfirm(client, 'inference_serving_deploy', {
      path: quantDir, owner: 'training-qat', purpose: `QAT 垂直闭环 validate ${runId}`,
      servedModelName: `qat-${runId}`, gpuMemoryUtilization: 0.18, maxNumSeqs: 4,
    }, { log })
    if (d.ok === false) throw new ToolError(ERROR_CODES.E_SERVING_UNAVAILABLE, { hint: `QAT 产物部署失败：${d.code} ${d.message ?? ''}`, details: d })
    qatEp = { url: d.endpoint.endpoint, model: d.endpoint.servedModelName, id: d.endpoint.id, ownerToken: d.ownerToken, container: d.endpoint.containerName, imageDigest: d.endpoint.imageDigest, usability: d.report?.attempts?.[0]?.modelUsability ?? null }
    candidateInstrument = instrumentFromDeployReport(d.report)
    guardInstrument(candidateInstrument, runId) // 有损 ⇒ 立刻停：否则 dropPct 里混进"降级代价"，无法归因
    log(`  → ${qatEp.url} usability.ok=${qatEp.usability?.ok} lossy=${candidateInstrument?.lossy ?? null}`)

    log('测量 QAT 模型（serving 仪器）…')
    results.qat = await measureAt({ endpoint: qatEp.url, model: qatEp.model, trainRows, valRows, nTrain, nVal, vocabSize, log })

    log(`部署基座 ${run.baseModel.path} …`)
    const b = await callWithConfirm(client, 'inference_serving_deploy', {
      path: run.baseModel.path, owner: 'training-qat', purpose: `QAT 基线测量 ${runId}`,
      servedModelName: `base-${runId}`, gpuMemoryUtilization: 0.18, maxNumSeqs: 4,
    }, { log })
    if (b.ok === false) throw new ToolError(ERROR_CODES.E_SERVING_UNAVAILABLE, { hint: `基座部署失败：${b.code}`, details: b })
    baseEp = { url: b.endpoint.endpoint, model: b.endpoint.servedModelName, id: b.endpoint.id, ownerToken: b.ownerToken, container: b.endpoint.containerName }
    baseInstrument = instrumentFromDeployReport(b.report)
    guardInstrument(baseInstrument, runId) // 基线仪器同样不得有损：两侧仪器不同，dropPct 就不成立
    log(`测量基座 … lossy=${baseInstrument?.lossy ?? null}`)
    results.base = await measureAt({ endpoint: baseEp.url, model: baseEp.model, trainRows, valRows, nTrain, nVal, vocabSize, log })

    // ④ 精度（口径与 parity.mjs 同一公式；判据来自 testing-core.dualGate）
    const dropPct = dropPctOf({ basePpl: results.base.val.ppl, candPpl: results.qat.val.ppl })
    const nllRelPct = dropPctOf({ basePpl: results.base.val.meanNll, candPpl: results.qat.val.meanNll })
    const gainPct = (() => {
      const b2 = results.base.perf.decodeTps; const c = results.qat.perf.decodeTps
      if (!Number.isFinite(b2) || !Number.isFinite(c) || b2 === 0) return null
      return ((c - b2) / b2) * 100
    })()

    const validity = [
      { name: 'qat-forward', ok: results.qat.val.texts > 0, detail: `QAT 侧在 val 集上拿到 ${results.qat.val.texts} 条逐文本 NLL` },
      { name: 'base-forward', ok: results.base.val.texts > 0, detail: `基座侧在 val 集上拿到 ${results.base.val.texts} 条逐文本 NLL` },
      { name: 'qat-not-uniform', ok: results.qat.uniform.ok === true, detail: String(results.qat.uniform.criterion ?? results.qat.uniform.reason ?? '') },
      { name: 'base-not-uniform', ok: results.base.uniform.ok === true, detail: String(results.base.uniform.criterion ?? results.base.uniform.reason ?? '') },
      { name: 'o1-leak-gate', ok: o1.ok === true, detail: `hash 交集 ${o1.track1.hits}；子串包含 ${o1.track2.hits}` },
      // 缺陷 16：**门禁必须记录它判在哪个 dtype 上**。这条 validity 项与 `gate.dtype` 一起，
      // 让"这次判决是在什么存储 dtype 上做的"成为产物上的可审计事实（而不是隐含前提）。
      {
        name: 'product-dtype-comparable', ok: dtypeComparability.comparable === true,
        detail: `base=${dtypeComparability.base} candidate=${dtypeComparability.candidate} judgedOn=${dtypeComparability.judgedOn} on=${dtypeComparability.on}`,
      },
    ]
    const gate = qatGate({ dropPct, gainPct, validity, thresholds: thresholdsResolved })
    // 判定所依据的 dtype（含是否经过显式声明的转换）—— 进 eval-result，随判决一起留档
    gate.dtype = dtypeComparability
    // 把"性能维为什么没被判定"显式记下来（否则报告里只看到一个测得的 gain，读者会以为它被判过了）
    gate.perfJudged = Number.isFinite(thresholdsResolved.minPerfGainPct)
    gate.perfJudgedReason = gate.perfJudged
      ? `调用方声明了 minPerfGainPct=${thresholdsResolved.minPerfGainPct} ⇒ 性能维纳入判定`
      : '调用方**未声明** minPerfGainPct ⇒ 性能维不参与判定（只有实测值，没有目标；本插件不替你定目标）'

    // ⑤ O2/O3/O5：**唯一参考实现**，本包不重写
    const overfit = criterion.mod.evaluateOverfit({
      base: { train: { perText: results.base.train.perText, n: results.base.train.texts }, val: { perText: results.base.val.perText, n: results.base.val.texts } },
      model: { train: { perText: results.qat.train.perText, n: results.qat.train.texts }, val: { perText: results.qat.val.perText, n: results.qat.val.texts } },
    })
    const seeds = criterion.mod.aggregateSeeds([{ value: dropPct, se: null }])

    // ⑥ 跨插件对账：把**真实产出**喂给 inference-quantization 的"验收已有 checkpoint"新入口
    let crossPlugin = { attempted: false, reason: 'crossPluginValidate=false' }
    if (crossPluginValidate) {
      try {
        const qp = await resolveQuantizationPath({})
        if (!qp.serverPath) {
          crossPlugin = { attempted: true, ok: false, reason: `找不到 inference-quantization 的 MCP 入口（tried: ${qp.tried.map((t) => t.path).join(', ')}）` }
        } else {
          const qc2 = await new ServingMcpClient(qp.serverPath).start()
          try {
            // 注意：`allowUnverifiedImage` 是**对方插件自己的**、会进审计的显式开关。
            // 本机量化镜像尚未跑过 scan+signature 验证（`E_IMAGE_NOT_VERIFIED`），不打开它这条跨插件
            // 路径根本走不到 parity。这里打开它并把"镜像未验证"这个事实**记进结果**（不静默）。
            // ⚠️ 缺陷修正（实测）：此前这里写的是 snake_case `allow_unverified_image`，
            // 而 `inference_quantization_validate` 的 schema 只认 camelCase `allowUnverifiedImage`
            // ⇒ 该开关**被静默忽略**，跨插件路径恒定以 `E_IMAGE_NOT_VERIFIED` 失败
            // （mf06 的 registry 里 `crossPlugin.ok=false / code=E_IMAGE_NOT_VERIFIED` 即此因；
            //  而 LT4 报告里的 1.294% 来自 `tests/realmachine/cross-plugin-parity.mjs` 的手工调用，
            //  那个脚本用的正是 camelCase —— 两条路径的参数名不一致，才出现"报告说跑通、工具里跑不通"）。
            const r2 = await callWithConfirm(qc2, 'inference_quantization_validate', {
              model: run.baseModel.path, candidate: quantDir,
              max_accuracy_drop_pct: thresholdsResolved.maxAccuracyDropPct,
              ...(thresholdsResolved.minPerfGainPct === null ? {} : { min_perf_gain_pct: thresholdsResolved.minPerfGainPct }),
              allowUnverifiedImage: true,
            }, { log })
            crossPlugin = {
              attempted: true, ok: r2?.ok !== false && r2?.data?.pass === true,
              code: r2?.code ?? 'OK',
              imageVerifiedForQuantization: r2?.code === 'E_IMAGE_NOT_VERIFIED' ? false : 'allow_unverified_image=true（对方插件的显式开关，已进其审计）',
              serverPath: qp.serverPath,
              scheme: r2?.data?.scheme ?? null,
              dropPct: r2?.data?.parity?.dropPct ?? null,
              gateDecision: r2?.data?.gate?.decision ?? null,
              recognizedAsSameScheme: r2?.data?.schemeRecognized === true,
              detail: r2?.data ?? r2?.error ?? null,
            }
          } finally { await qc2.close().catch(() => {}) }
        }
      } catch (e) {
        crossPlugin = { attempted: true, ok: false, reason: `跨插件对账调用异常：${String(e.message).slice(0, 300)}` }
      }
    }

    // ⑦ 写 eval-result（符合**冻结**的 eval-result schema）
    //   `measurement.n === 0` 一律**不合法**（eval-contract §3.2：样本量为 0 的评估不合法）。
    //   这里 fail-closed：不许写出一份"什么都没测到"的结果。
    const measuredN = results.qat.val.texts + results.qat.train.texts + results.qat.uniform.n
    if (!(measuredN > 0)) {
      throw new ToolError(ERROR_CODES.E_MEASUREMENT_INVALID, {
        hint: `measurement.n = ${measuredN}：样本量为 0 的评估不合法（eval-contract §3.2）—— 拒绝写出"什么都没测到"的结果`,
        details: { base: summarize(results.base), qat: summarize(results.qat) }, recoverable: false,
      })
    }
    const evalResult = {
      schema: 'ai-infra/eval-result@1',
      subject: { kind: 'checkpoint', ref: quantDir, revision: null },
      measurement: {
        n: measuredN,
        unit: 'samples',
        datasetRef: `${run.dataset.dir}#train+val`,
        datasetHash: String(run.dataset.lineageHash ?? '').replace(/^(sha256:)+/, 'sha256:') || null,
      },
      metrics: [
        { name: 'perplexity', value: results.qat.val.ppl, unit: 'ppl', direction: 'lower-is-better', baseline: results.base.val.ppl, relativePct: dropPct },
        // `relativePct` 存在 ⟺ `baseline` 存在（eval-contract §3.3）：
        // 收益算不出来时（decodeTps 非有限/为 0）**两边一起置 null**，不留下"有基线无相对量"的半截记录。
        {
          name: 'decode_throughput',
          value: results.qat.perf.decodeTps,
          unit: 'tok/s',
          direction: 'higher-is-better',
          baseline: gainPct === null ? null : results.base.perf.decodeTps,
          relativePct: gainPct,
        },
        { name: 'mean_nll_val', value: results.qat.val.meanNll, unit: 'nats', direction: 'lower-is-better', baseline: results.base.val.meanNll, relativePct: nllRelPct },
      ],
      validity,
      gate: { pass: gate.pass, validityOk: gate.validityOk, reason: gate.reason, checks: gate.checks, dtype: gate.dtype },
      versions: { testingCore: TESTING_CORE_VERSION, assets: EVAL_PROTOCOL },
      // 仪器来源（**additive**，schema 里是可选属性）：判决是在哪两次部署上测出来的、有没有有损降级。
      // 与下面 `trainSpecific.instrument: 'serving'`（字符串，指"判定仪器是 serving"）不是同一件事。
      instrument: { base: baseInstrument, candidate: candidateInstrument },
      trainSpecific: {
        instrument: 'serving',
        // 缺陷 16：产物存储 dtype（策略 / 实测 / 可比性判定 / 是否经显式声明的转换）
        productDtype: {
          policy: DTYPE_POLICY.evaluationProduct,
          why: DTYPE_POLICY.why,
          base: { ref: run.baseModel.path, dtype: baseDtypes.floatDtype, floatCounts: baseDtypes.floatCounts },
          candidate: { ref: quantDir, dtype: candDtypes.floatDtype, floatCounts: candDtypes.floatCounts, quantizedCounts: candDtypes.quantizedCounts },
          comparability: dtypeComparability,
          assertion: dtypeAssertion,
        },
        thresholds: thresholdsResolved,
        dropPct,
        gainPct,
        scheme: run.scheme?.scheme ?? null,
        leakage: o1,
        overfit: { ...overfit, criterionModule: criterion.path },
        seeds: { ...seeds, singleSeed: true, note: '本次只跑 1 个种子 ⇒ **不得**表述为"该流程"有该效应（training-eval-contract §4.5）' },
        selectedCheckpoint: { name: run.checkpoint?.last ?? null, step: run.result?.steps ?? null, payloadHash: null, selectionRule: 'single-final-checkpoint', candidates: run.checkpoint?.count ?? 0 },
        selectionOnly: false,
        crossPlugin,
      },
    }
    // 自检：指标记录的 `relativePct ⟺ baseline` 一致性（eval-contract §3.3）。
    // 这条规则**自己也要守**，不能只在别人的检查里被守。
    const incoherent = evalResult.metrics.filter((m) => ((m.relativePct === null || m.relativePct === undefined) !== (m.baseline === null || m.baseline === undefined)))
    if (incoherent.length > 0) {
      throw new ToolError(ERROR_CODES.E_INTERNAL, {
        hint: `eval-result 的指标记录不自洽（relativePct 与 baseline 的存在性必须一致）：${JSON.stringify(incoherent)}`,
        recoverable: false,
      })
    }
    // ── 裁决 B（major，training-qat-contract 2.0.0 §4.1）────────────────────────
    // O2 的原始口径是「提升显著」(`improvementVal > z·SE`)，它面向**以提升为目标**的训练。
    // 量化产物的目标是**保持**，不是提升 ⇒ 在它身上 O2 按构造恒为 false（与配置无关），
    // 会把"量化没把模型搞坏"也判成失败。改为**单侧退化判据** `improvementVal >= -z·SE`（"退化不显著"）。
    // **本改动不触碰任何阈值、不触碰 z、不触碰 O3**；原 O2 结果**原样保留**（可追溯、可回滚）。
    const o2r = overfit.o2_generalizationReal ?? {}
    const dg = degradationGateFor(o2r)
    const degradationOk = dg.pass
    evalResult.trainSpecific.degradationGate = {
      ...dg,
      basis: 'improvementVal >= -z·SE（z 取自冻结参考实现 checks/overfit-criterion.mjs，未改）',
    }
    evalResult.trainSpecific.overfitJudgement = {
      note: '契约 2.0.0 起：QAT 用 degradationGate（单侧"退化不显著"）判定。overfit.pass（O2"提升显著"）对量化产物按构造恒 false，仅作可追溯保留，**不再参与晋级**。',
      legacyO2Pass: overfit.pass === true,
      legacyO2: { improvementVal: o2r.improvementVal ?? null, zSe: o2r.zSeVal ?? null },
      degradationPass: degradationOk,
      o3Pass: overfit.o3_overfitBudget?.pass === true,
    }
    writeFileSync(join(p.runsDir, runId, 'eval-result.json'), `${JSON.stringify(evalResult, null, 2)}\n`)

    const pass = gate.pass && degradationOk === true && overfit.o3_overfitBudget?.pass === true
    const ar = await audit(root, {
      capability: 'validate', action: 'validate', phase: 'end', runId, outcome: pass ? 'pass' : 'fail',
      accuracyGate: gate.pass, validityOk: gate.validityOk, dropPct, gainPct,
      overfitPass: overfit.pass === true, overfitRatio: overfit.o3_overfitBudget?.ratio ?? null,
      // 裁决 B：晋级依据已从 O2（提升显著）改为**单侧退化不显著**；两者都入审计，便于事后复核口径。
      degradationPass: degradationOk === true, legacyO2Pass: overfit.pass === true,
      o3Pass: overfit.o3_overfitBudget?.pass === true,
      o1Ok: o1.ok, crossPluginOk: crossPlugin.ok === true,
      basePpl: results.base.val.ppl, qatPpl: results.qat.val.ppl, ms: Date.now() - t0,
    })
    await updateRegistry(root, (doc) => ({
      ...doc,
      runs: (doc.runs || []).map((r) => (r.id === runId ? {
        ...r, status: pass ? 'validated' : 'gate-failed', validatedAt: new Date().toISOString(),
        endpoint: qatEp,
        evaluation: { thresholds: thresholdsResolved, gate, dropPct, gainPct, overfitPass: overfit.pass === true, base: summarize(results.base), qat: summarize(results.qat), crossPlugin },
        evalResult: { schema: evalResult.schema, path: join(p.runsDir, runId, 'eval-result.json') },
        decidedBudget: { gpuHoursTrain: r.result?.gpuHours ?? null, gpuHoursValidate: Number(((Date.now() - t0) / 3_600_000).toFixed(4)) },
        auditSeqs: [...(r.auditSeqs || []), ar.seq],
      } : r)),
    }), { schemaPath: schemaPath() })

    return {
      runId, pass, gate, thresholds: thresholdsResolved, dropPct, gainPct,
      overfit: { pass: overfit.pass, o2: overfit.o2_generalizationReal, o3: overfit.o3_overfitBudget, sigmaBase: overfit.sigmaBase, gap: overfit.gap, improvementVal: overfit.improvementVal, n: overfit.n },
      leakage: o1, crossPlugin,
      dtype: { policy: DTYPE_POLICY.evaluationProduct, base: baseDtypes.floatDtype, candidate: candDtypes.floatDtype, comparability: dtypeComparability },
      endpoint: { url: qatEp.url, model: qatEp.model, id: qatEp.id, usabilityOk: qatEp.usability?.ok ?? null, imageDigest: qatEp.imageDigest ?? null },
      base: summarize(results.base), qat: summarize(results.qat),
      evalResultPath: join(p.runsDir, runId, 'eval-result.json'),
      audit: { seq: ar.seq, hash: ar.hash }, ms: Date.now() - t0,
    }
  } finally {
    if (teardown) {
      for (const [label, ep] of [['qat', qatEp], ['base', baseEp]]) {
        if (!ep) continue
        try {
          const r = await callWithConfirm(client, 'inference_serving_teardown', { selector: ep.id, ownerToken: ep.ownerToken }, { log })
          log(`  teardown ${label}: ok=${r.ok} ${r.code ?? ''}`)
        } catch (e) { log(`  teardown ${label} 异常：${e.message}`) }
      }
    }
    await client?.close?.().catch(() => {})
  }
}

/* ══════════════════════════════════════════════════════════ 4. rollback */

export async function rollback({ root = paths().root, runId, purge = false, log = () => {} }) {
  const reg = await readRegistry(root, schemaPath())
  const run = (reg.runs || []).find((r) => r.id === runId)
  if (!run) throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `run 不存在：${runId}`, recoverable: true })
  await audit(root, { capability: 'rollback', action: 'rollback', phase: 'begin', runId, purge })
  const purged = []
  if (purge) {
    // 只删**本 run 登记的路径**，且必须落在本包的 run 根 / 导出根之下；
    // 绝不碰源模型（源目录不在允许前缀内，且显式排除）。
    const runRoot = join(paths(root).runsDir, runId)
    const allowed = [runRoot]
    if (run.artifacts?.quantizedDir && run.artifacts.quantizedDir.startsWith(paths(root).exportsDir)) allowed.push(run.artifacts.quantizedDir)
    for (const d of allowed) {
      const under = d === runRoot || d.startsWith(paths(root).exportsDir)
      const isBase = d === run.baseModel?.path || String(run.baseModel?.path ?? '').startsWith(`${d}/`)
      if (d && under && !isBase && existsSync(d)) {
        await fs.rm(d, { recursive: true, force: true })
        purged.push(d)
      }
    }
  }
  const ar = await audit(root, { capability: 'rollback', action: 'rollback', phase: 'end', runId, purged: purged.length })
  await withRun(root, runId, (r) => ({ ...r, status: 'rolled-back', rolledBackAt: new Date().toISOString(), purged, auditSeqs: [...(r.auditSeqs || []), ar.seq] }))
  log(`run ${runId} → rolled-back（purge=${purge}, 删除 ${purged.length} 个目录）`)
  return { runId, status: 'rolled-back', purged, audit: { seq: ar.seq, hash: ar.hash } }
}

/* ══════════════════════════════════════════════════════ 5. get_report */

export async function getReport({ root = paths().root, runId = null, limit = 10 }) {
  const reg = await readRegistry(root, schemaPath())
  const chain = await auditVerify(root)
  if (!runId) {
    return {
      runs: (reg.runs || []).slice(-limit).map((r) => ({
        id: r.id, status: r.status, startedAt: r.startedAt,
        steps: r.result?.steps ?? null, gpuHours: r.result?.gpuHours ?? null,
        device: r.result?.device ?? null, engine: r.result?.engine ?? null,
        scheme: r.scheme?.scheme ?? null,
        dropPct: r.evaluation?.dropPct ?? null, pass: r.evaluation ? (r.evaluation.gate?.pass === true && r.evaluation.overfitPass === true) : null,
        quantizedDir: r.artifacts?.quantizedDir ?? null,
      })),
      audit: { ok: chain.ok, shards: chain.shards },
      schema: REGISTRY_SCHEMA,
    }
  }
  const run = (reg.runs || []).find((r) => r.id === runId)
  if (!run) throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `run 不存在：${runId}`, recoverable: true })
  let evalResult = null
  const ep = join(paths(root).runsDir, runId, 'eval-result.json')
  if (existsSync(ep)) { try { evalResult = JSON.parse(readFileSync(ep, 'utf8')) } catch { evalResult = null } }
  return { run, evalResult, audit: { ok: chain.ok, shards: chain.shards, results: (chain.results || []).map((x) => ({ day: x.day, checked: x.checked, ok: x.ok })) } }
}

/* ══════════════════════════════════════════════════════════════ 6. health */

export async function health({ root = paths().root, deep = false }) {
  const p = P(root)
  const probe = await sandboxProbe()
  const chain = await auditVerify(root)
  let gpu = null
  try {
    gpu = execFileSync('nvidia-smi', ['--query-gpu=name,driver_version,memory.used,memory.total', '--format=csv,noheader'], { encoding: 'utf8', timeout: 20000 }).trim()
  } catch { gpu = null }
  let imageOk = false
  let imageId = null
  try {
    imageId = execFileSync('docker', ['images', '-q', QAT_IMAGE], { encoding: 'utf8', timeout: 20000 }).trim() || null
    imageOk = Boolean(imageId)
  } catch { imageOk = false }
  const baseOk = process.env.DSH_BASE_MODEL ? existsSync(join(process.env.DSH_BASE_MODEL, 'config.json')) : false
  let servingOk = false
  let servingPath = null
  try { const sp = await resolveServingPath({ root }); servingPath = sp.serverPath; servingOk = true } catch { servingOk = false }
  const qp = await resolveQuantizationPath({})
  let criterion = null
  try { criterion = (await loadCriterion()).path } catch (e) { criterion = { error: String(e.message).slice(0, 200) } }
  const reg = await readRegistry(root, schemaPath())
  const c = (chain.results || [])
  return {
    ok: probe.container.available && chain.ok && imageOk && baseOk && servingOk,
    sandboxTiers: { process: probe.process.available, container: probe.container.available, vm: probe.vm.available },
    sandboxEvidence: probe.container.evidence,
    image: { name: QAT_IMAGE, available: imageOk, id: imageId },
    gpu,
    baseModel: { path: process.env.DSH_BASE_MODEL ?? null, available: baseOk },
    serving: { available: servingOk, serverPath: servingPath },
    quantizationEntry: { available: Boolean(qp.serverPath), serverPath: qp.serverPath },
    criterion,
    declaredScheme: QAT_SCHEME,
    audit: { ok: chain.ok, shards: chain.shards, detail: c.map((x) => ({ day: x.day, checked: x.checked, ok: x.ok })) },
    registry: { runs: (reg.runs || []).length, file: p.registryFile },
    limits: DEFAULT_LIMITS,
    version: { pkg: PACKAGE_VERSION, schema: SCHEMA_VERSION },
    ...(deep ? { deep: { paths: p, env: { TRAINING_QAT_ROOT: process.env.TRAINING_QAT_ROOT ?? null, DSH_WORKSPACE: process.env.DSH_WORKSPACE ?? null } } } : {}),
  }
}

export { BUDGET_DEFAULTS, QAT_SCHEME, QAT_SCHEMES, DEFAULT_QAT_SCHEME, DEFAULT_LIMITS, QAT_IMAGE }
