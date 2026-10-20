/**
 * noop-guard.spec —— **产物级断言**与**方案诚实**（M3）的正面防线。
 *
 * 两件事都必须可机器判定，不能靠"看代码觉得对"：
 *
 * ① **空操作训练必须失败**：QAT 后的权重指纹与源模型**逐位相同** ⇒ `E_NO_OP_TRAINING`；
 *    指纹缺失 ⇒ 按不通过处理（`E_ARTIFACT_FINGERPRINT_MISSING`）。
 *    用假 worker（模拟容器协议语义：真写文件、真递步号）驱动**真实的 runQat 编排**，
 *    零 GPU、零容器、零框架。
 *
 * ② **scheme 只能从产出读回**：产出没有 `quantization_config` ⇒ 不可用作"已量化 checkpoint"；
 *    产出 `num_bits` 与请求不符 ⇒ `E_SCHEME_DRIFT`（**拒绝冒充**）。
 *    另断言 scheme 串与 `inference-quantization/src/quantize.mjs:71` **完全同形**。
 */
import { mkdtempSync, rmSync, writeFileSync, mkdirSync, readFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { tmpdir } from 'node:os'
import { fileURLToPath } from 'node:url'

import { runQat, readActualScheme, QAT_SCHEME, QAT_SCHEMES, DEFAULT_QAT_SCHEME } from '../../src/train.mjs'
import { artifactLicense } from '../../src/core.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const PKG = join(HERE, '..', '..')
export const NAME = 'product-assertion（空操作断言 + scheme 诚实）'

const WEIGHTS = Buffer.from('FAKE-SAFETENSORS-BYTES-' + 'x'.repeat(512))

/** 造一个标准 HF 模型目录（只要有权重文件，指纹就能算）。 */
function makeModelDir(dir, bytes) {
  mkdirSync(dir, { recursive: true })
  writeFileSync(join(dir, 'config.json'), JSON.stringify({ model_type: 'qwen2', vocab_size: 256 }))
  writeFileSync(join(dir, 'model.safetensors'), bytes)
  return dir
}

function writeQuantized(dir, { numBits = 8, strategy = 'channel', bytes = Buffer.from('QUANT-BYTES') } = {}) {
  mkdirSync(dir, { recursive: true })
  writeFileSync(join(dir, 'config.json'), JSON.stringify({
    model_type: 'qwen2',
    quantization_config: {
      quant_method: 'compressed-tensors', format: 'int-quantized', version: '0.19.0',
      config_groups: {
        group_0: {
          targets: ['Linear'],
          weights: { num_bits: numBits, type: 'int', symmetric: true, strategy, dynamic: false, group_size: null },
          input_activations: { num_bits: 8, type: 'int', symmetric: true, strategy: 'token', dynamic: true, group_size: null },
        },
      },
      ignore: ['lm_head'],
    },
  }))
  writeFileSync(join(dir, 'model.safetensors'), bytes)
  return dir
}

/**
 * 假 worker：**模拟容器协议语义**（维护自己的步计数器、真写文件），不是返回固定值的空壳。
 * @param {object} p
 * @param {string} p.runDir        宿主侧 run 目录
 * @param {Buffer} p.floatBytes     save_float 写出的字节（用与源相同的字节即可造空操作）
 * @param {object} p.convertOpts    透传给 writeQuantized（造 scheme 漂移）
 */
function fakeWorker({ runDir, floatBytes, convertOpts = {}, stepOffset = 0, failAt = -1 }) {
  let steps = 0
  const calls = []
  return {
    calls,
    async call(cmd, payload = {}) {
      calls.push(cmd)
      if (cmd === 'init') {
        return {
          ok: true, model: 'Qwen2ForCausalLM', device: 'cpu', params: 494032768, trainableParams: 494032768,
          linear: 169, rows: 128, torch: '2.11.0+cu130', cudaAvailable: false, threads: 20,
          qat: { engine: 'torchao', fakeQuantizedLinear: 169, baseConfig: 'Int8DynamicActivationIntxWeightConfig(...)' },
        }
      }
      if (cmd === 'step') {
        steps += 1
        if (steps === failAt) throw new Error('模拟容器失联')
        return { ok: true, stepIndex: steps + stepOffset, loss: 2.0 - steps * 0.01, gradNorm: 1.0, lr: 1e-5, tokens: 256, ms: 12, epoch: 0, gpuMemGiB: null }
      }
      if (cmd === 'save_float') {
        const dir = join(runDir, 'qat-float')
        mkdirSync(dir, { recursive: true })
        writeFileSync(join(dir, 'config.json'), JSON.stringify({ model_type: 'qwen2', vocab_size: 256 }))
        writeFileSync(join(dir, 'model.safetensors'), floatBytes)
        return { ok: true, dir, files: ['config.json', 'model.safetensors'], weightFiles: ['model.safetensors'], bytes: floatBytes.length, weightsSha256: 'sha256:deadbeef', step: steps }
      }
      if (cmd === 'convert') {
        const dir = join(runDir, 'quantized')
        writeQuantized(dir, convertOpts)
        return { ok: true, dir, seconds: 1.4, calibrationSamples: 32, files: ['config.json', 'model.safetensors'], weightFiles: ['model.safetensors'], bytes: 64, scheme: `${convertOpts.numBits ?? 8}bit/${convertOpts.strategy ?? 'channel'}`, toolchain: { llmcompressor: '0.14.0', compressed_tensors: '0.19.0' } }
      }
      if (cmd === 'shutdown') return { ok: true }
      throw new Error(`未知命令 ${cmd}`)
    },
    async stop() { calls.push('stop') },
  }
}

const LIMITS = { maxSteps: 5, maxEpochs: null, maxWallMs: null, maxGpuHours: 2, concurrency: 1, maxParams: 0.5e9 }

export async function run(t) {
  const tmp = mkdtempSync(join(tmpdir(), 'tq-noop-'))

  // ── ① 正常路径：权重变了 → 通过，且 scheme 从产出读回 ──
  const base1 = makeModelDir(join(tmp, 'base1'), WEIGHTS)
  const runDir1 = join(tmp, 'run1')
  const w1 = fakeWorker({ runDir: runDir1, floatBytes: Buffer.from('QAT-DIFFERENT-WEIGHTS-'.repeat(8)) })
  const r1 = await runQat({
    root: tmp, runId: 'tq_1', runDir: runDir1, baseModel: base1, datasetDir: tmp,
    config: { device: 'cpu', scheme: 'int8-w8a8', maxSteps: 3 }, limits: { ...LIMITS, maxSteps: 3 }, checkpoint: null, workerFactory: () => w1,
  })
  t.assert(r1.fingerprints.artifactAssertion.ok === true, '权重指纹不同 ⇒ 产物级断言通过')
  t.assert(r1.fingerprints.source !== r1.fingerprints.qatFloat, `源与 QAT 指纹不同（${String(r1.fingerprints.source).slice(0, 18)}… vs ${String(r1.fingerprints.qatFloat).slice(0, 18)}…）`)
  t.assert(r1.result.state.step === 3, `跑满 3 步（实际 ${r1.result.state.step}）`)
  t.assert(r1.actual.scheme === '8bit/channel', `scheme 从产出读回（实际 ${r1.actual.scheme}）`)
  t.assert(r1.actual.numBits === 8 && r1.actual.strategy === 'channel', 'scheme 明细：8bit / channel')
  t.assert(r1.actual.inputActivations?.numBits === 8 && r1.actual.inputActivations?.dynamic === true, '输入激活也是 8bit 动态（W8A8）')
  t.assert(r1.result.state.gpuHours === 0, 'CPU 路径上 gpuHours 恒为 0（不占 GPU 就不记 GPU 账）')
  t.assert(w1.calls.filter((c) => c === 'step').length === 3, '容器侧恰好被调 3 次 step')

  // scheme 串必须与 inference-quantization 的读取方式**完全同形**
  const quantSrc = readFileSync(join(PKG, '..', '..', 'inference-quantization', 'src', 'quantize.mjs'), 'utf8')
  t.assert(/config_groups\?\.group_0\?\.weights/.test(quantSrc)
    && /weights\.num_bits\}bit\/\$\{qc\.config_groups\.group_0\.weights\.strategy\}/.test(quantSrc),
    'inference-quantization 读 scheme 的方式（config_groups.group_0.weights.{num_bits,strategy}）与我们的产出形状同形（quantize.mjs:71）')

  // ── ② 空操作：QAT 后权重与源**逐位相同** ⇒ 必须失败 ──
  const base2 = makeModelDir(join(tmp, 'base2'), WEIGHTS)
  const runDir2 = join(tmp, 'run2')
  const w2 = fakeWorker({ runDir: runDir2, floatBytes: WEIGHTS })
  let threw = null
  try {
    await runQat({
      root: tmp, runId: 'tq_2', runDir: runDir2, baseModel: base2, datasetDir: tmp,
      config: { device: 'cpu', scheme: 'int8-w8a8', maxSteps: 3 }, limits: LIMITS, checkpoint: null, workerFactory: () => w2,
    })
  } catch (e) { threw = { code: e.code, message: String(e.message) } }
  t.assert(threw !== null, '空操作训练**必须失败**（权重逐位相同）')
  t.assert(threw?.code === 'E_NO_OP_TRAINING', `错误码是 E_NO_OP_TRAINING（实际 ${threw?.code}）`)
  t.assert(/逐位相同|E_NO_OP_TRAINING/.test(threw?.message ?? ''), `错误信息点明"逐位相同"（${String(threw?.message).slice(0, 90)}）`)

  // ── ③ 指纹缺失（产出没有权重文件）⇒ 按不通过处理 ──
  const base3 = makeModelDir(join(tmp, 'base3'), WEIGHTS)
  const runDir3 = join(tmp, 'run3')
  const w3 = fakeWorker({ runDir: runDir3, floatBytes: WEIGHTS })
  const origCall = w3.call.bind(w3)
  w3.call = async (cmd, payload) => {
    if (cmd === 'save_float') { mkdirSync(join(runDir3, 'qat-float'), { recursive: true }); writeFileSync(join(runDir3, 'qat-float', 'config.json'), '{}'); return { ok: true, dir: join(runDir3, 'qat-float'), files: ['config.json'] } }
    return origCall(cmd, payload)
  }
  threw = null
  try {
    await runQat({ root: tmp, runId: 'tq_3', runDir: runDir3, baseModel: base3, datasetDir: tmp, config: { device: 'cpu', scheme: 'int8-w8a8', maxSteps: 2 }, limits: LIMITS, checkpoint: null, workerFactory: () => w3 })
  } catch (e) { threw = { code: e.code, message: String(e.message) } }
  t.assert(threw?.code === 'E_ARTIFACT_FINGERPRINT_MISSING', `指纹缺失按不通过处理（实际 ${threw?.code}）`)

  // ── ④ scheme 漂移：请求 8bit、产出 4bit ⇒ 拒绝冒充 ──
  const base4 = makeModelDir(join(tmp, 'base4'), WEIGHTS)
  const runDir4 = join(tmp, 'run4')
  const w4 = fakeWorker({ runDir: runDir4, floatBytes: Buffer.from('DIFFERENT-'.repeat(16)), convertOpts: { numBits: 4, strategy: 'group' } })
  threw = null
  try {
    await runQat({ root: tmp, runId: 'tq_4', runDir: runDir4, baseModel: base4, datasetDir: tmp, config: { device: 'cpu', scheme: 'int8-w8a8', maxSteps: 2 }, limits: LIMITS, checkpoint: null, workerFactory: () => w4 })
  } catch (e) { threw = { code: e.code, message: String(e.message) } }
  t.assert(threw?.code === 'E_SCHEME_DRIFT', `请求 8bit 产出 4bit ⇒ E_SCHEME_DRIFT（实际 ${threw?.code}）`)

  // ── ⑤ 产出没有 quantization_config ⇒ 不可用作"已量化 checkpoint" ──
  const noQc = join(tmp, 'noqc')
  mkdirSync(noQc, { recursive: true })
  writeFileSync(join(noQc, 'config.json'), JSON.stringify({ model_type: 'qwen2' }))
  writeFileSync(join(noQc, 'model.safetensors'), Buffer.from('x'))
  const bad = await readActualScheme(noQc)
  t.assert(bad.ok === false && /没有 quantization_config/.test(bad.reason), '缺 quantization_config ⇒ 判定"没真的量化"（与 verifyQuantOutput 同口径）')

  const empty = await readActualScheme(join(tmp, 'nonexistent'))
  t.assert(empty.ok === false && /缺 config\.json/.test(empty.reason), '缺 config.json ⇒ 失败（不静默）')

  // ── ⑥b license 可辨识性：判据是"下游能不能据此放行"，不是"文件在不在" ──
  const licRoot = join(tmp, 'lic')
  const mk = (name, files) => { const d = join(licRoot, name); mkdirSync(d, { recursive: true }); for (const [f, c] of Object.entries(files)) writeFileSync(join(d, f), c); return d }
  t.assert(artifactLicense(mk('none', { 'config.json': '{}' })).discernible === false, '既无 LICENSE 又无 README 前注 ⇒ 不可辨识（fail-closed）')
  const unknownOnly = mk('unknown', { 'README.md': '---\nlicense: unknown\n---\n\n# x\n' })
  t.assert(artifactLicense(unknownOnly).discernible === false && artifactLicense(unknownOnly).spdx === 'unknown',
    '`license: unknown` **不算**可辨识（不把"写了个占位值"当"授权明确"）')
  t.assert(artifactLicense(mk('readme', { 'README.md': '---\nlicense: apache-2.0\n---\n\n# x\n' })).discernible === true, 'README frontmatter 的 license 可辨识')
  const apacheText = '---\nlicense: unknown\n---\n\n# x\n' + 'x'.repeat(10)
  void apacheText
  t.assert(artifactLicense(mk('licfile', { LICENSE: 'Apache License\nVersion 2.0, January 2004\n' + 'y'.repeat(2100) })).spdx === 'apache-2.0',
    'LICENSE 文本（Apache）可被归类为 apache-2.0')
  t.assert(artifactLicense(mk('cfg', { 'config.json': JSON.stringify({ license: 'mit' }) })).spdx === 'mit', 'config.json.license 优先且可辨识')

  // ── ⑤b 方案 ↔ 设备：fp8 需要 CUDA（在起容器前判掉，便宜的地方失败）──
  const base6 = makeModelDir(join(tmp, 'base6'), WEIGHTS)
  const rd6 = join(tmp, 'run6')
  threw = null
  try {
    await runQat({ root: tmp, runId: 'tq_6', runDir: rd6, baseModel: base6, datasetDir: tmp, config: { device: 'cpu', scheme: 'fp8-dynamic' }, limits: LIMITS, checkpoint: null, workerFactory: () => fakeWorker({ runDir: rd6 }) })
  } catch (e) { threw = { code: e.code, message: String(e.message) } }
  t.assert(threw?.code === 'E_MISSING_PARAM' && /需要 CUDA/.test(threw.message), 'fp8 假量化 + CPU ⇒ 起容器前就被拒（不浪费一次容器启动）')

  // ── ⑤c int 与 float 是两种量化语义：type 不符也判漂移 ──
  const base7 = makeModelDir(join(tmp, 'base7'), WEIGHTS)
  const rd7 = join(tmp, 'run7')
  threw = null
  try {
    // 假 worker 写的 config 是 type=int，而请求 fp8（float）⇒ 必须拒绝
    await runQat({ root: tmp, runId: 'tq_7', runDir: rd7, baseModel: base7, datasetDir: tmp, config: { device: 'cuda', scheme: 'fp8-dynamic' }, limits: LIMITS, checkpoint: null, workerFactory: () => fakeWorker({ runDir: rd7, floatBytes: Buffer.from('DIFF-'.repeat(16)), device: 'cuda' }) })
  } catch (e) { threw = { code: e.code, message: String(e.message) } }
  t.assert(threw?.code === 'E_SCHEME_DRIFT' && /type/.test(threw.message), '请求 float 产出 int ⇒ E_SCHEME_DRIFT')

  // ── ⑥ 本包声明的方案常量必须是 8bit/channel，且**不是** W4A16 ──
  t.assert(QAT_SCHEME.weightBits === 8 && QAT_SCHEME.weightStrategy === 'channel', '声明的方案是 8bit/channel')
  t.assert(DEFAULT_QAT_SCHEME === 'fp8-dynamic', '默认方案是 fp8-dynamic（本机**唯一可被 vLLM 服务**的方案）')
  t.assert(QAT_SCHEMES['fp8-dynamic'].servableOnThisHost === true && QAT_SCHEMES['int8-w8a8'].servableOnThisHost === false,
    '方案表如实标注"本机可服务性"：fp8 可、int8 不可（SM121 无 INT8 内核）')
  t.assert(/SM121|Int8 not supported/.test(QAT_SCHEMES['int8-w8a8'].note), 'int8 档的 note 记录了真机根因（SM121）')
  t.assert(!/W4A16/.test(JSON.stringify({ b: QAT_SCHEME.weightBits })) && /不产出、也不声明/.test(QAT_SCHEME.note), 'W4A16 被显式标注为"本包不产出、也不声明"')

  rmSync(tmp, { recursive: true, force: true })
}
