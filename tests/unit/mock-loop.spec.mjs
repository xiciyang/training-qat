/**
 * mock-loop.spec —— 在 CI 里用**模拟容器协议语义**的假 worker 驱动
 * `training-core` 的**单体档位**循环（`createMonolithicLoop`），不真跑训练。
 *
 * 这个 spec 的价值在于它能在**零 GPU / 零容器 / 零框架**下验证：
 *   - 单体档位接线正确（`ctx.onStep` 每优化器步被调一次、步数记账）
 *   - **两侧步号一致性不变量**（故意让假 worker 漏一步 → 必须失败）
 *   - 资源上限真的会停（maxSteps），且超限策略是 `pause-and-report`（不降级）
 *   - **事后复核**：插件无视 stop 继续跑 ⇒ core 记违规并如实记账实际步数
 *   - checkpoint 编排（每 N 步存、meta 里有 payloadHash）
 *   - **GPU 小时的诚实记账**：device=cpu ⇒ gpuHours 恒为 0；device=cuda ⇒ 累加实测墙钟
 */
import { mkdtempSync, rmSync, existsSync, mkdirSync, writeFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { tmpdir } from 'node:os'
import { fileURLToPath } from 'node:url'

import { runQat } from '../../src/train.mjs'
import { listCheckpoints, loadCheckpoint } from '@xiciyang/training-core'

const HERE = dirname(fileURLToPath(import.meta.url))
export const NAME = 'mock-loop（CI 里用假 worker 驱动单体档位循环）'

function makeModelDir(dir) {
  mkdirSync(dir, { recursive: true })
  writeFileSync(join(dir, 'config.json'), '{"model_type":"qwen2"}')
  writeFileSync(join(dir, 'model.safetensors'), Buffer.from('BASE-' + 'b'.repeat(256)))
  return dir
}

function writeQuant(dir, n = 1) {
  mkdirSync(dir, { recursive: true })
  writeFileSync(join(dir, 'config.json'), JSON.stringify({
    model_type: 'qwen2',
    quantization_config: { quant_method: 'compressed-tensors', format: 'int-quantized', config_groups: { group_0: { weights: { num_bits: 8, strategy: 'channel', type: 'int' }, input_activations: { num_bits: 8, strategy: 'token', dynamic: true } } }, ignore: ['lm_head'] },
  }))
  writeFileSync(join(dir, 'model.safetensors'), Buffer.from(`Q-${n}-` + 'q'.repeat(200)))
}

/** 假 worker：模拟容器协议语义（维护自己的步计数器、真写文件）。 */
function fakeWorker({ runDir, device = 'cpu', stepOffset = 0, failAt = -1 }) {
  let steps = 0
  const calls = []
  return {
    calls,
    async call(cmd) {
      calls.push(cmd)
      if (cmd === 'init') {
        return { ok: true, model: 'Qwen2ForCausalLM', device, params: 494032768, trainableParams: 494032768, linear: 169, rows: 128, torch: '2.11.0+cu130', cudaAvailable: device === 'cuda', threads: 20, qat: { engine: 'torchao', fakeQuantizedLinear: 169 } }
      }
      // ⚠️ 协议契约：假 worker 必须**与真容器同一个命令面**。
      // 实测踩到：加了 `cmd_fingerprint` 之后没同步这个 mock ⇒ mock-loop spec 直接抛
      // "未知命令 fingerprint"（**这正是 mock 该有的行为**：它把"协议面漂移"在 CI 里逮住了）。
      // `hash` 随 step 变化 ⇒ 空操作断言在 mock 下**可判定**（不变才是空操作）。
      if (cmd === 'fingerprint') {
        const h = `sha256:${String(steps).padStart(8, '0')}`
        return { ok: true, hash: steps === 0 ? 'sha256:00000000' : h, initHash: 'sha256:00000000' }
      }
      if (cmd === 'step') {
        steps += 1
        if (steps === failAt) throw new Error('模拟容器失联')
        await new Promise((r) => setTimeout(r, 6)) // 造出可观测的墙钟：让 CPU/CUDA 的记账差异可判定
        return { ok: true, stepIndex: steps + stepOffset, loss: 3.0 - steps * 0.01, gradNorm: 1.0, lr: 1e-5, tokens: 256, ms: 5, epoch: 0, gpuMemGiB: device === 'cuda' ? 2.5 : null }
      }
      if (cmd === 'save_float') {
        const dir = join(runDir, 'qat-float')
        mkdirSync(dir, { recursive: true })
        writeFileSync(join(dir, 'config.json'), '{"model_type":"qwen2"}')
        writeFileSync(join(dir, 'model.safetensors'), Buffer.from(`TRAINED-${steps}-` + 't'.repeat(200)))
        return { ok: true, dir, files: ['config.json', 'model.safetensors'], weightFiles: ['model.safetensors'], bytes: 300, step: steps }
      }
      if (cmd === 'convert') { const dir = join(runDir, 'quantized'); writeQuant(dir, steps); return { ok: true, dir, seconds: 0.5, calibrationSamples: 32, files: ['config.json', 'model.safetensors'], weightFiles: ['model.safetensors'], bytes: 200, toolchain: {} } }
      if (cmd === 'shutdown') return { ok: true }
      throw new Error(`未知命令 ${cmd}`)
    },
    async stop() { calls.push('stop') },
  }
}

const L = (o = {}) => ({ maxSteps: 10, maxEpochs: null, maxWallMs: null, maxGpuHours: 2, concurrency: 1, maxParams: 0.5e9, ...o })

export async function run(t) {
  const tmp = mkdtempSync(join(tmpdir(), 'tq-loop-'))

  // ── ① 跑满 + 上限生效 ──
  const base1 = makeModelDir(join(tmp, 'b1'))
  const rd1 = join(tmp, 'r1')
  const w1 = fakeWorker({ runDir: rd1 })
  const r1 = await runQat({ root: tmp, runId: 'tq_a', runDir: rd1, baseModel: base1, datasetDir: tmp, config: { device: 'cpu', scheme: 'int8-w8a8' }, limits: L({ maxSteps: 7 }), checkpoint: { dir: join(rd1, 'checkpoints'), every: 3, keep: 2 }, workerFactory: () => w1 })
  t.assert(r1.result.state.step === 7, `跑到 maxSteps 即停（实际 ${r1.result.state.step}）`)
  t.assert(r1.result.stop.reason === 'budget-exhausted:maxSteps', `停止原因正确（${r1.result.stop.reason}）`)
  t.assert(r1.result.stop.policy === 'pause-and-report', '超限策略=暂停并报告（不降级）')
  t.assert(w1.calls.filter((c) => c === 'step').length === 7, '容器侧恰好被调 7 次 step')
  t.assert(r1.result.state.losses.length === 7, 'loss 逐笔记账')
  t.assert(r1.result.state.mode === 'monolithic', '记账里标明档位=monolithic')
  t.assert(r1.result.state.gpuHours === 0, `device=cpu ⇒ gpuHours 恒为 0（实际 ${r1.result.state.gpuHours}）`)
  // 恰好跑到上限时 core 会在**事后核对**里记一条 maxSteps 违规（"命中限额"就是一个事实，如实记账）
  t.assert(r1.result.violations.length === 1 && r1.result.violations[0].limit === 'maxSteps' && r1.result.violations[0].phase === 'post-run',
    `命中上限被如实记账为 post-run 违规（实际 ${JSON.stringify(r1.result.violations)}）`)
  t.assert(r1.result.state.wallMs > 0, `墙钟被记账（实际 ${r1.result.state.wallMs}ms）—— 与 gpuHours 是两个独立的量`)
  t.assert(r1.result.state.gpuHours === 0, 'CPU 路径：墙钟 > 0 但 gpuHours 仍恒为 0（不占 GPU 就不记 GPU 账）')

  // ── ② checkpoint 编排（单体档位同样受 checkpoint-contract 约束）──
  const ck = await listCheckpoints({ dir: join(rd1, 'checkpoints') })
  t.assert(ck.length === 2, `keep=2 生效（实际保留 ${ck.length}）`)
  const loaded = await loadCheckpoint({ dir: join(rd1, 'checkpoints'), name: ck[0].name })
  t.assert(loaded.meta.payloadHash === loaded.hash, 'meta.payloadHash 与复核结果一致')

  // ── ③ 步号一致性不变量（故意让容器漏一步 → 必须失败）──
  const base2 = makeModelDir(join(tmp, 'b2'))
  const rd2 = join(tmp, 'r2')
  let threw = null
  try {
    await runQat({ root: tmp, runId: 'tq_b', runDir: rd2, baseModel: base2, datasetDir: tmp, config: { device: 'cpu', scheme: 'int8-w8a8' }, limits: L({ maxSteps: 5 }), checkpoint: null, workerFactory: () => fakeWorker({ runDir: rd2, stepOffset: -1 }) })
  } catch (e) { threw = String(e.message) }
  t.assert(threw && /失步/.test(threw), `步号失步被检出并停止（实际：${String(threw).slice(0, 90)}）`)

  // ── ④ 容器失联 → 响亮失败（不产生"看起来跑完了"的产物）──
  const base3 = makeModelDir(join(tmp, 'b3'))
  const rd3 = join(tmp, 'r3')
  threw = null
  try {
    await runQat({ root: tmp, runId: 'tq_c', runDir: rd3, baseModel: base3, datasetDir: tmp, config: { device: 'cpu', scheme: 'int8-w8a8' }, limits: L({ maxSteps: 5 }), checkpoint: null, workerFactory: () => fakeWorker({ runDir: rd3, failAt: 3 }) })
  } catch (e) { threw = String(e.message) }
  t.assert(threw && /模拟容器失联/.test(threw), '容器中途失联 → 异常向上抛（不静默截断）')
  t.assert(!existsSync(join(rd3, 'quantized')), '失联的 run 没有留下 quantized 产出（不产生"看起来跑完了"的产物）')

  // ── ⑤ 预算闸门：maxGpuHours=0 让 core 在第一步就命中限额（插件读到 gate.stop 后停）──
  const base4 = makeModelDir(join(tmp, 'b4'))
  const rd4 = join(tmp, 'r4')
  const w4 = fakeWorker({ runDir: rd4 })
  const r4 = await runQat({
    root: tmp, runId: 'tq_d', runDir: rd4, baseModel: base4, datasetDir: tmp,
    config: { device: 'cpu', scheme: 'int8-w8a8' }, limits: L({ maxSteps: 4, maxGpuHours: 0 }), checkpoint: null, workerFactory: () => w4,
  })
  t.assert(r4.result.state.step <= 1, `maxGpuHours=0 ⇒ 第一步后即停（实际 ${r4.result.state.step} 步）`)
  t.assert(r4.result.stop.reason === 'budget-exhausted:maxGpuHours', `停止原因=预算耗尽（${r4.result.stop.reason}）`)

  // ── ⑥ GPU 路径的记账：device=cuda ⇒ gpuHours > 0（单卡并发 1 ⇒ 墙钟≈GPU 时间）──
  const base5 = makeModelDir(join(tmp, 'b5'))
  const rd5 = join(tmp, 'r5')
  const r5 = await runQat({ root: tmp, runId: 'tq_e', runDir: rd5, baseModel: base5, datasetDir: tmp, config: { device: 'cuda', scheme: 'int8-w8a8' }, limits: L({ maxSteps: 3 }), checkpoint: null, workerFactory: () => fakeWorker({ runDir: rd5, device: 'cuda' }) })
  t.assert(r5.result.state.gpuHours > 0, `device=cuda ⇒ gpuHours 被累加（实际 ${r5.result.state.gpuHours}）`)
  t.assert(r5.result.state.gpuHours < 0.01, '3 步的"GPU 小时"不可能大到不真实（记账没有量纲错误）')

  rmSync(tmp, { recursive: true, force: true })
  t.assert(!existsSync(tmp), '测试目录已清理（零残留）')
}
