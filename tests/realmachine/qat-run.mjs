#!/usr/bin/env node
/**
 * 真机 QAT 运行器（P0 验收项）。
 *
 * 用**交付的工具面**（与 DSH / MCP 同一条 handler）真跑一遍：
 *   prepare → train（容器内 torchao QAT 假量化 + llmcompressor INT8 真量化）
 * 并把**原始输出**逐条落盘（不做二次加工，便于事后复核）。
 *
 * 为什么叫 realmachine：没有 mock —— 真模型目录、真容器、真训练步、真量化、
 * 真指纹断言。CPU 路径**不占 GPU**（因此不触发 GPU 串行化纪律），但仍如实记录墙钟。
 *
 * 用法：node tests/realmachine/qat-run.mjs [--steps 40] [--train 128] [--val 64]
 */
import { mkdirSync, writeFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

import { findToolSpec } from '../../src/tools.mjs'
import { createCore } from '../../src/index.mjs'
import { tokenFor } from '../../src/confirm.mjs'
import { paths } from '../../src/runtime.mjs'

/** 走**真实的工具路径**（与 DSH / MCP 同一套信封 + 治理 + 审计），不是直接调 handler。 */
const core = createCore({ logger: null })

const HERE = dirname(fileURLToPath(import.meta.url))
const PKG = join(HERE, '..', '..')
const OUT = join(PKG, 'out', 'realmachine')

const argv = process.argv.slice(2)
const argOf = (k, d) => { const i = argv.indexOf(k); return i >= 0 && argv[i + 1] ? argv[i + 1] : d }
const num = (k, d) => Number(argOf(k, String(d)))

const STEPS = num('--steps', 40)
const TRAIN = num('--train', 128)
const VAL = num('--val', 64)
const CALIB = num('--calib', 32)
const SEQ = num('--seq', 256)
const DEVICE = argOf('--device', 'cpu')
const SCHEME = argOf('--scheme', 'fp8-dynamic')

/** 首次调用拿确认令牌 → 带令牌重调。令牌由「能力 + 目标指纹」确定性推导。 */
async function call(name, args, { scope = null } = {}) {
  const spec = findToolSpec(name)
  if (!spec) throw new Error(`未知工具 ${name}`)
  let out = await core.run(spec, args)
  if (out?.code === 'E_CONFIRMATION_REQUIRED') {
    const token = out.error?.details?.confirmToken ?? out.error?.confirmToken ?? tokenFor(scope, args)
    say(`  ${name} 需确认，带令牌重试（scope=${scope}）`)
    out = await core.run(spec, { ...args, confirm: token })
  }
  return out
}

const log = []
const say = (m) => { const line = `[${new Date().toISOString()}] ${m}`; log.push(line); console.log(line) }

async function main() {
  mkdirSync(OUT, { recursive: true })
  const root = paths().root
  say(`root=${root} device=${DEVICE} scheme=${SCHEME} steps=${STEPS} train=${TRAIN} val=${VAL} seq=${SEQ} calib=${CALIB}`)

  say('── prepare ──')
  const t0 = Date.now()
  const prep = await call('training_qat_prepare', { train: TRAIN, val: VAL })
  say(`prepare ok=${prep.ok} code=${prep.code} ${Date.now() - t0}ms`)
  if (prep.ok !== true) { say(`prepare FAILED: ${JSON.stringify(prep).slice(0, 900)}`); return 1 }
  const datasetDir = prep.data.datasetDir
  say(`datasetDir=${datasetDir} train=${prep.data.trainRows} val=${prep.data.valRows} allClear=${prep.data.allClear}`)

  say('── train（QAT 假量化 + 真量化）──')
  const t1 = Date.now()
  const train = await call('training_qat_train', {
    datasetDir, device: DEVICE, scheme: SCHEME, maxSteps: STEPS, maxSeqLen: SEQ,
    batchSize: 1, gradAccum: 1, lr: 1e-5, calibrationSamples: CALIB, checkpointEvery: 20,
    // `log` 不在工具 schema 里（工具面保持干净）；此处直接经 handler 透传以获得**流式进度**。
    // 长任务没有进度输出 = 无法区分"在跑"与"卡死"。
    log: (m) => say(`  ${m}`),
  }, { scope: 'train' })
  const trainMs = Date.now() - t1
  say(`train ok=${train.ok} code=${train.code} wall=${(trainMs / 1000).toFixed(1)}s`)
  if (train.ok !== true) { say(`train FAILED: ${JSON.stringify(train).slice(0, 2500)}`) }

  const report = {
    schema: 'training-qat/realmachine-run@1',
    at: new Date().toISOString(),
    parameters: { root, device: DEVICE, scheme: SCHEME, steps: STEPS, trainRows: TRAIN, valRows: VAL, maxSeqLen: SEQ, calibrationSamples: CALIB },
    prepareRaw: prep,
    trainRaw: train,
    wall: { prepareMs: Date.now() - t0 - trainMs, trainMs },
    log,
  }
  const p = join(OUT, `qat-run-${DEVICE}-${SCHEME}-${STEPS}steps.json`)
  writeFileSync(p, `${JSON.stringify(report, null, 2)}\n`)
  say(`原始输出已落盘：${p}`)
  return train.ok === true ? 0 : 1
}

main().then((c) => { process.exitCode = c }).catch((e) => { console.error('运行器异常：', e); process.exitCode = 2 })
