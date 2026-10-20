#!/usr/bin/env node
/**
 * 真机 validate 运行器（P0 验收项）：serving 上做配对测量 + 双重门禁 + O1–O5 + 跨插件对账。
 *
 * **阈值由调用方（这里）显式给出** —— 契约禁止插件内置默认值。
 * 本脚本是"调用方"，所以它必须把阈值写在命令行里、并且落盘，让"谁定的阈值"可追溯。
 *
 * 用法：node tests/realmachine/qat-validate.mjs --runId tq_... --max-drop 2.0 [--min-perf-gain -100]
 */
import { mkdirSync, writeFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

import { findToolSpec } from '../../src/tools.mjs'
import { createCore } from '../../src/index.mjs'
import { tokenFor } from '../../src/confirm.mjs'
import { paths } from '../../src/runtime.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const PKG = join(HERE, '..', '..')
const OUT = join(PKG, 'out', 'realmachine')

const argv = process.argv.slice(2)
const argOf = (k, d) => { const i = argv.indexOf(k); return i >= 0 && argv[i + 1] ? argv[i + 1] : d }
const RUN_ID = argOf('--runId', null)
const MAX_DROP = Number(argOf('--max-drop', 'NaN'))
const MIN_PERF = argOf('--min-perf-gain', null)
const NTRAIN = Number(argOf('--nTrain', '32'))
const NVAL = Number(argOf('--nVal', '32'))
const CROSS = argOf('--cross', 'true') !== 'false'

const log = []
const say = (m) => { const line = `[${new Date().toISOString()}] ${m}`; log.push(line); console.log(line) }
const core = createCore({ logger: null })

async function call(name, args, scope) {
  const spec = findToolSpec(name)
  let out = await core.run(spec, args)
  if (out?.code === 'E_CONFIRMATION_REQUIRED') {
    const tk = out.error?.details?.confirmToken ?? tokenFor(scope, args)
    say(`  ${name} 需确认，带令牌重试`)
    out = await core.run(spec, { ...args, confirm: tk })
  }
  return out
}

async function main() {
  mkdirSync(OUT, { recursive: true })
  const thresholds = { maxAccuracyDropPct: MAX_DROP, ...(MIN_PERF === null ? {} : { minPerfGainPct: Number(MIN_PERF) }) }
  say(`runId=${RUN_ID} thresholds=${JSON.stringify(thresholds)}（**由调用方显式给出**，非插件默认）`)

  const report = {
    schema: 'training-qat/realmachine-validate@1',
    at: new Date().toISOString(),
    runId: RUN_ID,
    thresholdsDeclaredByCaller: thresholds,
    parameters: { nTrain: NTRAIN, nVal: NVAL, crossPluginValidate: CROSS },
    log,
  }

  // ① 先只读一遍 registry / 产物，确认对象存在（失败要在最便宜的地方失败）
  const rep = await core.run(findToolSpec('training_qat_get_report'), { runId: RUN_ID })
  report.runBefore = rep?.data?.run ?? rep
  if (rep?.ok !== true) { say(`get_report 失败：${JSON.stringify(rep).slice(0, 400)}`); report.getReport = rep; return 1 }
  say(`run 状态=${rep.data.run.status} scheme=${rep.data.run.scheme?.scheme} quantizedDir=${rep.data.run.artifacts?.quantizedDir}`)

  // ② validate（serving 仪器 + 跨插件对账）
  const t0 = Date.now()
  const val = await call('training_qat_validate', { runId: RUN_ID, nTrain: NTRAIN, nVal: NVAL, thresholds, crossPluginValidate: CROSS }, 'validate')
  report.validateRaw = val
  report.wallMs = Date.now() - t0
  say(`validate ok=${val.ok} code=${val.code} wall=${((Date.now() - t0) / 1000).toFixed(1)}s`)
  if (val.ok !== true) say(`validate 失败：${JSON.stringify(val).slice(0, 1500)}`)

  const p = join(OUT, `qat-validate-${RUN_ID}.json`)
  writeFileSync(p, `${JSON.stringify(report, null, 2)}\n`)
  say(`原始输出已落盘：${p}`)
  return val.ok === true ? 0 : 1
}

main().then((c) => { process.exitCode = c }).catch((e) => { console.error('运行器异常：', e); process.exitCode = 2 })
