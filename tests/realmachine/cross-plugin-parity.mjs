#!/usr/bin/env node
/**
 * 跨插件对账（直接调 `inference-quantization` 的 MCP 入口）：
 * 把**真实的 QAT 产出**喂给 `inference_quantization_validate`，证明
 *   ① 该入口能验收一个**外部**产出的量化 checkpoint（M2 的闭环节点）；
 *   ② 它把产出的 scheme 识别为与产出自身一致（`8bit/channel`）；
 *   ③ 它给得出 parity.dropPct（用**量化插件自己的** parity 实现，同一测试集）。
 *
 * 用法：node tests/realmachine/cross-plugin-parity.mjs <candidateDir> <baselineModel> [maxDrop]
 */
import { writeFileSync, mkdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { ServingMcpClient, resolveServingPath } from '../../src/serving-client.mjs'
import { resolveQuantizationPath } from '../../src/core.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const OUT = join(HERE, '..', '..', 'out', 'realmachine')
const [cand, baseline, maxDrop = '3'] = process.argv.slice(2)
if (!cand || !baseline) { console.error('usage: <candidateDir> <baselineModel> [maxDrop]'); process.exit(2) }
const say = (m) => console.log(`[${new Date().toISOString()}] ${m}`)

const qp = await resolveQuantizationPath({})
say(`quantization MCP: ${qp.serverPath}`)
const client = await new ServingMcpClient(qp.serverPath, { timeoutMs: 1800000 }).start()
const callWithConfirm = async (name, args) => {
  let a = { ...args }
  for (let i = 0; i < 3; i += 1) {
    const r = await client.callTool(name, a)
    if (r && r.ok !== false) return r
    const tk = r?.details?.confirmToken
    if (r?.code === 'E_CONFIRMATION_REQUIRED' && tk) { say(`  ${name}: 需确认，带令牌重试`); a = { ...a, confirm: tk }; continue }
    return r
  }
  return client.callTool(name, a)
}
mkdirSync(OUT, { recursive: true })
const t0 = Date.now()
const r = await callWithConfirm('inference_quantization_validate', {
  candidate: cand, baseline, max_accuracy_drop_pct: Number(maxDrop),
  // 对方插件的**显式**开关：本机量化镜像尚未验证。打开它并把这个事实记进结果（不静默）。
  allowUnverifiedImage: true,
})
const report = { schema: 'training-qat/cross-plugin-parity@1', at: new Date().toISOString(), candidate: cand, baseline, maxDrop: Number(maxDrop), ms: Date.now() - t0, raw: r }
writeFileSync(join(OUT, `cross-plugin-parity-${cand.split('/').pop()}.json`), `${JSON.stringify(report, null, 2)}\n`)
say(`ok=${r.ok} code=${r.code} wall=${((Date.now() - t0) / 1000).toFixed(1)}s`)
say(`scheme=${r?.data?.candidate?.scheme} schemeRecognized=${r?.data?.candidate?.schemeRecognized} dropPct=${r?.data?.parity?.dropPct} decision=${r?.data?.gate?.decision}`)
await client.close()
process.exit(r?.ok === true ? 0 : 1)
