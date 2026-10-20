/**
 * registration.spec —— 「注册路径必须被测试覆盖」+「工具名不与既有命名空间冲突」。
 *
 * 判据来自本机实测教训：**注入返回 host ✓ ≠ 工具注册成功**。
 * mock ctx 的 `register` 必须**像 DSH 一样挑刺**（缺 output{schema,render} 就抛），
 * 然后**真调一次**只读工具 —— 两件事都做才算覆盖。
 *
 * 本 spec 额外断言：**每个工具的返回都是无损 JSON**（长任务 2 缺陷 3 的正面防线）
 * 以及 **6 个工具面与 manifest.yaml 逐条一致**（声明漂移是这类机制唯一真正的失败模式）。
 */
import { mkdtempSync, rmSync, readFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { tmpdir } from 'node:os'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'
import { apply, TOOL_NAMES, TOOL_RISK, existingNamespaces } from '../../src/index.mjs'
import { findLosslessProblems } from '../../src/output.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const PKG = join(HERE, '..', '..')
const WORKSPACE = '/home/xiciyang/deepseek_harness'
export const NAME = 'registration（注册路径 / 无损 JSON / 命名空间 / manifest 一致）'

function strictCtx(sink) {
  const routes = []
  return {
    routes,
    logger: Object.assign(() => ({ info() {}, warn() {}, error() {} }), { info() {}, warn() {}, error() {} }),
    effect(fn) { return fn() },
    get(n) { return n === 'webServer' ? { route(m, p, h) { routes.push({ m, p, h }); return () => {} } } : undefined },
    on() { return () => {} },
    tools: {
      register(tool) {
        if (!tool.output || !tool.output.schema || typeof tool.output.render !== 'function') {
          throw new Error(`tool "${tool.name}" must declare output { schema, render, presentationMeta? }`)
        }
        if (!tool.parameters || tool.parameters.type !== 'object') throw new Error(`tool "${tool.name}" 缺 parameters`)
        if (typeof tool.description !== 'string' || tool.description.length < 40) throw new Error(`tool "${tool.name}" description 太短`)
        sink.push(tool)
        return () => {}
      },
    },
  }
}

export async function run(t) {
  const tmp = mkdtempSync(join(tmpdir(), 'tq-reg-'))
  const prev = process.env.TRAINING_QAT_ROOT
  process.env.TRAINING_QAT_ROOT = tmp
  const registered = []
  const ctx = strictCtx(registered)
  let threw = null
  try { apply(ctx) } catch (e) { threw = String(e.message) }
  t.assert(threw === null, `apply() 在严格 mock ctx 下不抛错（实际：${threw}）`)
  t.assert(registered.length === 6, `注册了 6 个工具（实际 ${registered.length}）`)
  t.assert(JSON.stringify(registered.map((x) => x.name)) === JSON.stringify([...TOOL_NAMES]), '注册顺序与 TOOL_NAMES 一致')
  t.assert(registered.every((x) => x.name.startsWith('training_qat_')), '全部工具带 training_qat_ 前缀')

  for (const tool of registered) {
    t.assert(typeof tool.execute === 'function', `${tool.name} 有 execute`)
    t.assert(tool.output && tool.output.schema && typeof tool.output.render === 'function', `${tool.name} 声明 output{schema,render}`)
    const rendered = tool.output.render({}, { probe: 1 })
    t.assert(Array.isArray(rendered) && rendered[0]?.type === 'text' && typeof rendered[0].text === 'string', `${tool.name} 的 render 返回 text 块`)
  }

  // 真调一次（只读、无副作用）——「注册成功 ≠ 能用」
  const health = registered.find((x) => x.name === 'training_qat_health')
  const out = await health.execute({})
  t.assert(out && typeof out === 'object', '真调 training_qat_health 返回对象')
  t.assert(out.ok === true || out.ok === false, '返回值是标准信封（ok 字段）')
  t.assert(out.versions?.pkg === '0.1.0', `信封带 versions（实际 ${JSON.stringify(out.versions)}）`)
  t.assert(out.governance?.audit?.ok === true, '返回信封带 governance.audit（只读调用也留痕）')
  t.assert(findLosslessProblems(out).ok, `health 的返回是无损 JSON（问题：${JSON.stringify(findLosslessProblems(out).problems.slice(0, 3))}）`)
  t.assert(out.data?.declaredScheme?.weightBits === 8, 'health 里声明的方案是 8bit（**不是** W4A16 —— M3 的正面防线）')
  t.assert(/W4A16/.test(out.data?.declaredScheme?.note ?? ''), 'health 明确记录"W4A16 本包不产出、也不声明"')

  const report = registered.find((x) => x.name === 'training_qat_get_report')
  const rep = await report.execute({})
  t.assert(rep.ok === true && Array.isArray(rep.data.runs), '真调 training_qat_get_report 返回 runs 数组')
  t.assert(rep.data.audit && typeof rep.data.audit.ok === 'boolean', 'get_report 附审计链校验结果')
  t.assert(findLosslessProblems(rep).ok, 'get_report 的返回是无损 JSON')

  // 每个工具的输出都过无损 JSON 闸门（用会失败的输入触发错误信封，也必须无损）
  for (const tool of registered) {
    // eslint-disable-next-line no-await-in-loop
    const r = await tool.execute({ runId: 'tq_does_not_exist' })
    t.assert(findLosslessProblems(r).ok, `${tool.name} 的错误信封也是无损 JSON`)
    t.assert(typeof r.code === 'string', `${tool.name} 的错误信封带闭集错误码`)
  }

  t.assert(ctx.routes.some((r) => r.p === '/__training_qat/health'), '注册了 /__training_qat/health 路由')

  // 命名空间：去兄弟包源码实时抓名，交集必须为空
  const siblings = existingNamespaces(WORKSPACE)
  t.assert(siblings.length >= 30, `抓到兄弟包工具名（${siblings.length} 个）`)
  const overlap = TOOL_NAMES.filter((n) => siblings.includes(n))
  t.assert(overlap.length === 0, `与既有命名空间无交集（实际重叠：${overlap.join(', ') || 'none'}）`)

  // 风险声明与工具面双向一致
  t.assert(Object.keys(TOOL_RISK).length === TOOL_NAMES.length, '风险声明数与工具数一致')
  for (const n of TOOL_NAMES) t.assert(Boolean(TOOL_RISK[n]), `${n} 有风险声明`)
  t.assert(TOOL_RISK.training_qat_train.risk === 'write' && TOOL_RISK.training_qat_train.sandbox === 'container', 'train 是 write + container 沙箱')
  t.assert(TOOL_RISK.training_qat_train.confirmation === 'required', 'train 需要确认令牌')
  t.assert(TOOL_RISK.training_qat_health.risk === 'read', 'health 是 read 档')

  // manifest.yaml 与代码逐条一致
  const mf = readFileSync(join(PKG, 'manifest.yaml'), 'utf8')
  for (const n of TOOL_NAMES) t.assert(mf.includes(`tool: ${n}`), `manifest.yaml 声明了 ${n}`)
  for (const [n, d] of Object.entries(TOOL_RISK)) {
    const block = mf.split(`tool: ${n}`)[1]?.split('- id:')[0] ?? ''
    t.assert(block.includes(`risk: ${d.risk}`), `manifest ${n} risk 一致`)
    t.assert(block.includes(`sandbox: ${d.sandbox}`), `manifest ${n} sandbox 一致`)
    t.assert(block.includes(`confirmation: ${d.confirmation}`), `manifest ${n} confirmation 一致`)
  }

  // 包元数据：QAT 框架必须**显式**声明为可选依赖组
  const pkg = JSON.parse(readFileSync(join(PKG, 'package.json'), 'utf8'))
  t.assert(pkg.name === '@xiciyang/training-qat', 'package.json name')
  t.assert(pkg.dsh?.bundle?.patch, '声明 dsh.bundle.patch（否则 DSH 不会装载）')
  for (const f of ['torch', 'transformers', 'torchao', 'llmcompressor', 'compressed-tensors', 'datasets']) {
    t.assert(Boolean(pkg.optionalDependencies?.[f]), `optionalDependencies 显式声明 ${f}`)
  }
  for (const l of ['testing-core', 'data-core', 'state-core', 'audit-core', 'sandbox-core', 'training-core']) {
    t.assert(Boolean(pkg.dependencies?.[`@xiciyang/${l}`]), `声明共享库依赖 @xiciyang/${l}`)
  }

  // ── MCP 通道：必须与 DSH 通道**同一套信封 + 同一道确认闸门**（不能绕过审计）──
  const req = [
    { jsonrpc: '2.0', id: 1, method: 'initialize', params: {} },
    { jsonrpc: '2.0', id: 2, method: 'tools/list', params: {} },
    { jsonrpc: '2.0', id: 3, method: 'tools/call', params: { name: 'training_qat_get_report', arguments: {} } },
    { jsonrpc: '2.0', id: 4, method: 'tools/call', params: { name: 'training_qat_rollback', arguments: { runId: 'tq_nope', purge: false } } },
  ]
  const run = spawnSync(process.execPath, [join(PKG, 'mcp', 'server.mjs')], {
    input: `${req.map((r) => JSON.stringify(r)).join('\n')}\n`, encoding: 'utf8', timeout: 120000,
  })
  const lines = String(run.stdout || '').split('\n').filter(Boolean).map((l) => { try { return JSON.parse(l) } catch { return null } }).filter(Boolean)
  const byId = new Map(lines.map((m) => [m.id, m]))
  t.assert(byId.get(1)?.result?.serverInfo?.name === 'training-qat', 'MCP initialize 返回 serverInfo')
  t.assert((byId.get(2)?.result?.tools ?? []).length === 6, `MCP tools/list 返回 6 个工具（实际 ${(byId.get(2)?.result?.tools ?? []).length}）`)
  const call3 = JSON.parse(byId.get(3)?.result?.content?.[0]?.text ?? 'null')
  t.assert(call3?.ok === true && call3?.code === 'OK', 'MCP tools/call 返回与 DSH 同一套信封')
  t.assert(Boolean(call3?.data) && Array.isArray(call3?.data?.runs), 'MCP 信封带 data')
  t.assert(call3?.governance?.audit?.ok === true, 'MCP 调用也**进审计**（不是无痕调用）')
  t.assert(typeof call3?.versions?.pkg === 'string', 'MCP 信封带 versions')
  const call4 = JSON.parse(byId.get(4)?.result?.content?.[0]?.text ?? 'null')
  t.assert(call4?.code === 'E_CONFIRMATION_REQUIRED',
    `MCP 通道**不绕过**确认闸门（rollback 未带令牌应被拒，实际 ${call4?.code}）`)
  t.assert(findLosslessProblems(call3).ok && findLosslessProblems(call4).ok, 'MCP 两个响应都是无损 JSON')

  if (prev === undefined) delete process.env.TRAINING_QAT_ROOT; else process.env.TRAINING_QAT_ROOT = prev
  rmSync(tmp, { recursive: true, force: true })
}
