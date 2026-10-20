/**
 * training-qat —— 插件入口（DSH 侧）。
 *
 * 六个工具（`training_qat_*`）+ 一条只读健康路由 `/__training_qat/health`。
 * 本文件只做「装配 + 契约映射 + 治理注入 + **无损 JSON 闸门**」；业务逻辑在 src/core.mjs。
 *
 * 装配期做四件事（都 fail-closed）：
 *   ① 风险声明双向一致（声明面 ⇄ 工具面）
 *   ② 工具名与既有命名空间无交集
 *   ③ 每个工具都声明 output{schema, render}（**缺了会让整条 entry 被拒**）
 *   ④ **每个工具的输出都过 `assertLossless`** —— 注册成功 ≠ 输出契约成立
 *      （含 `undefined` 的返回会被 DSH 的无损 JSON 校验器拒掉，而 JSON.stringify 会静默丢掉它）
 */
import { readdirSync, readFileSync, existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

import { ALL_TOOL_SPECS } from './tools.mjs'
import { assertLossless } from './output.mjs'
import * as api from './core.mjs'
import { PACKAGE_VERSION, SCHEMA_VERSION, ALL_CODES, ToolError, ERROR_CODES } from './version.mjs'
import { paths, audit, digestArgs } from './runtime.mjs'
import { requireConfirm } from './confirm.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
export const name = '@xiciyang/training-qat'
export const inject = ['tools']

export const TOOL_NAMES = Object.freeze(ALL_TOOL_SPECS.map((t) => t.name))

/** 风险分级（与 manifest.yaml 必须逐条一致，由 tests/unit/registration.spec.mjs 断言）。 */
export const TOOL_RISK = Object.freeze({
  training_qat_prepare: { risk: 'write', sandbox: 'none', idempotent: true, confirmation: 'not-required', resources: ['disk:write', 'state:read'] },
  training_qat_train: { risk: 'write', sandbox: 'container', idempotent: false, confirmation: 'required', resources: ['gpu:write', 'disk:write', 'state:write', 'model:read'] },
  training_qat_validate: { risk: 'write', sandbox: 'container', idempotent: false, confirmation: 'required', resources: ['gpu:write', 'port:bind', 'state:write', 'model:read'] },
  training_qat_rollback: { risk: 'write', sandbox: 'none', idempotent: true, confirmation: 'required', resources: ['disk:write', 'state:write'] },
  training_qat_get_report: { risk: 'read', sandbox: 'none', idempotent: true, confirmation: 'not-required', resources: ['state:read'] },
  training_qat_health: { risk: 'read', sandbox: 'none', idempotent: true, confirmation: 'not-required', resources: [] },
})

const CONFIRM_SCOPE = { training_qat_train: 'train', training_qat_validate: 'validate', training_qat_rollback: 'rollback' }

export function assertRiskDeclarations(toolNames) {
  const problems = []
  for (const n of toolNames) if (!TOOL_RISK[n]) problems.push(`工具 ${n} 没有风险声明`)
  for (const n of Object.keys(TOOL_RISK)) if (!toolNames.includes(n)) problems.push(`风险声明 ${n} 没有对应工具`)
  if (problems.length) throw new Error(`风险声明不合法（fail-closed）：${problems.join('；')}`)
  return true
}

/** 工作区根**从模块位置推导**（可用环境变量覆盖），不硬编码本机绝对路径。 */
export const WORKSPACE = process.env.DSH_WORKSPACE || join(HERE, '..', '..', '..')

/** 既有命名空间（实时从兄弟包源码抓取）—— 只用于装配期粗检，细检在测试里。 */
export function existingNamespaces(workspace = WORKSPACE) {
  const out = []
  for (const pkg of ['inference-serving', 'inference-quantization', 'inference-tuning']) {
    for (const f of ['src/tools.mjs', 'src/index.mjs']) {
      const p = join(workspace, pkg, f)
      if (!existsSync(p)) continue
      out.push(...[...readFileSync(p, 'utf8').matchAll(/name:\s*'(inference_[a-z_]+)'/g)].map((m) => m[1]))
    }
  }
  for (const pkg of ['training-sft', 'training-pretrain', 'training-rl']) {
    for (const f of ['src/tools.mjs']) {
      const p = join(workspace, 'inference-training', pkg, f)
      if (!existsSync(p)) continue
      out.push(...[...readFileSync(p, 'utf8').matchAll(/name:\s*'(training_[a-z_]+)'/g)].map((m) => m[1]))
    }
  }
  return [...new Set(out)]
}

export function ok(data, versions, extras = {}) {
  return { ok: true, code: 'OK', data, versions, ...extras }
}

export function toEnvelope(err, versions) {
  if (err instanceof ToolError) {
    return { ok: false, code: err.code, error: { message: err.message, ...err.body }, versions }
  }
  return {
    ok: false,
    code: ERROR_CODES.E_INTERNAL,
    error: { message: '内部错误（必须可诊断）', detail: String(err?.message ?? err).slice(0, 400), stack: String(err?.stack ?? '').slice(-800), recoverable: false },
    versions,
  }
}

export function createCore({ logger = null } = {}) {
  const run = async (spec, args) => {
    const root = args?.root || paths().root
    const versions = { pkg: PACKAGE_VERSION, schema: SCHEMA_VERSION }
    const t0 = Date.now()
    let env
    try {
      const scope = CONFIRM_SCOPE[spec.name]
      if (scope) {
        await requireConfirm({ root, scope, payload: { ...args, root: undefined }, confirm: args?.confirm, summary: `${spec.name} 会写入共享状态与产出目录（${scope}）` })
      }
      const data = await spec.handler(args ?? {})
      // **工具边界的无损 JSON 闸门**：宁可在这里响亮失败，也不让 DSH 用一个看不懂的错误拒掉
      assertLossless(data, { tool: spec.name })
      env = ok(data, versions)
    } catch (e) {
      env = toEnvelope(e, versions)
    }
    try {
      const rec = await audit(root, { capability: spec.name.replace('training_qat_', ''), action: 'capability-call', phase: 'end', ...digestArgs(args), resultOk: env.ok === true, resultCode: env.code, ms: Date.now() - t0, risk: TOOL_RISK[spec.name].risk })
      env.governance = {
        schema: 'training-qat/governance@1',
        risk: TOOL_RISK[spec.name].risk,
        sandbox: TOOL_RISK[spec.name].sandbox,
        confirmation: TOOL_RISK[spec.name].confirmation,
        confirmationRequired: TOOL_RISK[spec.name].confirmation === 'required',
        audit: { ok: true, seq: rec.seq, hash: rec.hash, file: rec.file },
      }
    } catch (e) {
      env.governance = { schema: 'training-qat/governance@1', risk: TOOL_RISK[spec.name].risk, audit: { ok: false, reason: String(e.message).slice(0, 200) } }
    }
    logger?.info?.(`[${name}] ${spec.name} → ${env.code} (${Date.now() - t0}ms)`)
    return env
  }
  return { run, status: () => ({ tools: [...TOOL_NAMES], version: PACKAGE_VERSION }) }
}

export function apply(ctx) {
  assertRiskDeclarations(TOOL_NAMES)
  const overlap = TOOL_NAMES.filter((n) => existingNamespaces().includes(n))
  if (overlap.length) throw new Error(`工具名与既有命名空间冲突：${overlap.join(', ')}`)
  const core = createCore({ logger: ctx.logger })

  for (const spec of ALL_TOOL_SPECS) {
    ctx.effect(() => ctx.tools.register({
      name: spec.name,
      description: spec.description,
      parameters: spec.parameters,
      output: spec.output,
      async execute(args) { return core.run(spec, args || {}) },
    }), `${name}: register ${spec.name}`)
  }

  const web = ctx.get?.('webServer')
  if (web && typeof web.route === 'function') {
    ctx.effect(() => web.route('get', '/__training_qat/health', async (_req, res) => {
      try {
        const h = await api.health({ deep: false })
        res.writeHead(200, { 'content-type': 'application/json' })
        res.end(JSON.stringify(h))
      } catch (e) {
        res.writeHead(500, { 'content-type': 'application/json' })
        res.end(JSON.stringify({ ok: false, error: String(e.message) }))
      }
    }), `${name}: route /__training_qat/health`)
  }
  ctx.logger?.info?.(`[${name}] 六工具已注册：${TOOL_NAMES.join(', ')}`)
}

export { ALL_CODES, ALL_TOOL_SPECS, api }
export default { name, inject, apply }
