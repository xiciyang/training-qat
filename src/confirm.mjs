/**
 * confirm —— 确认令牌（R2 就地变更闸门）。
 *
 * 口径与推理三插件、training-sft 一致：**per-class-once，24h 内同类操作不再询问**。
 * 令牌由「能力 + 目标指纹」确定性推导 —— 目标变了令牌就变。
 * 状态落在 `<root>/state/training-qat-confirmations.json`（走 state-core 的原子写）。
 */
import { join } from 'node:path'
import { createHash } from 'node:crypto'
import { readText, atomicWrite } from '@xiciyang/state-core'
import { canonicalJson } from '@xiciyang/training-core'
import { ToolError, ERROR_CODES } from './version.mjs'

const TTL_MS = 24 * 3600 * 1000

function fp(scope, payload) {
  const clean = {}
  for (const k of Object.keys(payload ?? {}).sort()) {
    if (['confirm', 'options', 'root', 'log'].includes(k)) continue
    if (payload[k] === undefined) continue
    clean[k] = payload[k]
  }
  return `${scope}:${createHash('sha256').update(canonicalJson(clean)).digest('hex').slice(0, 16)}`
}

export function tokenFor(scope, payload) {
  return `tq_${createHash('sha256').update(fp(scope, payload)).digest('hex').slice(0, 16)}`
}

async function storePath(root) {
  return join(root, 'state', 'training-qat-confirmations.json')
}

export async function isConfirmed(root, scope, payload) {
  const p = await storePath(root)
  const raw = await readText(p, null)
  if (!raw) return false
  let doc = null
  try { doc = JSON.parse(raw) } catch { return false }
  const rec = doc?.[fp(scope, payload)]
  if (!rec) return false
  return Date.now() - rec.at < TTL_MS
}

export async function markConfirmed(root, scope, payload) {
  const p = await storePath(root)
  const raw = await readText(p, null)
  let doc = {}
  try { doc = raw ? JSON.parse(raw) : {} } catch { doc = {} }
  doc[fp(scope, payload)] = { at: Date.now(), scope, token: tokenFor(scope, payload) }
  await atomicWrite(p, `${JSON.stringify(doc, null, 2)}\n`)
}

/**
 * 闸门：未确认即抛 `E_CONFIRMATION_REQUIRED`（带 confirmToken 与影响说明）。
 */
export async function requireConfirm({ root, scope, payload, confirm, summary }) {
  if (await isConfirmed(root, scope, payload)) return { confirmed: true, source: 'memo' }
  const expected = tokenFor(scope, payload)
  if (confirm === expected) {
    await markConfirmed(root, scope, payload)
    return { confirmed: true, source: 'token' }
  }
  throw new ToolError(ERROR_CODES.E_CONFIRMATION_REQUIRED, {
    hint: `操作需要确认：${scope}（风险档 R2，策略 per-class-once）。确认影响后带 confirm="${expected}" 重新调用即可执行（24h 内同类操作不再询问）。`,
    details: { scope, confirmToken: expected, summary, retriable: false },
    recoverable: true,
  })
}
