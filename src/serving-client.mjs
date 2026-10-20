/**
 * serving-client —— 调 `inference-serving` 的 MCP 工具（**接口层复用，不改对方源码**）。
 *
 * 为什么自己起一个 MCP client 而不是让 DSH 转发：训练侧要在一个能力调用内完成
 * "合并模型 → 部署 → 评估 → 下线"的闭环；跨工具手工编排既不可复现、也不进审计。
 * 这与 inference-quantization / inference-tuning 已有的做法一致（同一条路径解析逻辑）。
 *
 * 解析顺序（与兄弟包同形）：配置 → 兄弟目录 → 包内 → 环境变量。
 */
import { spawn } from 'node:child_process'
import { existsSync, readFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createInterface } from 'node:readline'
import { ToolError, ERROR_CODES } from './version.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))

export async function resolveServingPath({ root, env = process.env, packageRoot = join(HERE, '..') } = {}) {
  const tried = []
  const cfg = join(root, 'config', 'serving-path.txt')
  tried.push({ path: cfg, from: 'config file' })
  if (existsSync(cfg)) {
    const p = readFileSync(cfg, 'utf8').trim()
    if (p && existsSync(p)) return { root: p, serverPath: join(p, 'mcp', 'server.mjs'), from: 'config file', tried }
  }
  // 候选路径**从包位置推导**（training-qat 与 inference-serving 是兄弟目录），不硬编码本机路径
  const workspace = env.DSH_WORKSPACE || join(packageRoot, '..', '..')
  for (const cand of [env.INFERENCE_SERVING_ROOT, join(workspace, 'inference-serving'), join(packageRoot, '..', '..', 'inference-serving')].filter(Boolean).filter((v, i, a) => a.indexOf(v) === i)) {
    tried.push({ path: cand, from: cand === env.INFERENCE_SERVING_ROOT ? 'env' : 'sibling directory' })
    if (existsSync(join(cand, 'mcp', 'server.mjs'))) return { root: cand, serverPath: join(cand, 'mcp', 'server.mjs'), from: 'sibling/env', tried }
  }
  throw new ToolError(ERROR_CODES.E_SERVING_UNAVAILABLE, { hint: `找不到 inference-serving 的 MCP 入口（tried: ${tried.map((t) => t.path).join(', ')}）`, details: { tried } })
}

/** 极简 MCP stdio 客户端（行分隔 JSON-RPC 2.0）。 */
export class ServingMcpClient {
  constructor(serverPath, { timeoutMs = 600000 } = {}) {
    this.serverPath = serverPath
    this.timeoutMs = timeoutMs
    this.child = null
    this.seq = 0
    this.pending = new Map()
    this.stderr = []
  }

  async start() {
    this.child = spawn(process.execPath, [this.serverPath], { stdio: ['pipe', 'pipe', 'pipe'] })
    const rl = createInterface({ input: this.child.stdout })
    rl.on('line', (line) => {
      let msg
      try { msg = JSON.parse(line) } catch { return }
      const p = this.pending.get(msg.id)
      if (!p) return
      this.pending.delete(msg.id)
      if (msg.error) p.reject(new Error(`MCP error: ${JSON.stringify(msg.error).slice(0, 300)}`))
      else p.resolve(msg.result)
    })
    const rle = createInterface({ input: this.child.stderr })
    rle.on('line', (l) => { this.stderr.push(l); if (this.stderr.length > 30) this.stderr.shift() })
    this.child.on('exit', (code) => {
      for (const [, p] of this.pending) p.reject(new Error(`serving MCP 退出 code=${code}：${this.stderr.slice(-3).join(' | ').slice(0, 300)}`))
      this.pending.clear()
      this.child = null
    })
    await this.request('initialize', { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'training-qat', version: '0.1.0' } })
    this.notify('notifications/initialized', {})
    return this
  }

  request(method, params) {
    if (!this.child) throw new Error('serving MCP 未启动')
    const id = ++this.seq
    return new Promise((resolve, reject) => {
      const t = setTimeout(() => { this.pending.delete(id); reject(new Error(`MCP ${method} 超时`)) }, this.timeoutMs)
      this.pending.set(id, { resolve: (r) => { clearTimeout(t); resolve(r) }, reject: (e) => { clearTimeout(t); reject(e) } })
      this.child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`)
    })
  }

  notify(method, params) {
    if (!this.child) return
    this.child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', method, params })}\n`)
  }

  /** 调一个 MCP 工具，返回解析后的 JSON（工具返回的是 text content）。 */
  async callTool(name, args = {}) {
    const res = await this.request('tools/call', { name, arguments: args })
    const text = res?.content?.find?.((c) => c.type === 'text')?.text ?? JSON.stringify(res)
    try { return JSON.parse(text) } catch { return { _raw: text } }
  }

  async close() {
    if (!this.child) return
    try { this.child.kill('SIGTERM') } catch { /* ignore */ }
    const c = this.child
    await new Promise((r) => { const t = setTimeout(() => { try { c.kill('SIGKILL') } catch { /* ignore */ } ; r() }, 5000); c.on('exit', () => { clearTimeout(t); r() }) })
    this.child = null
  }
}

/**
 * 带确认令牌的调用：serving 的 write 档操作返回 `E_CONFIRMATION_REQUIRED` + `confirmToken`，
 * 把令牌回传即可执行。**有界重试**（最多 3 轮），不做无限循环。
 * 训练侧的所有 serving 调用都**只发生在自己的临时端点上**（不允许触碰存量端点）。
 */
export async function callWithConfirm(client, name, args, { maxRounds = 3, log = () => {} } = {}) {
  let a = { ...args }
  for (let i = 0; i < maxRounds; i += 1) {
    const r = await client.callTool(name, a)
    if (r && r.ok !== false) return r
    const token = r?.details?.confirmToken
    if (r?.code === 'E_CONFIRMATION_REQUIRED' && token) { log(`  ${name}: 需确认，带令牌重试（${i + 1}/${maxRounds}）`); a = { ...a, confirm: token }; continue }
    return r
  }
  return await client.callTool(name, a)
}
