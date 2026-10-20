#!/usr/bin/env node
/**
 * MCP server 入口（stdio，行分隔 JSON-RPC 2.0）。
 *
 * 与 DSH 通道共用 src/tools.mjs 的同一份规格与 src/core.mjs 的实现 —— 不存在行为分叉。
 * DSH 在 MCP 上扮演 client，不会消费自家插件的 MCP server；本入口服务于第三方 MCP 客户端，
 * 也是 training-qat 调 inference-quantization 的**同一条**通道形态。
 */
import readline from 'node:readline'
import { ALL_TOOL_SPECS, findToolSpec } from '../src/tools.mjs'
import { createCore } from '../src/index.mjs'
import { PACKAGE_VERSION, MCP_INTERFACE_VERSION } from '../src/version.mjs'

/**
 * **与 DSH 通道共用同一套执行语义**：同一个 `createCore()`。
 *
 * 为什么不能用 `spec.handler()` 了事（同侧兄弟包的既有写法）：那条路径**绕过了**
 *   ① 确认令牌闸门（R2 就地变更必须确认）；
 *   ② 治理信封（risk / sandbox / confirmation）；
 *   ③ **审计留痕**（capability-call 那条审计就是 createCore 写的）。
 * 于是"MCP 调一次"在审计链里**没有痕迹** —— 而审计是这套插件最重要的可追溯性承诺。
 * 这不是"行为分叉"的措辞问题，是**证据缺失**问题。故本包两个通道共用一个 core。
 */
const core = createCore({ logger: null })

const send = (o) => process.stdout.write(`${JSON.stringify(o)}\n`)

const rl = readline.createInterface({ input: process.stdin })
rl.on('line', async (line) => {
  if (!line.trim()) return
  let req
  try { req = JSON.parse(line) } catch { return }
  const { id, method, params } = req
  try {
    if (method === 'initialize') {
      send({ jsonrpc: '2.0', id, result: { protocolVersion: '2024-11-05', capabilities: { tools: {} }, serverInfo: { name: 'training-qat', version: PACKAGE_VERSION } } })
      return
    }
    if (method === 'notifications/initialized') return
    if (method === 'tools/list') {
      send({ jsonrpc: '2.0', id, result: { tools: ALL_TOOL_SPECS.map((t) => ({ name: t.name, description: t.description, inputSchema: t.parameters })) } })
      return
    }
    if (method === 'tools/call') {
      const spec = findToolSpec(params?.name)
      if (!spec) { send({ jsonrpc: '2.0', id, error: { code: -32601, message: `unknown tool ${params?.name}` } }); return }
      const out = await core.run(spec, params?.arguments ?? {})
      send({ jsonrpc: '2.0', id, result: { content: [{ type: 'text', text: JSON.stringify(out, null, 2) }], isError: out?.ok === false } })
      return
    }
    if (method === 'ping') { send({ jsonrpc: '2.0', id, result: {} }); return }
    send({ jsonrpc: '2.0', id, error: { code: -32601, message: `unknown method ${method}` } })
  } catch (e) {
    send({ jsonrpc: '2.0', id, error: { code: -32603, message: String(e?.message ?? e), data: { interfaceVersion: MCP_INTERFACE_VERSION } } })
  }
})
