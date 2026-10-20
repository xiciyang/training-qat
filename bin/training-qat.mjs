#!/usr/bin/env node
/**
 * CLI（薄）—— 与 MCP / DSH 两条通道共用 src/core.mjs 的同一份实现，无行为分叉。
 * 用法：training-qat <prepare|train|validate|rollback|get_report|health> [--k v ...]
 */
import { ALL_TOOL_SPECS } from '../src/tools.mjs'

const argv = process.argv.slice(2)
const cmd = argv[0]
const spec = ALL_TOOL_SPECS.find((t) => t.name === `training_qat_${cmd}`)
if (!spec) {
  console.error(`用法：training-qat <${ALL_TOOL_SPECS.map((t) => t.name.replace('training_qat_', '')).join('|')}> [--key value ...]`)
  process.exit(2)
}
const args = {}
for (let i = 1; i < argv.length; i += 1) {
  if (!argv[i].startsWith('--')) continue
  const k = argv[i].slice(2)
  const v = argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[++i] : 'true'
  args[k] = v === 'true' ? true : v === 'false' ? false : (/^-?\d+$/.test(v) ? Number(v) : (/^[{[]/.test(v) ? JSON.parse(v) : v))
}
const out = await spec.handler(args)
console.log(JSON.stringify(out, null, 2))
process.exit(out && out.ok === false ? 1 : 0)
