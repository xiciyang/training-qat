#!/usr/bin/env node
/** 极简测试驱动（零依赖，与四个插件同款风格）。用法：node tests/run.mjs [spec 相对路径...] */
import { readdirSync, statSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const ROOT = join(HERE, '..')

function collect(dir) {
  const out = []
  for (const f of readdirSync(dir)) {
    const p = join(dir, f)
    if (statSync(p).isDirectory()) out.push(...collect(p))
    else if (f.endsWith('.spec.mjs')) out.push(p)
  }
  return out.sort()
}

const targets = process.argv.slice(2).length
  ? process.argv.slice(2).map((p) => join(ROOT, p))
  : collect(join(HERE, 'unit'))
let failed = 0
let passed = 0
let asserts = 0
let skipped = 0

for (const file of targets) {
  const mod = await import(pathToFileURL(file).href)
  const label = mod.NAME || file.split('/').slice(-2).join('/')
  let n = 0
  const failures = []
  const skips = []
  const t = {
    assert(cond, msg) { n += 1; if (!cond) failures.push(msg) },
    skip(msg) { skips.push(msg) },
  }
  console.log(`\n=== ${label} ===`)
  try { await mod.run(t) } catch (e) { failures.push(`抛出异常：${String(e?.stack ? e.stack.split('\n').slice(0, 3).join(' | ') : e)}`) }
  asserts += n
  if (failures.length === 0) {
    passed += 1
    if (skips.length) { skipped += 1; console.log(`SKIP  ${n} 项断言（跳过：${skips.join('；')}）`) }
    else console.log(`PASS  ${n} 项断言`)
  } else {
    failed += 1
    console.log(`FAIL  ${failures.length}/${n} 项断言未通过：`)
    for (const f of failures) console.log(`  - ${f}`)
  }
}

console.log(`\n${failed === 0 ? 'ALL SPECS PASSED' : 'SPEC FAILURES'}: ${passed} passed / ${failed} failed / ${skipped} skipped；断言合计 ${asserts}`)
process.exit(failed === 0 ? 0 : 1)
