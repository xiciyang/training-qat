/**
 * data-o1.spec —— **O1 去泄漏闸门**的双轨判据（`training-eval-contract` §4.1）。
 *
 * O1 是硬闸且**完全可判定**：任一轨非空 ⇒ fail-closed（拒绝训练），不是"标注一下继续"。
 * 这个 spec 用合成语料把两轨**分别**打红一次，证明闸门不是摆设：
 *   轨 1：整行内容哈希相等（val 整行出现在 train 里）
 *   轨 2：val 文本被**嵌在** train 行内部（哈希不等，但包含关系成立）
 * 并验证 `reverify()` 现场重算能独立复现同一结论（validate 用它，不信任 prepare 的结论）。
 */
import { mkdtempSync, rmSync, writeFileSync, mkdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { tmpdir } from 'node:os'
import { fileURLToPath } from 'node:url'

import { prepareQatData, reverify, discoverSource, assertTrainableLicense } from '../../src/data/prepare.mjs'

/** 合成语料没有真实来源血缘 ⇒ 显式给一个 license（**测试夹具**，不是生产路径）。 */
const FIXTURE_LICENSE = { spdx: 'cc-by-4.0', class: 'permissive', source: 'unit-test fixture（合成语料，非真实数据集）' }

const HERE = dirname(fileURLToPath(import.meta.url))
export const NAME = 'data O1（双轨去泄漏闸门）'

function corpus(dir, trainTexts, valTexts) {
  mkdirSync(dir, { recursive: true })
  writeFileSync(join(dir, 'calib.jsonl'), trainTexts.map((t) => JSON.stringify({ t })).join('\n') + '\n')
  writeFileSync(join(dir, 'test.jsonl'), valTexts.map((t) => JSON.stringify({ t })).join('\n') + '\n')
  return dir
}

const LONG = (tag) => `${tag} ` + 'lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor '.repeat(2)
const unique = (n, tag) => Array.from({ length: n }, (_, i) => `${tag}-${i}-` + 'the quick brown fox jumps over the lazy dog '.repeat(2))

export async function run(t) {
  const tmp = mkdtempSync(join(tmpdir(), 'tq-o1-'))

  // ── ① 干净语料：两轨都空 ⇒ allClear=true ──
  const clean = corpus(join(tmp, 'src-clean'), unique(40, 'tr'), unique(20, 'va'))
  const r1 = await prepareQatData({ sourceLicense: FIXTURE_LICENSE, sourceDir: clean, datasetsRoot: tmp, outDir: join(tmp, 'out1'), train: 40, val: 20 })
  t.assert(r1.allClear === true, `干净语料 allClear=true（实际 ${r1.allClear}，verify=${JSON.stringify(r1.verify)}）`)
  t.assert(r1.verify.length === 3, '三轨（hash / 子串包含 / recordHash）都被记账')
  t.assert(r1.verify.every((v) => v.ok), '三轨全部 ok')
  t.assert(r1.trainRows === 40 && r1.valRows === 20, `切分规模正确（train=${r1.trainRows} val=${r1.valRows}）`)

  // ── ② 轨 1 打红：val 里有一行与 train 逐字节相同 ⇒ 哈希交集非空 ──
  const dup = LONG('SHARED-LINE')
  const src2 = corpus(join(tmp, 'src-dup'), [...unique(10, 'tr'), dup], [dup, ...unique(10, 'va')])
  const r2 = await prepareQatData({ sourceLicense: FIXTURE_LICENSE, sourceDir: src2, datasetsRoot: tmp, outDir: join(tmp, 'out2'), train: 11, val: 11 })
  t.assert(r2.allClear === false, '轨 1 打红 ⇒ allClear=false（fail-closed）')
  t.assert(r2.verify.some((v) => v.track === 'content-sha256' && v.ok === false), 'hash 交集轨被判定非空')

  // ── ③ 轨 2 打红：val 文本**嵌在** train 行内部（哈希不等）⇒ 包含轨非空 ──
  const inner = LONG('EMBEDDED-EVAL-TEXT')
  const src3 = corpus(join(tmp, 'src-contain'), [inner, ...unique(10, 'tr')], [inner.slice(0, inner.length - 1), ...unique(10, 'va')])
  // 上面 val 行比 train 里的子串少一个字符 —— 交换方向：让 val 的文本是 train 的子串
  const src3b = corpus(join(tmp, 'src-contain2'), [`PREFIX-PADDING ${inner} SUFFIX-PADDING`, ...unique(10, 'tr')], [inner, ...unique(10, 'va')])
  void src3
  const r3 = await prepareQatData({ sourceLicense: FIXTURE_LICENSE, sourceDir: src3b, datasetsRoot: tmp, outDir: join(tmp, 'out3'), train: 11, val: 11 })
  t.assert(r3.allClear === false, '轨 2 打红 ⇒ allClear=false')
  t.assert(r3.verify.some((v) => v.track === 'substring-containment' && v.ok === false), '子串包含轨被判定命中')
  // 且这一例的 hash 轨是**通的** —— 证明两轨真的互相独立（子串轨不是哈希轨的复制）
  const hashTrack = r3.verify.find((v) => v.track === 'content-sha256')
  t.assert(hashTrack.ok === true, `该例的整行哈希交集为空（hash 轨 ok=${hashTrack.ok}）—— 两轨独立`)

  // ── ④ O1 现场重算：reverify 能独立复现结论 ──
  const re1 = reverify(join(tmp, 'out1'))
  t.assert(re1.ok === true && re1.track1.hits === 0 && re1.track2.hits === 0, 'reverify 在干净集上复现 ok')
  t.assert(/^sha256:/.test(re1.trainFileHash) && /^sha256:/.test(re1.valFileHash), 'reverify 给出两侧文件的内容哈希（可复核）')

  // ── ⑤ 去重与确定性：同 seed 同输入 ⇒ 逐条相同的切分 ──
  const r5a = await prepareQatData({ sourceLicense: FIXTURE_LICENSE, sourceDir: clean, datasetsRoot: tmp, outDir: join(tmp, 'out5a'), train: 40, val: 20, seed: 7 })
  const r5b = await prepareQatData({ sourceLicense: FIXTURE_LICENSE, sourceDir: clean, datasetsRoot: tmp, outDir: join(tmp, 'out5b'), train: 40, val: 20, seed: 7 })
  t.assert(JSON.stringify(r5a.trainTexts) === JSON.stringify(r5b.trainTexts), '同 seed 同输入 ⇒ 逐条相同的训练切分（可复现）')
  const r5c = await prepareQatData({ sourceLicense: FIXTURE_LICENSE, sourceDir: clean, datasetsRoot: tmp, outDir: join(tmp, 'out5c'), train: 40, val: 20, seed: 8 })
  t.assert(JSON.stringify(r5a.trainTexts) !== JSON.stringify(r5c.trainTexts), '不同 seed ⇒ 不同切分（种子真的起作用）')

  // ── ⑥ discoverSource 在空目录上返回 null（不假装找到）──
  const emptyRoot = join(tmp, 'no-datasets')
  mkdirSync(emptyRoot, { recursive: true })
  t.assert(discoverSource(emptyRoot) === null, '空的 datasets 根 ⇒ discoverSource 返回 null（不猜路径）')

  rmSync(tmp, { recursive: true, force: true })
}
