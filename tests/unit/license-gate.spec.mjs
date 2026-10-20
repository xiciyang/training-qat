/**
 * license-gate.spec —— `license-whitelist.md` §2.1 硬边界的正面防线。
 *
 * 这条闸门不是设想出来的：**它是 2026-09-29 阶段 4 第一次真机 QAT 跑出来的**。
 * 那次用的是 wikitext（`cc-by-sa-3.0` + `gfdl` 双许可），而 §2.1 写死
 * 「wikitext 只作校准集与评估集，**不得当训练语料引入**」。
 * QAT 会更新权重 ⇒ 它就是训练 ⇒ 必须被拒。
 *
 * 处置是**改语料 + 加闸门**，不是给规则开例外。这个 spec 把闸门钉住：
 *   ① share-alike ⇒ `E_LICENSE_NOT_TRAINABLE`
 *   ② 授权不明 ⇒ `E_LICENSE_UNRESOLVED`（fail-closed：宁可拒绝，不在不明授权上训练）
 *   ③ 自动发现（discoverSource）**跳过** share-alike 目录
 *   ④ license **继承自来源血缘**，产出血缘里逐字一致（本包不自己写 SPDX 值）
 */
import { mkdtempSync, rmSync, writeFileSync, mkdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { tmpdir } from 'node:os'
import { fileURLToPath } from 'node:url'

import { prepareQatData, assertTrainableLicense, discoverSource, readSourceLicense } from '../../src/data/prepare.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
export const NAME = 'license gate（share-alike 不得作训练语料）'

const text = (tag, n) => Array.from({ length: n }, (_, i) => `${tag}-${i}-` + 'the quick brown fox jumps over the lazy dog '.repeat(2))

function corpus(dir, { trainFile, valFile, license }) {
  mkdirSync(dir, { recursive: true })
  writeFileSync(join(dir, trainFile), text('tr', 20).map((t) => JSON.stringify({ t })).join('\n') + '\n')
  writeFileSync(join(dir, valFile), text('va', 10).map((t) => JSON.stringify({ t })).join('\n') + '\n')
  if (license !== undefined) writeFileSync(join(dir, 'lineage.json'), JSON.stringify({ schema: 'ai-infra/lineage@1', license }))
  return dir
}

export async function run(t) {
  const tmp = mkdtempSync(join(tmpdir(), 'tq-lic-'))

  // ── ① share-alike ⇒ 拒绝 ──
  let threw = null
  try { assertTrainableLicense({ spdx: 'cc-by-sa-3.0', class: 'share-alike' }, { sourceDir: '/x' }) } catch (e) { threw = String(e.message) }
  t.assert(threw && /E_LICENSE_NOT_TRAINABLE/.test(threw) && /不得当作训练语料/.test(threw),
    `share-alike 被拒绝且理由点明硬边界（实际：${String(threw).slice(0, 120)}）`)

  // ── ② 授权不明 ⇒ 拒绝（fail-closed）──
  threw = null
  try { assertTrainableLicense(null, { sourceDir: '/x' }) } catch (e) { threw = String(e.message) }
  t.assert(threw && /E_LICENSE_UNRESOLVED/.test(threw), '授权不明 ⇒ E_LICENSE_UNRESOLVED（不在不明授权上训练）')
  threw = null
  try { assertTrainableLicense({ class: 'permissive' }, { sourceDir: '/x' }) } catch (e) { threw = String(e.message) }
  t.assert(threw && /E_LICENSE_UNRESOLVED/.test(threw), '有 class 但无 spdx 同样拒绝（缺字段就是缺授权来源）')

  // ── ③ permissive ⇒ 放行 ──
  const ok = assertTrainableLicense({ spdx: 'cc-by-4.0', class: 'permissive' }, { sourceDir: '/x' })
  t.assert(ok.spdx === 'cc-by-4.0', 'permissive 放行')

  // ── ④ discoverSource 跳过 share-alike 目录 ──
  const root = join(tmp, 'datasets')
  corpus(join(root, 'a_wikitext_sharealike'), { trainFile: 'calib.jsonl', valFile: 'test.jsonl', license: { spdx: 'cc-by-sa-3.0', class: 'share-alike' } })
  corpus(join(root, 'b_alpaca_permissive'), { trainFile: 'train.jsonl', valFile: 'heldout.jsonl', license: { spdx: 'cc-by-4.0', class: 'permissive' } })
  const disc = discoverSource(root)
  t.assert(disc !== null && disc.dir.endsWith('b_alpaca_permissive'), `自动发现选中 permissive 语料（实际 ${disc?.dir}）`)
  t.assert(disc.trainFile === 'train.jsonl' && disc.valFile === 'heldout.jsonl', '自动发现选中了正确的语料文件对')

  // 只有 share-alike 时：发现结果为 null（不"退而求其次"用禁用语料）
  const root2 = join(tmp, 'only-sa')
  corpus(join(root2, 'wikitext'), { trainFile: 'calib.jsonl', valFile: 'test.jsonl', license: { spdx: 'cc-by-sa-3.0', class: 'share-alike' } })
  t.assert(discoverSource(root2) === null, '只有 share-alike 语料时 ⇒ discoverSource 返回 null（不回退到禁用语料）')

  // ── ⑤ prepareQatData 端到端：share-alike 来源被拒 ──
  threw = null
  try {
    await prepareQatData({ sourceDir: join(root, 'a_wikitext_sharealike'), datasetsRoot: root, outDir: join(tmp, 'o1'), train: 20, val: 10 })
  } catch (e) { threw = String(e.message) }
  t.assert(threw && /E_LICENSE_NOT_TRAINABLE/.test(threw), 'prepareQatData 从来源血缘读到 share-alike ⇒ 拒绝')

  // ── ⑥ license **继承**：产出血缘里的 license 与来源逐字一致 ──
  const srcLic = { spdx: 'cc-by-4.0', class: 'permissive', source: 'https://example.invalid/card', evidence: ['e1', 'e2'] }
  corpus(join(root, 'c_with-evidence'), { trainFile: 'train.jsonl', valFile: 'heldout.jsonl', license: srcLic })
  const out = await prepareQatData({ sourceDir: join(root, 'c_with-evidence'), datasetsRoot: root, outDir: join(tmp, 'o2'), train: 20, val: 10 })
  t.assert(JSON.stringify(out.license) === JSON.stringify(srcLic), '产出 license 与来源**逐字一致**（含 evidence 链，本包不自己写 SPDX）')
  t.assert(out.lineage.license.source === srcLic.source, '血缘里的 license.source 可指认')
  t.assert(JSON.stringify(readSourceLicense(join(root, 'c_with-evidence'))) === JSON.stringify(srcLic), 'readSourceLicense 从来源血缘读回')

  // ── ⑦ 产出目录的 lineage.json 也带许可（C20b 的正面防线）──
  const lg = JSON.parse((await import('node:fs')).readFileSync(join(out.out, 'lineage.json'), 'utf8'))
  t.assert(lg.license?.spdx === 'cc-by-4.0', '产出 lineage.json 带 license.spdx')
  t.assert(lg.allClear === true, '产出 lineage.json 带 allClear')

  rmSync(tmp, { recursive: true, force: true })
}
