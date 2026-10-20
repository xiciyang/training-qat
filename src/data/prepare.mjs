/**
 * data/prepare —— QAT 语料准备与**两道闸门**：
 *   ① `license-whitelist.md` §2.1 的硬边界：**share-alike 语料不得当训练语料引入**；
 *   ② `training-eval-contract` §4.1 的 **O1 去泄漏**（双轨）。
 *
 * 切分/去重/去泄漏/血缘**全部**走 `@xiciyang/data-core`（不绕开共享库契约）。
 *
 * ## O1（硬，无先验、完全可判定）
 *   ① train 与 val 的**内容哈希交集必须为空**（整行 sha256 相等）；
 *   ② **且** val 任一行文本（长度 ≥ 64）不得作为子串出现在任何 train 行内部（双轨）；
 *   任一非空 ⇒ `allClear=false` ⇒ 调用方拒绝训练（fail-closed）。
 * 为什么双轨：整行哈希相等只判"逐行相同"，判不出"val 文本被**嵌在** train 行内部"。
 *
 * ## license 闸门（**2026-09-29 阶段 4 真机跑出来的**）
 * 第一次真机 QAT 用的是 wikitext（`Salesforce/wikitext`，`cc-by-sa-3.0` + `gfdl` 双许可）。
 * 但 `license-whitelist.md` §2.1 写死：**wikitext 只作校准集与评估集，不得当训练语料引入**。
 * 那次运行的处置是**改正**而不是给规则开例外：
 *   - 本文件新增 `class === 'share-alike'` 的**训练语料拒绝**（fail-closed）；
 *   - license **一律从来源血缘继承**（`source lineage.json`），
 *     **绝不由本包自己写一个 SPDX 值** —— 这正是"license 以实际卡片为准"的落点；
 *   - 默认语料改为 `yahma/alpaca-cleaned`（`cc-by-4.0`，permissive，可训练）。
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync } from 'node:fs'
import { join } from 'node:path'

import { contentHash, recordHash, dedupByContent, splitDeterministic, assertNoLeakage, lineage } from '@xiciyang/data-core'

const MIN_CONTAINMENT_LEN = 64

function readJsonl(p) {
  const out = []
  const text = readFileSync(p, 'utf8')
  for (const line of text.split('\n')) {
    const s = line.trim()
    if (!s) continue
    try { out.push(JSON.parse(s)) } catch { /* 跳过坏行 */ }
  }
  return out
}

/**
 * 一行语料的**文本表示**（QAT 就在这段文本上做假量化感知训练）。
 * - 纯文本语料：`t` / `text`；
 * - 指令语料：与 `training-sft` 的 `PROMPT_TMPL` **逐字一致**，否则"训练格式"与"评估格式"会对不上。
 */
export function textOf(r) {
  if (typeof r?.t === 'string' && r.t) return r.t
  if (typeof r?.text === 'string' && r.text) return r.text
  if (typeof r?.instruction === 'string') {
    return '### Instruction:\n' + String(r.instruction).trim() + '\n\n'
      + (String(r.input ?? '').trim() ? '### Input:\n' + String(r.input).trim() + '\n\n' : '')
      + '### Response:\n' + String(r.output ?? '')
  }
  return ''
}

/** 语料文件对：优先指令语料（train/heldout），退回量化侧的（calib/test）。 */
export function corpusFiles(dir) {
  for (const [trainFile, valFile] of [['train.jsonl', 'heldout.jsonl'], ['calib.jsonl', 'test.jsonl'], ['train.jsonl', 'val.jsonl']]) {
    if (existsSync(join(dir, trainFile)) && existsSync(join(dir, valFile))) return { trainFile, valFile }
  }
  return null
}

export function readSourceLicense(dir) {
  const p = join(dir, 'lineage.json')
  if (!existsSync(p)) return null
  try {
    const doc = JSON.parse(readFileSync(p, 'utf8'))
    // `data-core.lineage()` 把 extra **平铺到顶层**，故 license 在顶层
    return doc?.license ?? null
  } catch { return null }
}

/** share-alike（含 gfdl 双许可）**不得**当训练语料 —— `license-whitelist.md` §2.1 的硬边界。 */
export function assertTrainableLicense(license, { sourceDir }) {
  if (!license || typeof license !== 'object' || !license.spdx) {
    throw new Error(`E_LICENSE_UNRESOLVED: 无法确定语料 license（source=${sourceDir}）：`
      + 'QAT 会更新权重，属**训练**。拒绝在授权不明的语料上训练 —— 请提供带 license.spdx 的来源血缘，或显式传 sourceLicense。')
  }
  if (license.class === 'share-alike') {
    throw new Error(`E_LICENSE_NOT_TRAINABLE: 语料 license=${license.spdx}（class=share-alike）**不得当作训练语料**`
      + '（license-whitelist.md §2.1 的硬边界：share-alike 语料只可作校准集 / 评估集，不可用于训练权重）。'
      + `source=${sourceDir}。请改用可训练语料（例如 yahma/alpaca-cleaned，cc-by-4.0）。`)
  }
  return license
}

/**
 * 在状态目录里自动发现最近的**可训练**语料（离线、确定性）。
 * 优先级：permissive > 其它非 share-alike；share-alike 直接跳过（并记录原因）。
 */
export function discoverSource(datasetsRoot) {
  if (!existsSync(datasetsRoot)) return null
  const cands = []
  for (const d of readdirSync(datasetsRoot).sort()) {
    const dir = join(datasetsRoot, d)
    const files = corpusFiles(dir)
    if (!files) continue
    const license = readSourceLicense(dir)
    cands.push({ dir, ...files, license, spdx: license?.spdx ?? null, class: license?.class ?? null })
  }
  const trainable = cands.filter((c) => c.spdx && c.class !== 'share-alike')
  const perm = trainable.filter((c) => c.class === 'permissive')
  const pool = perm.length ? perm : trainable
  return pool.length ? pool[pool.length - 1] : null
}

/**
 * @returns {{out, trainRows, valRows, verify, allClear, lineage, source}}
 */
export async function prepareQatData({
  sourceDir = null, sourceLicense = null, datasetsRoot, outDir = null, seed = 20260929,
  train = 128, val = 64, force = false, write = true, log = () => {},
}) {
  const disc = sourceDir ? { dir: sourceDir, ...(corpusFiles(sourceDir) ?? {}) } : discoverSource(datasetsRoot)
  if (!disc || !disc.dir) {
    throw new Error(`找不到 QAT 语料源：请用 sourceDir 显式指定一个含语料文件对的目录（已查 ${datasetsRoot}）`)
  }
  const src = disc.dir
  const { trainFile, valFile } = disc
  if (!trainFile || !valFile) throw new Error(`语料源缺文件对（需要 train/heldout 或 calib/test）：${src}`)
  const trainPath = join(src, trainFile)
  const valPath = join(src, valFile)

  // ── license 闸门（**继承来源，不自己写 SPDX**）──
  const license = sourceLicense ?? disc.license ?? readSourceLicense(src)
  assertTrainableLicense(license, { sourceDir: src })
  log(`语料 license = ${license.spdx}（class=${license.class ?? 'n/a'}，继承自来源血缘）`)

  const rawTrain = readJsonl(trainPath)
  const rawVal = readJsonl(valPath)
  const keyOf = (r) => textOf(r)

  const steps = []
  const dTrain = dedupByContent(rawTrain, { keyOf })
  const dVal = dedupByContent(rawVal, { keyOf })
  steps.push({ step: 'dedup', train: { in: rawTrain.length, out: dTrain.kept.length, dropped: dTrain.dropped.length },
    val: { in: rawVal.length, out: dVal.kept.length, dropped: dVal.dropped.length } })

  const effTrain = Math.min(train, dTrain.kept.length)
  const effVal = Math.min(val, dVal.kept.length)
  const shortfall = { train: train - effTrain, val: val - effVal }
  if (shortfall.train > 0 || shortfall.val > 0) {
    steps.push({ step: 'split-shortfall', note: '语料行数少于请求 ⇒ 已缩水；缩水幅度在此如实记账（不静默）', requested: { train, val }, effective: { train: effTrain, val: effVal } })
  }

  const sp = splitDeterministic(dTrain.kept, { seed, calib: effTrain, test: 0, keyOf })
  const spVal = splitDeterministic(dVal.kept, { seed: seed + 1, calib: effVal, test: 0, keyOf })
  const trainRows = sp.calib
  const valRows = spVal.calib
  steps.push({ step: 'split', seed, requested: { train, val }, effective: { train: trainRows.length, val: valRows.length },
    shortfall, trainSource: trainPath, valSource: valPath })

  // O1 轨 1：内容哈希交集必须为空
  const hashCheck = assertNoLeakage(trainRows, valRows, { keyOf })
  const verify = [{ pair: 'train × val', track: 'content-sha256', intersection: hashCheck.intersection.length, ok: hashCheck.ok }]

  // O1 轨 2：val 文本不得作为子串出现在任何 train 行内部
  const trainTexts = trainRows.map(keyOf)
  const contained = []
  for (const r of valRows) {
    const t = keyOf(r)
    if (t.length < MIN_CONTAINMENT_LEN) continue
    if (trainTexts.some((x) => x.includes(t))) contained.push(`sha256:${contentHash(t).slice(0, 16)}`)
  }
  verify.push({ pair: 'train × val', track: 'substring-containment', minLen: MIN_CONTAINMENT_LEN, hits: contained.length, ok: contained.length === 0 })

  const crossDup = new Set(valRows.map((r) => recordHash(r, keyOf)))
  const overlapWithVal = trainRows.filter((r) => crossDup.has(recordHash(r, keyOf))).length
  verify.push({ pair: 'train × val（recordHash 复核）', track: 'record-hash', hits: overlapWithVal, ok: overlapWithVal === 0 })

  const allClear = verify.every((v) => v.ok)
  const OUT = outDir || join(datasetsRoot, `qat__${src.split('/').filter(Boolean).pop()}__s${seed}__train${train}-val${val}`)

  const lg = lineage({
    source: `QAT 语料（继承来源）: ${src}`,
    revision: null,
    seed,
    calib: trainRows,
    test: valRows,
    extra: {
      // **继承**来源的 license 对象（含 evidence 链），本包不写新的 SPDX 值
      license,
      licenseInheritedFrom: join(src, 'lineage.json'),
      sourceDir: src,
      sourceFiles: { train: trainPath, val: valPath },
      sourceHashes: { train: `sha256:${contentHash(readFileSync(trainPath, 'utf8'))}`, val: `sha256:${contentHash(readFileSync(valPath, 'utf8'))}` },
      purpose: 'QAT 的假量化感知微调语料 + 留出集（O1 双轨去泄漏；license 已过 not-share-alike 闸门）',
      pipeline: steps,
      verification: verify,
      allClear,
      criterion: 'O1: content-sha256 intersection empty AND no val-substring containment in train;'
        + ' license-whitelist §2.1: share-alike 不得作训练语料',
    },
  })

  if (write) {
    mkdirSync(OUT, { recursive: true, mode: 0o700 })
    const line = (r) => JSON.stringify({ t: keyOf(r), h: recordHash(r, keyOf) })
    writeFileSync(join(OUT, 'train.jsonl'), `${trainRows.map(line).join('\n')}\n`, { mode: 0o600 })
    writeFileSync(join(OUT, 'val.jsonl'), `${valRows.map(line).join('\n')}\n`, { mode: 0o600 })
    writeFileSync(join(OUT, 'lineage.json'), `${JSON.stringify(lg, null, 2)}\n`, { mode: 0o600 })
  }

  return { out: OUT, source: src, sourceFiles: { train: trainPath, val: valPath }, license, steps, verify, allClear, lineage: lg,
    trainRows: trainRows.length, valRows: valRows.length,
    trainTexts: trainRows.map(keyOf), valTexts: valRows.map(keyOf) }
}

/** 独立复核一个已备好的 QAT 数据集（只读；validate 用它做 O1 的现场重算）。 */
export function reverify(dir) {
  const trainPath = join(dir, 'train.jsonl')
  const valPath = join(dir, 'val.jsonl')
  if (!existsSync(trainPath) || !existsSync(valPath)) return { ok: false, reason: `缺 train.jsonl/val.jsonl：${dir}` }
  const tr = readJsonl(trainPath).map((r) => r.t)
  const va = readJsonl(valPath).map((r) => r.t)
  const hashIntersection = []
  const vaSet = new Set(va.map((t) => contentHash(t)))
  for (const t of tr) if (vaSet.has(contentHash(t))) hashIntersection.push(`sha256:${contentHash(t).slice(0, 16)}`)
  const contained = []
  for (const t of va) {
    if (t.length < MIN_CONTAINMENT_LEN) continue
    if (tr.some((x) => x.includes(t))) contained.push(`sha256:${contentHash(t).slice(0, 16)}`)
  }
  const srcHash = (p) => `sha256:${contentHash(readFileSync(p, 'utf8'))}`
  const lg = (() => { try { return JSON.parse(readFileSync(join(dir, 'lineage.json'), 'utf8')) } catch { return null } })()
  return {
    ok: hashIntersection.length === 0 && contained.length === 0,
    track1: { name: 'content-sha256-intersection', hits: hashIntersection.length },
    track2: { name: 'substring-containment', minLen: MIN_CONTAINMENT_LEN, hits: contained.length },
    trainRows: tr.length, valRows: va.length,
    trainFileHash: srcHash(trainPath), valFileHash: srcHash(valPath),
    license: lg?.license ?? null,
    trainable: lg?.license ? lg.license.class !== 'share-alike' : null,
    criterion: 'O1: content-sha256 intersection empty AND no val-substring containment in train',
  }
}
