/**
 * weight-fingerprint.spec —— QAT 的**权重指纹守卫**回归。
 *
 * ## 背景（本缺口的根因，写在测试里免得被忘）
 * QAT 的 checkpoint 在训练**过程中**写，而当时 `qat-float/` **还没有落盘**。
 * 旧实现注入的钩子是 `fingerprintArtifacts(join(runDir,'qat-float')).hash` ⇒
 * 空目录的 `hash` 是 `null` ⇒ payload 里 `artifactFingerprint: null`
 * ⇒ **QAT 完全没有"空操作训练"防线**（CI `fingerprint-coverage` 实测报红）。
 *
 * 修法：改成向容器要**内存指纹**（`cmd_fingerprint`，只哈希 `requires_grad` 参数）。
 *
 * ## 本 spec 断言什么
 *   ① **判据自检**：用**线上真实 payload**（缺口期产物）作"必须被拒"的用例 ——
 *      守卫空转时这一条最先红（"检查没红"必须先排除"检查没看对象"）。
 *   ② 缺指纹 ⇒ `E_WEIGHT_FINGERPRINT_MISSING`
 *   ③ **反向对照**：指纹相同 ⇒ `E_NO_OP_TRAINING`（权重未变必须判失败）
 *   ④ 单体档位的字段名（`artifactFingerprint`/`initArtifactFingerprint`）必须被认
 *   ⑤ 两种病因的码必须可区分
 */
import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { homedir } from 'node:os'
import { assertWeightFingerprint } from '../../src/core.mjs'

export const NAME = '权重指纹守卫（空操作训练防线·qat）'

/**
 * **动态发现**最新的 QAT checkpoint payload —— 不写死 run id。
 *
 * 硬编码会有两个毛病：① 换一台机器/清一次 run 就 ENOENT；
 * ② 修好之后"必须被拒"这条断言会**永久失效**（新 payload 是有指纹的）。
 * 改成"发现真对象 + 断言守卫的判定**与它的实际状态一致**"：
 * 修复前它必须被拒，修复后它必须被接受 —— **两种情况下都在看真东西**。
 */
function newestQatPayload() {
  const base = join(homedir(), '.ai-infra/run/training-qat')
  if (!existsSync(base)) return null
  const runs = readdirSync(base).filter((d) => statSync(join(base, d)).isDirectory()).sort().reverse()
  for (const run of runs) {
    const cps = join(base, run, 'checkpoints')
    if (!existsSync(cps)) continue
    for (const cp of readdirSync(cps).sort().reverse()) {
      const f = join(cps, cp, 'payload.json')
      if (existsSync(f)) return { path: f, payload: JSON.parse(readFileSync(f, 'utf8')) }
    }
  }
  return null
}

const expectThrow = (fn, code) => {
  try { fn(); return { ok: false, detail: '未抛出' } }
  catch (e) { const m = String(e?.message ?? ''); return m.includes(code) ? { ok: true } : { ok: false, detail: `码不符：${m.slice(0, 100)}` } }
}

export async function run(t) {
  t.assert(typeof assertWeightFingerprint === 'function', '守卫必须可被 import —— 拿不到就是在测空气')

  /* ① 判据自检：线上**真实** payload，守卫的判定必须与它的实际状态一致 */
  const found = newestQatPayload()
  if (found === null) {
    t.skip('磁盘上没有 QAT checkpoint payload（run 已被清理）—— 判据自检没有对象可看；守卫行为由 ②–⑤ 的合成用例继续覆盖')
  }
  if (found !== null) {
  const real = found.payload
  const fp = real.artifactFingerprint ?? real.fingerprint
  const init = real.initArtifactFingerprint ?? real.initFingerprint
  const hasBoth = typeof fp === 'string' && fp !== '' && typeof init === 'string' && init !== ''
  if (hasBoth) {
    let ok = false
    try { ok = assertWeightFingerprint(real).ok === true } catch { ok = false }
    t.assert(ok, `① 真实 payload 有完整指纹（${fp.slice(0, 20)}…）⇒ 守卫必须**接受**它（这是修复后的取证）`)
  } else {
    const r1 = expectThrow(() => assertWeightFingerprint(real), 'E_WEIGHT_FINGERPRINT_MISSING')
    t.assert(r1.ok, `① 真实 payload 无完整指纹（fp=${JSON.stringify(fp)}）⇒ 守卫必须**拒绝**它（缺口期形态）：${r1.detail}`)
  }
  }

  /* ② 缺指纹 */
  for (const bad of [{}, { artifactFingerprint: 'sha256:aa' }, { initArtifactFingerprint: 'sha256:bb' }, { artifactFingerprint: null, initArtifactFingerprint: null }]) {
    const r = expectThrow(() => assertWeightFingerprint(bad), 'E_WEIGHT_FINGERPRINT_MISSING')
    t.assert(r.ok, `② 缺指纹必须拒绝（${JSON.stringify(bad)}）：${r.detail}`)
  }

  /* ③ 反向对照：相同 ⇒ 空操作 */
  const same = { artifactFingerprint: 'sha256:same', initArtifactFingerprint: 'sha256:same' }
  t.assert(expectThrow(() => assertWeightFingerprint(same), 'E_NO_OP_TRAINING').ok, '③ 指纹相同必须判空操作（反向对照）')

  /* ④ 单体档位字段名必须被认 */
  let out = null
  try { out = assertWeightFingerprint({ artifactFingerprint: 'sha256:1111', initArtifactFingerprint: 'sha256:2222' }) } catch (e) { out = { error: String(e.message) } }
  t.assert(out?.ok === true, `④ 单体档位字段名必须通过（实得 ${JSON.stringify(out)}）`)
  t.assert(out?.after === 'sha256:1111' && out?.before === 'sha256:2222', '④ 必须回传 before/after')

  /* ⑤ 两种病因可区分 */
  const m1 = (() => { try { assertWeightFingerprint({}) } catch (e) { return String(e.message) } return '' })()
  const m2 = (() => { try { assertWeightFingerprint(same) } catch (e) { return String(e.message) } return '' })()
  t.assert(m1.includes('E_WEIGHT_FINGERPRINT_MISSING') && m2.includes('E_NO_OP_TRAINING') && m1 !== m2, '⑤ 两种病因必须给出不同的码与理由')
}
