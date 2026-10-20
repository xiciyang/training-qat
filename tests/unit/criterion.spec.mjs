/**
 * criterion.spec —— **裁决 B 回归测试**：QAT 的精度判据（`training-qat-contract` 2.0.0 §4.1.1）。
 *
 * 变更：由 O2「提升显著」（`improvementVal > z·SE`）改为**单侧**「退化不显著」（`improvementVal >= -z·SE`）。
 * 变更**不改任何阈值、不改 z、不改 O3**。
 *
 * 本 spec 全部**不依赖 GPU / 容器 / 网络**：只断言这一纯函数。
 * 它必须同时证明两件事：① 新口径在"保持住了"时通过；② **它没有被偷偷放宽** ——
 * "真的退化了"和"数据不足"都仍然不通过。
 */
import { degradationGateFor } from '../../src/core.mjs'

export const NAME = 'QAT 精度判据（裁决 B：单侧退化不显著）'

export async function run(t) {
  // ① 保持住了：improvementVal ≈ 0（略负但未超 z·SE）⇒ 通过
  t.assert(degradationGateFor({ improvementVal: 0, zSeVal: 0.01 }).pass === true, '零变化应通过（退化不显著）')
  t.assert(degradationGateFor({ improvementVal: -0.005, zSeVal: 0.01 }).pass === true, '退化未超 z·SE 应通过')
  t.assert(degradationGateFor({ improvementVal: 0.05, zSeVal: 0.01 }).pass === true, '明显提升当然应通过')

  // ② 边界：恰好等于 -z·SE ⇒ 通过（>= 是闭区间）；再小一点 ⇒ 不通过
  t.assert(degradationGateFor({ improvementVal: -0.01, zSeVal: 0.01 }).pass === true, '恰好 -z·SE 应通过（闭区间）')
  t.assert(degradationGateFor({ improvementVal: -0.0100001, zSeVal: 0.01 }).pass === false, '超过 -z·SE 应不通过')

  // ③ **没有被放宽**：真退化仍然失败（且与"提升显著"口径不同 —— 这正是本次修正的要点）
  const degraded = { improvementVal: -0.5, zSeVal: 0.01 }
  t.assert(degradationGateFor(degraded).pass === false, '真退化必须失败')
  t.assert(degraded.improvementVal > degraded.zSeVal === false, '（对照）旧 O2 口径对它也是失败')

  // ④ 数据不足 ⇒ `pass: null`，**不得**被当作通过（null ≠ true）
  t.assert(degradationGateFor({}).pass === null, '缺字段应返回 null 而不是 true')
  t.assert(degradationGateFor(null).pass === null, 'null 输入应返回 null')
  t.assert(degradationGateFor({ improvementVal: 0.1 }).pass === null, '缺 zSeVal 应返回 null')
  t.assert(degradationGateFor({ improvementVal: 0.1, zSeVal: 0.01 }).pass !== null, '字段齐全不得返回 null')

  // ⑤ 口径标识可追溯（契约 §4.1.1 要求）
  t.assert(degradationGateFor({ improvementVal: 0, zSeVal: 1 }).criterion === 'one-sided-non-degradation', 'criterion 标识必须是 one-sided-non-degradation')

  // ⑥ 只读性：函数不得修改入参（判据必须是纯的，否则重跑会漂）
  const frozen = Object.freeze({ improvementVal: -0.005, zSeVal: 0.01 })
  let threw = false
  try { degradationGateFor(frozen) } catch { threw = true }
  t.assert(threw === false, '不得修改入参（冻结对象传入不应抛错）')
}
