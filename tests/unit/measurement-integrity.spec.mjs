/**
 * measurement-integrity.spec —— **缺陷 14 回归测试（真实失败路径，无 GPU / 无容器）**。
 *
 * 与 sft 的同名 spec 不同：这里**不造假数据**，而是把 `evalLM` 指向一个**确定连不上的端点**，
 * 让它在真实网络失败下把 attempted / failed / partial 记出来。
 *
 * 为什么必须测真实失败路径：缺陷 14 的本质是"**失败被静默吞掉**"。
 * 只测合成夹具无法证明"真失败时确实记账"—— 因为合成的失败对象是**我们**构造的，
 * 而不是 `post()` 在真实 ECONNREFUSED 下产出的。
 */
import { evalLM } from '../../src/evaluate.mjs'

export const NAME = '测量完整性（缺陷 14 回归·真实失败路径）'

const DEAD = 'http://127.0.0.1:9'   // discard 端口：本机未监听 ⇒ 连接立即被拒
const mkRows = (n) => Array.from({ length: n }, (_, i) => ({ t: `row ${i} 这是一段用于触发前向的文本` }))

export async function run(t) {
  // ① 端点失效：5 条全部失败 ⇒ 必须**如实记账**，而不是返回"texts=0"了事
  const bad = await evalLM({ endpoint: DEAD, model: 'none', rows: mkRows(5), limit: 5 })
  t.assert(bad.attempted === 5, `attempted 应为 5（真发出去 5 次），实得 ${bad.attempted}`)
  t.assert(bad.succeeded === 0, `succeeded 应为 0，实得 ${bad.succeeded}`)
  t.assert(bad.failed === 5, `failed 应为 5，实得 ${bad.failed}`)
  t.assert(bad.partial === true, 'partial 必须为 true（success < attempted）')
  t.assert(bad.texts === 0, 'texts（向后兼容字段）应为 0')

  // ② 必须能**定位**：firstError 要带原始下标、状态与细节
  t.assert(bad.firstError !== null && bad.firstError !== undefined, 'firstError 必须存在（否则无法定位失败原因）')
  t.assert(bad.firstError?.index === 0, 'firstError.index 应为首个失败行的下标 0')
  t.assert(typeof bad.firstError?.status === 'number', 'firstError.status 必须是数字（连接失败时为 0）')
  t.assert(typeof bad.firstError?.detail === 'string' && bad.firstError.detail.length > 0, 'firstError.detail 必须非空（含错误摘要）')

  // ③ **可区分"端点失效"与"评估集小"** —— 这是缺陷 14 的核心要求
  const small = await evalLM({ endpoint: DEAD, model: 'none', rows: mkRows(2), limit: 2 })
  t.assert(small.attempted === 2 && small.failed === 2, '小评估集：attempted=2、failed=2')
  t.assert(bad.attempted === 5 && small.attempted === 2, '两者的 attempted 不同 ⇒ "本来要看多少"是可读的')
  t.assert(bad.considered === 5 && small.considered === 2, 'considered 必须反映评估集大小（与 attempted 独立）')

  // ④ 空文本必须记 `skipped`，**不计入 attempted**（否则会把"跳过"混进"失败"）
  const mixed = await evalLM({ endpoint: DEAD, model: 'none', rows: [{ t: '  ' }, { t: 'x 有内容' }, { t: '' }], limit: 3 })
  t.assert(mixed.skipped === 2, `空白行应记 skipped=2，实得 ${mixed.skipped}`)
  t.assert(mixed.attempted === 1, `只有 1 行有内容 ⇒ attempted=1，实得 ${mixed.attempted}`)
  t.assert(mixed.considered === 3, 'considered 应仍是 3（看过 3 行）')

  // ⑤ okIndices 必须与 perText 等长（配对依据不能与实际样本脱节）
  t.assert(Array.isArray(bad.okIndices) && bad.okIndices.length === bad.perText.length, 'okIndices 长度必须等于 perText 长度')
}
