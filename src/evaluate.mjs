/**
 * evaluate —— training-qat 的**评估适配层**（任务 A：合并四份 evaluate.mjs 之后）。
 *
 * ## 边界（**不绕开共享库契约**）
 * 合并前本文件是一份 203 行的**实现**，与 training-sft / training-pretrain / training-rl
 * 的同名文件是**同一算法、四个所有者**。现在：
 *
 *   - 端点原语与通用评估（`post` / `get` / `evalLM` / `evalUniform`）
 *     **只有一份实现**：`@xiciyang/testing-core`（`src/evaluate.mjs`）。
 *     本文件只做**再导出**（`export … from`）。
 *   - 本包**特有**的东西留在这里：过拟合参考实现的定位与加载（`criterionPath` /
 *     `loadCriterion`）、解码吞吐 `evalThroughput`、QAT 门禁 `qatGate`、困惑度退化 `dropPctOf`。
 *
 *   双重门禁 / 相对变化 / 均匀签名 / 困惑度  → `@xiciyang/testing-core`
 *   过拟合判据（O2/O3）与多种子聚合（O5）    → `inference-training/checks/overfit-criterion.mjs`
 *                                              （`training-eval-contract` §4 的**唯一参考实现**，
 *                                                本包**不重写一份**：重写就会出现
 *                                                "契约写一套、脚本算另一套"）
 *   本文件只做：**指标抽取**（经 serving 端点）、装配、以及把数交给上面两处判。
 *
 * ## 三条硬纪律
 *   ① **判定必须走 serving**（training-eval-contract §5.1）：容器内评估只能作选点信号，
 *      且必须标 `selectionOnly: true`。
 *   ② **先判测量成立，再判数值**：`dualGate` 的 validity 项不过 ⇒ 不得据此下"达标/未达标"结论。
 *   ③ **相对量必须有基线**；`baseline === 0` 由 testing-core 响亮拒绝（不许 Infinity 冒充结论）。
 */
import { post, dualGate, uniformSignature, relativeChange, perplexity, mean, meanNllFromLogprobs, CONTRACT_VERSION as TESTING_CORE_VERSION } from '@xiciyang/testing-core'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { existsSync } from 'node:fs'

const HERE = dirname(fileURLToPath(import.meta.url))
export const PKG_ROOT = join(HERE, '..')
const WORKSPACE = process.env.DSH_WORKSPACE || join(PKG_ROOT, '..', '..')

/** overfit-criterion 的路径（**从包位置推导**，不硬编码本机绝对路径）。 */
export function criterionPath() {
  const cands = [
    process.env.TRAINING_QAT_CRITERION,
    join(WORKSPACE, 'inference-training', 'checks', 'overfit-criterion.mjs'),
    join(PKG_ROOT, '..', 'checks', 'overfit-criterion.mjs'),
  ].filter(Boolean)
  for (const c of cands) if (existsSync(c)) return c
  return null
}

/**
 * **唯一参考实现**必须可加载；加载不到就 fail-closed。
 * 不允许"自己再实现一遍"作为兜底 —— 那正是契约 §6 第 7 条要防的。
 */
export async function loadCriterion() {
  const p = criterionPath()
  if (!p) {
    throw new Error('E_CRITERION_MISSING: 找不到 overfit-criterion.mjs（training-eval-contract §4 的唯一参考实现）。'
      + '本包**不自己实现**该判据 —— 宁愿失败也不制造第二份实现。')
  }
  return { path: p, mod: await import(p) }
}

export const EVAL_PROTOCOL = 'qat-eval@1'

/* ------------------------------------------------------- 通用评估（唯一实现在 testing-core） */

export { get, evalLM, evalUniform } from '@xiciyang/testing-core'
// `post` 在本文件内部被 `evalThroughput` 使用（import 绑定），按原导出面再导出
export { post, dualGate, uniformSignature, relativeChange, perplexity, mean, meanNllFromLogprobs, TESTING_CORE_VERSION }

/* ------------------------------------------------------------------ 本包特有 */

/** 解码吞吐（供可选的性能门禁；单并发、温度 0、固定 seed）。 */
export async function evalThroughput({ endpoint, model, requests = 8, maxTokens = 32 }) {
  const prompt = 'The history of the province begins with the arrival of settlers who built'
  let outTokens = 0
  let ms = 0
  let nOk = 0
  for (let i = 0; i < requests; i += 1) {
    const t0 = Date.now()
    // 合并前本包 `post` 的默认超时是 180s（另外三个包是 120s）。共享实现取 120s，
    // 故这里**显式**给出 180s，保持本包这一条实际行为不变（不是靠默认值碰巧一致）。
    // eslint-disable-next-line no-await-in-loop
    const res = await post(endpoint, '/v1/completions', { model, prompt, max_tokens: maxTokens, temperature: 0, seed: 0 }, { timeoutMs: 180000 })
    ms += Date.now() - t0
    if (res.status === 200) {
      nOk += 1
      outTokens += Number(res.json?.usage?.completion_tokens ?? 0)
    }
  }
  // **缺陷 14 记账**：`attempted/succeeded/failed/partial` 必须落进产物 ——
  // 否则"端点中途失效（failed 高）"与"本来只发了几个请求"在**落盘后不可区分**。
  return {
    requests, nOk, outTokens, ms, decodeTps: ms > 0 ? outTokens / (ms / 1000) : null, maxTokens,
    attempted: requests, succeeded: nOk, failed: requests - nOk, skipped: 0,
    partial: requests > 0 && nOk < requests,
  }
}

/* ------------------------------------------------------------------ 门禁 */

/**
 * QAT 的门禁。
 *
 * **形状取自 `training-qat-contract` §4.1**：QAT 验收不是"达到某绝对精度"，而是
 * 「在与 PTQ 最优档**相同的量化方案**下，使 `dropPct` ≤ 调用方阈值」。
 * 所以 precision 判据是**精度**（`dropPct`），性能项只有在调用方给出
 * `minPerfGainPct` 时才纳入（同 scheme ⇒ 压缩比相同 ⇒ 收益可比，性能不是 QAT 的职责）。
 *
 * 判定逻辑**全部来自 `@xiciyang/testing-core.dualGate`**，本文件不重写一份。
 */
export function qatGate({ dropPct, gainPct = null, validity, thresholds }) {
  // **性能维只在调用方给了目标时才纳入判定。**
  // 为什么要这么写（本机实测踩到）：早先的实现按 `gainPct === null` 决定是否纳入 perf，
  // 于是"调用方没给 minPerfGainPct（null）"会变成 `perf: {gainPct}` + `threshold: null` ⇒
  // `dualGate` 判 `ok=false`，报告里读起来像"性能不达标"，实际是"**根本没定性能目标**"。
  // 这是一条会误导人的记录，必须消除：没目标 ⇒ 不判该维（同时把实测值照记在 metrics 里）。
  const judgePerf = gainPct !== null && Number.isFinite(thresholds?.minPerfGainPct)
  return dualGate({
    validity,
    accuracy: { dropPct },
    perf: judgePerf ? { gainPct } : null,
    maxAccuracyDropPct: thresholds.maxAccuracyDropPct,
    minPerfGainPct: judgePerf ? thresholds.minPerfGainPct : undefined,
    // 说明：`perfJudged=false` 时 gate.checks 里不会出现 perf 项，且下面会把原因记进结果。
  })
}

/**
 * 相对困惑度退化（**与 `inference-quantization/src/testing/parity.mjs:97` 同一公式**）。
 * 刻意调 testing-core 的 `relativeChange` 而不是自己写公式（唯一实现）。
 */
export function dropPctOf({ basePpl, candPpl }) {
  return relativeChange({ baseline: basePpl, candidate: candPpl, direction: 'lower-is-better' }).rawPct
}
