/**
 * dtype-policy.spec —— **缺陷 16 的正面防线**：产物的存储 dtype 是**测量变量**。
 *
 * 背景（真机实测，见 `DUAL_PATH_CALIBRATION_REPORT.md` §4）：QAT 产物把**未量化张量**
 * 存成 F32，基座与 PTQ 存 BF16。同一会话、同一数据集、同一 serving 仪器下，
 * **只把 dtype F32→BF16** 就使测得的 `dropPct` 从 **1.7205% 变成 1.3212%（0.40pp）**。
 * ⇒ dtype 必须被当作测量条件管起来：**策略固定 + 断言落在产物上 + 跨 dtype 一律拒绝**。
 *
 * 本 spec 全部用**合成产物**（自己写 safetensors 头）驱动真实判据 ——
 * 零 GPU、零容器、零网络、零真实权重；断言数因此是确定的，不随本机状态漂移。
 */
import { mkdtempSync, rmSync, writeFileSync, mkdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { tmpdir } from 'node:os'
import { fileURLToPath } from 'node:url'

import { DTYPE_POLICY, normalizeDtype, readTensorDtypes, assertProductDtype, assertComparableDtype } from '@xiciyang/training-core'

const HERE = dirname(fileURLToPath(import.meta.url))
const PKG = join(HERE, '..', '..')
void PKG
export const NAME = 'dtype-policy（缺陷 16：产物存储 dtype 是测量变量）'

/** 造一个**只含合法头**的 safetensors 文件：8 字节小端头长 + JSON header。 */
function fakeSafetensors(file, counts) {
  const header = {}
  let i = 0
  for (const [dtype, count] of Object.entries(counts)) {
    for (let k = 0; k < count; k += 1) { header[`t${i}`] = { dtype, shape: [1], data_offsets: [0, 0] }; i += 1 }
  }
  const buf = Buffer.from(JSON.stringify(header), 'utf8')
  const len = Buffer.alloc(8)
  len.writeBigUInt64LE(BigInt(buf.length))
  writeFileSync(file, Buffer.concat([len, buf]))
}

function product(root, name, counts, config = {}) {
  const dir = join(root, name)
  mkdirSync(dir, { recursive: true })
  fakeSafetensors(join(dir, 'model.safetensors'), counts)
  writeFileSync(join(dir, 'config.json'), JSON.stringify({ model_type: 'qwen2', ...config }))
  return dir
}

const codeOf = (fn) => { try { fn(); return null } catch (e) { return e } }

export async function run(t) {
  const tmp = mkdtempSync(join(tmpdir(), 'tq-dtype-'))

  // ── ① 策略是显式常量（改它必须显式改，不许隐式漂移）──
  t.assert(DTYPE_POLICY.evaluationProduct === 'bfloat16', `策略 dtype = bfloat16（实得 ${DTYPE_POLICY.evaluationProduct}）`)
  t.assert(/BF16/.test(DTYPE_POLICY.why) && /0\.40pp/.test(DTYPE_POLICY.why), '策略自带理由（为什么是 BF16 + 实测效应量）')
  t.assert(normalizeDtype('BF16') === 'bfloat16' && normalizeDtype('torch.float32') === 'float32' && normalizeDtype('nope') === null,
    'dtype 归一化：认得出 safetensors/torch 写法；认不出 ⇒ null（不猜）')

  // ── ② 断言落在**产物**上：读头 → 判据 ──
  const bf16 = product(tmp, 'bf16', { BF16: 290, F8_E4M3: 168 })        // PTQ/基座的形状
  const f32 = product(tmp, 'f32', { F32: 290, F8_E4M3: 168 })           // 历史 QAT 产物的形状
  const mixed = product(tmp, 'mixed', { BF16: 2, F32: 3 })              // 浮点 dtype 不唯一
  const allQuant = product(tmp, 'allquant', { F8_E4M3: 8 })             // 没有未量化浮点
  const empty = product(tmp, 'empty', {})

  const d = {}
  for (const [k, dir] of Object.entries({ bf16, f32, mixed, allQuant, empty })) d[k] = await readTensorDtypes({ dir })

  t.assert(d.bf16.floatDtype === 'bfloat16' && d.bf16.floatTotal === 290, `读头：BF16×290 ⇒ bfloat16（实得 ${d.bf16.floatDtype}/${d.bf16.floatTotal}）`)
  t.assert(d.bf16.quantizedCounts.float8_e4m3fn === 168, `F8_E4M3 归入**量化**张量而不是未量化浮点（实得 ${JSON.stringify(d.bf16.quantizedCounts)}）`)
  t.assert(d.f32.floatDtype === 'float32', `读头：F32×290 ⇒ float32（实得 ${d.f32.floatDtype}）`)

  t.assert(assertProductDtype({ dtypes: d.bf16, what: '合成 BF16 产物' }).ok === true, 'BF16 产物 ⇒ 通过')
  const viol = codeOf(() => assertProductDtype({ dtypes: d.f32, what: '合成 F32 产物' }))
  t.assert(viol?.code === 'E_PRODUCT_DTYPE_VIOLATION', `F32 产物 ⇒ E_PRODUCT_DTYPE_VIOLATION（实得 ${viol?.code}）`)
  t.assert(/float32/.test(String(viol?.message)) && /bfloat16/.test(String(viol?.message)), '错误信息同时给出"实测"与"策略"两个 dtype（否则无法定位）')
  t.assert(/不得改写历史产物/.test(String(viol?.message)), '错误信息写明处置方向：重新导出并重测，**不改写历史产物**')
  t.assert(codeOf(() => assertProductDtype({ dtypes: d.mixed }))?.code === 'E_PRODUCT_DTYPE_UNKNOWN', '浮点 dtype 不唯一 ⇒ 未知 ⇒ 不通过')
  t.assert(codeOf(() => assertProductDtype({ dtypes: d.allQuant }))?.code === 'E_PRODUCT_DTYPE_UNKNOWN', '没有未量化浮点 ⇒ 未知 ⇒ 不通过')
  t.assert(codeOf(() => assertProductDtype({ dtypes: d.empty }))?.code === 'E_PRODUCT_DTYPE_UNKNOWN', '空产物 ⇒ 未知 ⇒ 不通过')
  const missingDir = await readTensorDtypes({ dir: join(tmp, 'does-not-exist') })
  t.assert(codeOf(() => assertProductDtype({ dtypes: missingDir }))?.code === 'E_PRODUCT_DTYPE_UNKNOWN', '目录不存在 ⇒ 未知 ⇒ 不通过（fail-closed）')

  // ── ③ 跨 dtype 的比较**一律拒绝**，除非有**显式声明**的转换 ──
  const same = assertComparableDtype({ baseline: { ref: bf16, dtype: d.bf16.floatDtype }, candidate: { ref: bf16, dtype: d.bf16.floatDtype } })
  t.assert(same.comparable === true && same.on === 'same-dtype' && same.judgedOn === 'bfloat16', `同 dtype ⇒ 可比且记下判定的 dtype（${JSON.stringify(same)}）`)

  const refused = codeOf(() => assertComparableDtype({ baseline: { ref: bf16, dtype: 'bfloat16' }, candidate: { ref: f32, dtype: 'float32' } }))
  t.assert(refused?.code === 'E_DTYPE_COMPARISON_REFUSED', `BF16 基线 vs F32 候选 ⇒ 拒绝比较（实得 ${refused?.code}）`)
  t.assert(/1\.7205%/.test(String(refused?.message)) && /1\.3212%/.test(String(refused?.message)),
    '拒绝信息带**实测效应量**（1.7205% → 1.3212%），让人能判断"这值不值得转换后重测"')
  t.assert(codeOf(() => assertComparableDtype({ baseline: { ref: bf16, dtype: null }, candidate: { ref: f32, dtype: 'float32' } }))?.code === 'E_PRODUCT_DTYPE_UNKNOWN',
    '任一侧 dtype 未知 ⇒ 拒绝（未知 ≠ 可比）')

  const conv = assertComparableDtype({
    baseline: { ref: bf16, dtype: 'bfloat16' }, candidate: { ref: bf16, dtype: 'bfloat16' },
    conversion: { from: 'float32', to: 'bfloat16', declaredBy: 'qat-validate', reason: 'out-of-band cast, then re-measured' },
  })
  t.assert(conv.on === 'declared-conversion' && conv.conversion.from === 'float32' && conv.conversion.to === 'bfloat16',
    '显式声明的转换被记进判定记录（from/to/declaredBy/reason 全留档）')
  t.assert(codeOf(() => assertComparableDtype({ baseline: { ref: bf16, dtype: 'bfloat16' }, candidate: { ref: bf16, dtype: 'bfloat16' }, conversion: { from: 'x', to: 'y', declaredBy: 'a', reason: 'b' } }))?.code === 'E_DTYPE_CONVERSION_INVALID',
    '声明里的 dtype 认不出 ⇒ 无效（"显式"不是"随便写一句话"）')
  t.assert(codeOf(() => assertComparableDtype({ baseline: { ref: bf16, dtype: 'bfloat16' }, candidate: { ref: bf16, dtype: 'bfloat16' }, conversion: { from: 'float32', to: 'bfloat16', declaredBy: '', reason: '' } }))?.code === 'E_DTYPE_CONVERSION_INVALID',
    '声明缺 declaredBy/reason ⇒ 无效（谁declare的、为什么，必须是记录的一部分）')

  // ── ④ 策略常量与容器侧**必须同名**（跨语言对账的前提）──
  const src = await import('node:fs').then((fs) => fs.readFileSync(join(PKG, 'python', 'qat_trainer.py'), 'utf8'))
  t.assert(new RegExp(`PRODUCT_DTYPE\\s*=\\s*os\\.environ\\.get\\("QAT_PRODUCT_DTYPE",\\s*"${DTYPE_POLICY.evaluationProduct}"\\)`).test(src),
    `容器侧默认 dtype 与判定侧策略同名（${DTYPE_POLICY.evaluationProduct}）`)
  t.assert(/torch_dtype"\]\s*=\s*PRODUCT_DTYPE/.test(src), '容器侧把 dtype **写进产物自己的 config.json**（产物自描述，不是只写在日志里）')
  t.assert(/E_PRODUCT_DTYPE_VIOLATION/.test(src) && /convertedFrom/.test(src), '容器侧转换后**读回核对**并把 convertedFrom 记进返回值（不静默转换）')

  rmSync(tmp, { recursive: true, force: true })
}
