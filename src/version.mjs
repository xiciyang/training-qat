/**
 * version / 错误码闭集 —— training-qat。
 *
 * 错误码是**闭集**（与同侧其余插件同一纪律）：每个码对应一个可独立失效、
 * 且失效语义不同的依赖源或闸门。不设"其他"桶。
 *
 * 文档修订记录（本文件即错误码的文档）：
 *   - 追加 `E_INSTRUMENT_LOSSY`：**additive**，未改任何既有码的语义、未改 SCHEMA_VERSION。
 *     语义 = "测量仪器经过 serving 的有损降级档位，故本次测量不得被签收"（fail-closed 闸门）。
 *   - 追加产物 dtype 三个码（**缺陷 16**）：**additive**，未改任何既有码的语义、未改 SCHEMA_VERSION。
 *     语义 = "产物的**存储 dtype** 是测量相关变量"（实测仅 F32→BF16 就使 serving 的 dropPct 变化 0.40pp）：
 *       `E_PRODUCT_DTYPE_UNKNOWN`      —— 读不出来（"未知 ≠ 合规"，不静默放行）
 *       `E_PRODUCT_DTYPE_VIOLATION`    —— 读到了明确的、非策略（bfloat16）的 dtype
 *       `E_DTYPE_COMPARISON_REFUSED`   —— 基线与候选的存储 dtype 不同，而调用方没有显式声明转换
 */
export const PACKAGE_VERSION = '0.1.0'
export const SCHEMA_VERSION = '1'
export const MCP_INTERFACE_VERSION = '1.0.0'
export const AUDIT_SCHEMA = 'training-qat/audit@1'
export const REGISTRY_SCHEMA = 'ai-infra/qat-registry@1'

export const ERROR_CODES = Object.freeze({
  // 参数与输入
  E_MISSING_PARAM: 'E_MISSING_PARAM',
  E_DATASET_INVALID: 'E_DATASET_INVALID',
  E_DATASET_LEAK: 'E_DATASET_LEAK',
  // license 闸门（license-whitelist §2.1：share-alike 不得作训练语料）
  E_LICENSE_UNRESOLVED: 'E_LICENSE_UNRESOLVED',
  E_LICENSE_NOT_TRAINABLE: 'E_LICENSE_NOT_TRAINABLE',
  // 依赖源
  E_BASE_MODEL_NOT_FOUND: 'E_BASE_MODEL_NOT_FOUND',
  E_IMAGE_UNAVAILABLE: 'E_IMAGE_UNAVAILABLE',
  E_SANDBOX_UNAVAILABLE: 'E_SANDBOX_UNAVAILABLE',
  E_SERVING_UNAVAILABLE: 'E_SERVING_UNAVAILABLE',
  E_QUANTIZER_UNAVAILABLE: 'E_QUANTIZER_UNAVAILABLE',
  // 预算
  E_BUDGET_EXCEEDED: 'E_BUDGET_EXCEEDED',
  // 状态
  E_STATE_CONFLICT: 'E_STATE_CONFLICT',
  E_RUN_NOT_FOUND: 'E_RUN_NOT_FOUND',
  E_CONFIRMATION_REQUIRED: 'E_CONFIRMATION_REQUIRED',
  // 产物级断言（"只断言过程抓不到空操作，必须断言权重"）
  E_NO_OP_TRAINING: 'E_NO_OP_TRAINING',
  E_ARTIFACT_FINGERPRINT_MISSING: 'E_ARTIFACT_FINGERPRINT_MISSING',
  E_OUTPUT_NOT_LOSSLESS: 'E_OUTPUT_NOT_LOSSLESS',
  // 量化语义（M3：方案必须诚实）
  E_SCHEME_UNSPECIFIED: 'E_SCHEME_UNSPECIFIED',
  E_SCHEME_DRIFT: 'E_SCHEME_DRIFT',
  // 门禁
  E_GATE_FAILED: 'E_GATE_FAILED',
  E_INSTRUMENT_LOSSY: 'E_INSTRUMENT_LOSSY',    // 测量仪器走了 serving 的有损降级档位（fail-closed）
  // 产物存储 dtype（缺陷 16：dtype 是测量相关变量 ⇒ 断言落在产物上、跨 dtype 一律拒绝）
  E_PRODUCT_DTYPE_UNKNOWN: 'E_PRODUCT_DTYPE_UNKNOWN',
  E_PRODUCT_DTYPE_VIOLATION: 'E_PRODUCT_DTYPE_VIOLATION',
  E_DTYPE_COMPARISON_REFUSED: 'E_DTYPE_COMPARISON_REFUSED',
  E_MEASUREMENT_INVALID: 'E_MEASUREMENT_INVALID',
  E_THRESHOLD_NOT_PROVIDED: 'E_THRESHOLD_NOT_PROVIDED',
  E_INTERNAL: 'E_INTERNAL',
})

export class ToolError extends Error {
  constructor(code, body = {}) {
    super(`${code}: ${body.hint || ''}`)
    this.name = 'ToolError'
    this.code = code
    this.body = body
  }
}

export const ALL_CODES = Object.freeze(Object.values(ERROR_CODES))
