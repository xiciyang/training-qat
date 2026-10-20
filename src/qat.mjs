/**
 * qat —— **单体档位**（monolithic）的 QAT 插件对象。
 *
 * 为什么是单体档位：QAT 的循环属于**框架**（容器内的 torch：前向 → 反向 → AdamW），
 * Node 侧不持有张量、也不该逐微批搬运数据。`training-core` v0.2.0 为此提供了
 * `{ id, monolithic: true, run(ctx) }`：插件必须**在每个优化器步**调用一次 `ctx.onStep(info)`，
 * core 事后复核实际步数/墙钟/GPU 小时（**0 步也算违规**），并触发 checkpoint 与钩子。
 *
 * 与逐步档位的等价关系（`training-core` CONTRACT §判据）：
 *   hooks.afterStep ↔ ctx.onStep      type:checkpoint ↔ ctx.onCheckpoint
 *   enforceLimits 命中 ↔ onBudgetExceeded
 *
 * **步号一致性不变量**：容器回 `stepIndex`，Node 侧断言
 * `stepIndex === state.step + 1`。两侧一旦失步（漏步/重放）立刻失败，
 * 绝不写一个"看起来跑完了"的产物。
 *
 * **GPU 小时的记账是诚实的**：只有 `device === 'cuda'` 时才累计墙钟为 GPU 小时；
 * CPU 路径上 `gpuHours` 恒为 0（本机 FX/QAT CPU 路径不占 GPU，就不该记 GPU 账）。
 */
import { ToolError, ERROR_CODES } from './version.mjs'

/**
 * 构造 training-core 的单体插件。
 * @param {{worker: object, config: object, log?: Function}} p
 */
export function makeQatPlugin({ worker, config, log = () => {} }) {
  const gpuClock = { acc: 0 }

  return {
    id: 'qat-int8',
    algorithm: 'qat',
    monolithic: true,

    /** 产物指纹进 checkpoint payload ⇒ 往返可校验（checkpoint-contract 三项判据）。 */
    artifactFingerprint: null, // 由 train.mjs 在拿到目录后注入

    async run(ctx) {
      const init = await worker.call('init', { config }, { timeoutMs: 30 * 60 * 1000 })
      log(`QAT 就绪：${init.model} params=${(init.params / 1e6).toFixed(1)}M rows=${init.rows} device=${init.device} engine=${init.qat?.engine}`)

      const maxSteps = Number(ctx.limits.maxSteps ?? 0)
      let steps = 0

      while (steps < maxSteps) {
        if (ctx.shouldStop()) {
          log(`预算闸门已触发（${ctx.stopReason()}），停止训练循环`)
          break
        }
        const t0 = Date.now()
        // eslint-disable-next-line no-await-in-loop
        const r = await worker.call('step', {}, { timeoutMs: 60 * 60 * 1000 })
        const ms = Date.now() - t0
        const expected = ctx.state.step + 1
        if (r.stepIndex !== expected) {
          throw new ToolError(ERROR_CODES.E_INTERNAL, {
            hint: `两侧失步：容器 stepIndex=${r.stepIndex}，期望 ${expected}（Node state.step=${ctx.state.step}）。已停止，避免写出"看起来跑完了"的产物。`,
            details: { containerStep: r.stepIndex, nodeStep: ctx.state.step },
            recoverable: false,
          })
        }
        // CPU 路径不占 GPU ⇒ 不记 GPU 账；CUDA 路径单卡并发 1 ⇒ 墙钟 ≈ GPU 时间
        if (init.device === 'cuda') gpuClock.acc += ms
        const gpuHours = gpuClock.acc / 3_600_000
        const gate = await ctx.onStep({
          loss: r.loss,
          epoch: r.epoch,
          gpuHours,
          metrics: {
            gradNorm: r.gradNorm ?? null, lr: r.lr ?? null, tokens: r.tokens ?? null,
            ms, gpuMemGiB: r.gpuMemGiB ?? null, engine: init.qat?.engine ?? null,
          },
        })
        steps += 1
        if (steps % 10 === 0 || steps === 1) log(`  QAT step ${steps}/${maxSteps} loss=${Number(r.loss).toFixed(4)} gpuHours=${gpuHours.toFixed(4)}`)
        if (gate.stop) {
          log(`限额命中（${gate.reason}），停止训练循环`)
          break
        }
      }

      return {
        stopReason: ctx.shouldStop() ? ctx.stopReason() : 'plugin-done',
        gpuHours: init.device === 'cuda' ? gpuClock.acc / 3_600_000 : 0,
        init,
        steps,
      }
    },
  }
}
