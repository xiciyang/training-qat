/**
 * worker —— 容器内 QAT 执行体（`python/qat_trainer.py`）的**持久 worker 客户端**。
 *
 * 与 training-sft 同款协议（行分隔 JSON，stdout 只走协议、stderr 走人类日志），
 * 但跨的镜像不同：`training-qat/toolchain:0.1.0`
 * = `inference-quantization/toolchain:0.1.0` + `torchao==0.18.0`
 * （QAT 的"真量化"步要用仓库自己的 llmcompressor INT8 实现，那只在量化镜像里）。
 *
 * 安全：容器 `--rm`、源模型只读挂载、输出目录可写挂载、以宿主 uid:gid 运行。
 * 非 root uid 必须显式给 USER/LOGNAME（长任务 3 阶段 3 冒烟闸实测的坑：torch 的
 * 缓存目录解析走 `getpass.getuser()`，容器 /etc/passwd 里没有该 uid 会 KeyError）。
 */
import { spawn } from 'node:child_process'
import { createInterface } from 'node:readline'
import { join, dirname } from 'node:path'
import { userInfo } from 'node:os'
import { fileURLToPath } from 'node:url'
import { ToolError, ERROR_CODES } from './version.mjs'

export const QAT_IMAGE = 'training-qat/toolchain:0.1.0'

export class ContainerWorker {
  constructor({ image = QAT_IMAGE, mounts = [], gpus = true, entry = '/work/qat_trainer.py', python = 'python3', env = {}, log = () => {}, user = null }) {
    this.image = image
    this.mounts = mounts
    this.gpus = gpus
    this.entry = entry
    this.python = python
    this.user = user === null ? (typeof process.getuid === 'function' ? `${process.getuid()}:${process.getgid()}` : null) : user
    const username = (() => {
      try { return userInfo().username || 'user' } catch { return process.env.USER || process.env.LOGNAME || 'user' }
    })()
    this.env = {
      HOME: '/tmp', USER: username, LOGNAME: username,
      XDG_CACHE_HOME: '/tmp/.cache', HF_HOME: '/tmp/.cache/huggingface',
      TRITON_CACHE_DIR: '/tmp/.triton', TORCHINDUCTOR_CACHE_DIR: '/tmp/.torchinductor',
      HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1', HF_DATASETS_OFFLINE: '1',
      ...env,
    }
    this.log = log
    this.child = null
    this.seq = 0
    this.pending = new Map()
    this.stderrTail = []
  }

  args() {
    const a = ['run', '-i', '--rm']
    if (this.gpus) a.push('--gpus', 'all')
    if (this.user) a.push('--user', this.user)
    for (const m of this.mounts) a.push('-v', m)
    for (const [k, v] of Object.entries(this.env)) a.push('-e', `${k}=${v}`)
    a.push('--entrypoint', this.python, this.image, this.entry)
    return a
  }

  start() {
    if (this.child) return this
    this.child = spawn('docker', this.args(), { stdio: ['pipe', 'pipe', 'pipe'] })
    createInterface({ input: this.child.stdout }).on('line', (line) => {
      let msg
      try { msg = JSON.parse(line) } catch { this.log(`[worker] 非协议行（忽略）：${line.slice(0, 160)}`); return }
      const p = this.pending.get(msg.id)
      if (!p) { this.log(`[worker] 未知 id 的应答 ${msg.id}（忽略）`); return }
      this.pending.delete(msg.id)
      p.resolve(msg)
    })
    createInterface({ input: this.child.stderr }).on('line', (line) => {
      this.stderrTail.push(line)
      if (this.stderrTail.length > 60) this.stderrTail.shift()
      if (!/^(Loading weights|Writing model shards|Tokenizing|Applying quantization|Compressing model|Dispatching model)/.test(line) && line.trim() !== '') {
        this.log(`[worker:err] ${line.slice(0, 220)}`)
      }
    })
    this.child.on('exit', (code, sig) => {
      for (const [, p] of this.pending) p.reject(new Error(`QAT worker 退出（code=${code} sig=${sig}）；stderr 尾部：${this.stderrTail.slice(-4).join(' | ').slice(0, 300)}`))
      this.pending.clear()
      this.child = null
    })
    return this
  }

  call(cmd, payload = {}, { timeoutMs = 60 * 60 * 1000 } = {}) {
    if (!this.child) throw new Error('worker 未启动')
    const id = ++this.seq
    const req = JSON.stringify({ id, cmd, ...payload })
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id)
        reject(new Error(`worker 命令 ${cmd} 超时（${timeoutMs}ms）`))
      }, timeoutMs)
      this.pending.set(id, {
        resolve: (m) => {
          clearTimeout(timer)
          if (m.ok === false) {
            reject(new ToolError(ERROR_CODES.E_INTERNAL, {
              hint: `QAT worker 内部错误（${cmd}@${m.stage}）：${m.error}`,
              details: { cmd, stage: m.stage, trace: String(m.trace || '').slice(-800) },
            }))
          } else resolve(m)
        },
        reject: (e) => { clearTimeout(timer); reject(e) },
      })
      this.child.stdin.write(`${req}\n`)
    })
  }

  async stop() {
    if (!this.child) return
    try { await this.call('shutdown', {}, { timeoutMs: 20000 }) } catch { /* 已退出 */ }
    try { this.child?.stdin?.end() } catch { /* ignore */ }
    const c = this.child
    if (!c) return
    await new Promise((r) => {
      if (c.exitCode !== null || c.signalCode !== null) { r(); return }
      const t = setTimeout(() => { try { c.kill('SIGKILL') } catch { /* ignore */ } ; r() }, 8000)
      c.on('exit', () => { clearTimeout(t); r() })
    })
    this.child = null
  }
}

/** QAT 容器的标准挂载（模型只读 / 语料只读 / 输出可写 / 执行体只读）。 */
export function qatMounts({ modelDir, datasetDir, outDir, pythonDir = null, extra = [] }) {
  // 模型目录必须挂成 `/models/<basename>`（挂成 /models 会让容器内看到目录内容，
  // transformers 找不到 config.json 就当成 HF repo id —— 实测踩过的坑）
  const base = String(modelDir).split('/').filter(Boolean).pop()
  const py = pythonDir || join(dirname(fileURLToPath(import.meta.url)), '..', 'python')
  return [`${modelDir}:/models/${base}:ro`, `${datasetDir}:/data:ro`, `${outDir}:/out`, `${py}:/work:ro`, ...extra]
}
