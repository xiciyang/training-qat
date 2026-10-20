#!/usr/bin/env python3
"""
QAT 执行体（在容器内运行）。—— 这是 M1 缺失的那个执行体。

协议与 `training-sft/python/trainer.py` 同款：stdin/stdout 行分隔 JSON，
**stdout 只允许协议行**，人类日志一律走 stderr。请求-应答按 `id` 严格配对。

它做的事（**两段式 QAT**，与 torchao / llmcompressor 的 QAT 语义一致）：

  ① prepare : 用 **torchao `QATConfig(..., step="prepare")`** 在模型上插入
              STE 假量化（`FakeQuantizedLinear`），量化方案取 **int8 权重（per-axis）**
              + int8 动态激活 —— 与最终的 llmcompressor `INT8` 档位**同方案**。
              `step` 命令做真实的 fwd+bwd+AdamW 优化器步（QAT **训练**）。
  ② convert : QAT 的"把权重真的量化"这一步。torchao 的 `step="convert"` 在本机产出的
              `IntxUnpackedToInt8Tensor` **无法 `save_pretrained`**（safetensors 取 storage
              指针直接 RuntimeError，见 out/qat-probe/step3d-artifact.txt），因此**导出**
              这一步交给仓库自己的 INT8 真量化实现（`llmcompressor` 的
              `QuantizationModifier(scheme="INT8")`）—— 与 `inference-quantization` 的
              `int8-w8a8` 档位**同一套实现、同一 scheme**，产出 compressed-tensors 目录，
              可被 inference-serving(vLLM) 直接加载。

**诚实性纪律（M3）**：本脚本从不"声明"量化方案。它把产出 `config.json` 里的
`quantization_config` **读回来**当作实际 scheme 上报，并断言 `num_bits` 与请求一致；
不一致即失败（`E_SCHEME_DRIFT`）。声明 W4A16 而实际出 INT8 这种事在这里不发生。

源模型目录**只读**：从不写回源目录；输出一律落在独立目录。
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback


# ---------------------------------------------------------------- 协议

def _emit(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _log(msg):
    sys.stderr.write(f"[qat] {msg}\n")
    sys.stderr.flush()


def _err(mid, cmd, e, stage):
    _emit({"id": mid, "ok": False, "cmd": cmd, "stage": stage,
           "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})


# ---------------------------------------------------------------- 产物 dtype 策略（缺陷 16）

# **为什么是 bfloat16**：基座模型与 PTQ 产物的**未量化**张量都是 BF16（本机实测：
# 基座 290/290 BF16、PTQ 290 BF16 + 168 F8_E4M3），而验收判据是 **candidate-vs-base**
# 的配对比较 ⇒ 被测产物的存储 dtype 必须等于**基线**的，否则测得的 Δ 里混进一个与训练/
# 量化无关的 dtype 效应。真机实测：仅把 QAT 产物的未量化张量 F32→BF16，serving 测得的
# `dropPct` 就从 **1.7205% 变成 1.3212%（0.40pp）**——见 `DUAL_PATH_CALIBRATION_REPORT.md` §4。
#
# 与 `@ai-infra/training-core` 的 `DTYPE_POLICY` 是**同一策略**（那边是判定侧的真源；
# 这里是容器侧的执行点）。本脚本**不静默转换**：转换会写进返回值（convertedFrom）与
# 产物自身的 `config.json`（`torch_dtype`），再**读回核对**；对不上就抛错，不交付。
PRODUCT_DTYPE = os.environ.get("QAT_PRODUCT_DTYPE", "bfloat16")
DTYPE_POLICY_WHY = "基座与 PTQ 产物的未量化张量都是 BF16；验收是 candidate-vs-base 配对比较"
_FLOAT_ST_DTYPES = ("F64", "F32", "F16", "BF16")
_TORCH_FLOAT_DTYPES = ("float64", "float32", "float16", "bfloat16")
# safetensors 头里的记号 → **归一化 dtype 名**（与 `@ai-infra/training-core` 的
# `normalizeDtype` / `readTensorDtypes` 用同一套名字）：两侧报出来的 dtype 必须能直接对账，
# 否则"JS 说 bfloat16、Python 说 BF16"会把一次一致判成不一致。
_ST_TO_CANON = {"F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
                "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
                "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8", "U8": "uint8", "BOOL": "bool"}


def read_product_dtypes(dirpath):
    """读产物里每个 `*.safetensors` 的**文件头**，按 dtype 计数（不加载任何权重数据）。

    safetensors 的开头是「8 字节小端 header 长度 + JSON header」，dtype 就写在里面 ——
    解析它零依赖、恒定开销，且判据落在**产物本身**上，而不是流程报告上。
    """
    counts, files = {}, []
    if not os.path.isdir(dirpath):
        return {"files": [], "counts": {}, "floatCounts": {}, "quantizedCounts": {},
                "floatDtype": None, "floatTotal": 0}
    for name in sorted(os.listdir(dirpath)):
        if not name.endswith(".safetensors"):
            continue
        with open(os.path.join(dirpath, name), "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            hdr = json.loads(f.read(n).decode("utf-8"))
        hdr.pop("__metadata__", None)
        per = {}
        for _k, v in hdr.items():
            d = v.get("dtype")
            per[d] = per.get(d, 0) + 1
        files.append({"name": name, "counts": per})
        for d, c in per.items():
            counts[d] = counts.get(d, 0) + c
    float_counts = {_ST_TO_CANON.get(d, d): c for d, c in counts.items() if d in _FLOAT_ST_DTYPES}
    quant_counts = {_ST_TO_CANON.get(d, d): c for d, c in counts.items() if d not in _FLOAT_ST_DTYPES}
    return {"files": files, "counts": counts, "floatCounts": float_counts, "quantizedCounts": quant_counts,
            "floatDtype": (list(float_counts)[0] if len(float_counts) == 1 else None),
            "floatTotal": sum(float_counts.values())}


def enforce_product_dtype(dirpath):
    """把产物里**未量化**的浮点张量统一到 `PRODUCT_DTYPE`，写回，**读回核对**，不符即抛错。

    纪律（与训练侧的 `assertProductDtype` 同一族）：只断言过程抓不到问题，必须断言**权重**。
    `torch` **惰性导入**：产物已经是策略 dtype 时（转换后的一切正常路径）不需要它。
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    if PRODUCT_DTYPE not in _TORCH_FLOAT_DTYPES:
        raise RuntimeError(f"E_PRODUCT_DTYPE_UNKNOWN: 不认识的策略 dtype {PRODUCT_DTYPE!r}")
    if not os.path.isdir(dirpath):
        raise RuntimeError(f"E_PRODUCT_DTYPE_VIOLATION: 产物目录不存在 {dirpath} —— 读不到就不能声称合规（未知 ≠ 合规）")
    before = read_product_dtypes(dirpath)
    converted_from = {}
    torch = None
    for name in sorted(os.listdir(dirpath)):
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(dirpath, name)
        with safe_open(path, framework="pt", device="cpu") as f:
            state = {k: f.get_tensor(k) for k in f.keys()}
            meta = f.metadata()
        changed = False
        for key, tensor in list(state.items()):
            name_of = str(tensor.dtype).replace("torch.", "")
            if name_of in _TORCH_FLOAT_DTYPES and name_of != PRODUCT_DTYPE:
                if torch is None:
                    try:
                        import torch as _torch
                    except Exception as e:  # noqa: BLE001
                        raise RuntimeError(
                            f"E_PRODUCT_DTYPE_VIOLATION: 产物 {dirpath} 的未量化张量是 {name_of}，"
                            f"需要转成 {PRODUCT_DTYPE}，但本容器没有 torch（{type(e).__name__}: {e}）—— 拒绝交付") from e
                    torch = _torch
                converted_from[name_of] = converted_from.get(name_of, 0) + 1
                state[key] = tensor.to(getattr(torch, PRODUCT_DTYPE))
                changed = True
        if changed:
            save_file(state, path, metadata=meta or {})
    # 让产物**自己声明**存储 dtype（此前 QAT 产物的 config.json 里根本没有 torch_dtype）
    cfg_path = os.path.join(dirpath, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["torch_dtype"] = PRODUCT_DTYPE
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
            f.write("\n")
    after = read_product_dtypes(dirpath)
    if after["floatTotal"] == 0 or after["floatDtype"] != PRODUCT_DTYPE:
        raise RuntimeError(
            f"E_PRODUCT_DTYPE_VIOLATION: 产物 {dirpath} 的未量化张量 dtype={after['floatCounts']}"
            f"（策略 {PRODUCT_DTYPE}；读不到的浮点张量数={after['floatTotal']}）—— 拒绝交付")
    return {"policy": PRODUCT_DTYPE, "why": DTYPE_POLICY_WHY,
            "floatDtype": after["floatDtype"], "floatCounts": after["floatCounts"],
            "quantizedCounts": after["quantizedCounts"], "floatTotal": after["floatTotal"],
            "beforeFloatCounts": before["floatCounts"],
            "convertedFrom": (converted_from or None)}


# ---------------------------------------------------------------- license 携带

LICENSE_FILES = ("LICENSE", "LICENSE.txt", "LICENSE.md", "license", "NOTICE")
LICENSE_TEXT_HINTS = (("apache", "apache-2.0"), ("mit license", "mit"),
                      ("bsd 3-clause", "bsd-3-clause"), ("bsd 2-clause", "bsd-2-clause"))


def resolve_license(src_dir):
    """
    从**基座模型目录**解析 license（与 inference-serving 的读取口径一致）：
    config.json.license → README frontmatter → LICENSE 文本启发式。
    解析不出来就如实返回 resolved=False —— **绝不自己发明一个 license**。
    """
    import re
    cfg_path = os.path.join(src_dir, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                v = json.load(f).get("license")
            if isinstance(v, str) and v.strip():
                return {"resolved": True, "spdx": v.strip(), "source": "config.json"}
        except Exception:
            pass
    readme = os.path.join(src_dir, "README.md")
    if os.path.exists(readme):
        try:
            with open(readme, "r", encoding="utf-8", errors="replace") as f:
                txt = f.read(4000)
            m = re.match(r"^---\n(.*?)\n---", txt, re.S)
            if m:
                for key in ("license", "license_name"):
                    lm = re.search(rf"^\s*{key}\s*:\s*(.+)$", m.group(1), re.M | re.I)
                    if lm:
                        return {"resolved": True, "spdx": lm.group(1).strip().strip("'\""), "source": "README frontmatter"}
        except Exception:
            pass
    for name in LICENSE_FILES:
        p = os.path.join(src_dir, name)
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                head = f.read(2000).lower()
        except Exception:
            continue
        for hint, spdx in LICENSE_TEXT_HINTS:
            if hint in head:
                return {"resolved": True, "spdx": spdx, "source": name}
        return {"resolved": True, "spdx": "unknown-text", "source": name}
    return {"resolved": False, "spdx": None, "source": None}


# 基座模型里"模型本体之外"但**服务端必需**的文件。
# 本机实测：llmcompressor 的 save_compressed 会写出一个 **1.6 KB 的 tokenizer.json 桩**
# （真实的是 7 MB），于是 `Qwen2Tokenizer` 把任何 prompt 都 tokenize 成 `[]`，
# vLLM 直接报 `The decoder prompt cannot be empty`。这类"派生物缺附件"的缺陷必须在这里堵住。
ANCILLARY_FILES = (
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "chat_template.jinja", "chat_template.json", "special_tokens_map.json",
    "added_tokens.json", "generation_config.json", "preprocessor_config.json",
)


def carry_ancillary(src_dir, out_dir):
    """
    把基座模型的 tokenizer / chat template 等附件**按原文**带进产出目录
    （覆盖 llmcompressor 写出的桩文件 —— 覆盖是刻意的：桩是**错的**，不是"另一种表示"）。
    """
    import shutil
    carried, overwritten = [], []
    for name in ANCILLARY_FILES:
        p = os.path.join(src_dir, name)
        if not (os.path.exists(p) and os.path.isfile(p)):
            continue
        dst = os.path.join(out_dir, name)
        existed = os.path.exists(dst)
        try:
            if existed:
                with open(p, "rb") as f1, open(dst, "rb") as f2:
                    if f1.read() == f2.read():
                        carried.append(name)
                        continue
            shutil.copyfile(p, dst)
            (overwritten if existed else carried).append(name)
        except Exception:
            pass
    return {"carried": carried, "overwrittenStubs": overwritten}


def verify_tokenizer(out_dir):
    """
    **可执行的** tokenizer 自检：真实模型上 tokenize 一个非空 prompt 必须得到**非空** id 序列。
    本机实测踩到过一次（桩 tokenizer.json → 全部 tokenize 成空），故留成永久判据。
    """
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(out_dir, local_files_only=True)
        ids = tk("The quick brown fox jumps over the lazy dog.")["input_ids"]
        return {"ok": len(ids) > 1, "nTokens": len(ids), "tokenizerClass": type(tk).__name__,
                "reason": None if len(ids) > 1 else "tokenize 结果为空/过短 —— tokenizer 附件不完整"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "nTokens": 0, "tokenizerClass": None, "reason": f"{type(e).__name__}: {e}"}


def carry_license(src_dir, out_dir, extra_note=""):
    """
    派生模型**必须**带上自己的 license（否则下游 inference-serving 的 license 闸门会拒绝部署，
    本机实测：`E_LICENSE_UNRESOLVED: 模型 license 需要确认：(未能识别)`）。
    本函数把基座的 license **原样携带**过去，并写一份带 frontmatter 的 README 记录出处。
    解析不出来时不发明值，只记 resolved=False。
    """
    import shutil
    lic = resolve_license(src_dir)
    copied = []
    for name in LICENSE_FILES:
        p = os.path.join(src_dir, name)
        if os.path.exists(p) and os.path.isfile(p):
            try:
                shutil.copyfile(p, os.path.join(out_dir, name))
                copied.append(name)
            except Exception:
                pass
    lines = [
        "---",
        f"license: {lic['spdx'] if lic['resolved'] else 'unknown'}",
        "library_name: transformers",
        "tags: [qat, int8, compressed-tensors]",
        "---",
        "",
        "# QAT 产出（quantization-aware training）",
        "",
        f"- 基座模型：`{src_dir}`",
        f"- license：`{lic['spdx']}`（来源：{lic['source']}；**继承自基座，不由本插件发明**）",
        "- 量化方案：INT8 W8A8（compressed-tensors；权重 per-channel 对称，激活 per-token 动态）",
        f"- 生成方式：`training-qat` 的容器内执行体（torchao QATConfig STE 假量化训练 → llmcompressor INT8 真量化）{extra_note}",
        "",
    ]
    try:
        with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception:
        pass
    return {**lic, "copiedFiles": copied}


# ---------------------------------------------------------------- 运行时状态

class Runtime:
    def __init__(self):
        self.model = None
        self.tok = None
        self.opt = None
        self.sched = None
        self.cfg = None
        self.rows = []
        self.order = []
        self.cursor = 0
        self.step_index = 0
        self.epoch = 0
        self.device = "cpu"
        self.qat_meta = {}

    # ---- 数据 ----
    def load_rows(self, path):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out

    def next_text(self):
        """确定性取样：固定种子的打乱序列，耗尽后进入下一 epoch 再打乱。"""
        if self.cursor >= len(self.order):
            self.epoch += 1
            rng = random.Random(int(self.cfg.get("seed", 20260929)) + self.epoch)
            self.order = list(range(len(self.rows)))
            rng.shuffle(self.order)
            self.cursor = 0
        idx = self.order[self.cursor]
        self.cursor += 1
        return self.rows[idx]["t"]


RT = Runtime()


# ---------------------------------------------------------------- QAT 引擎

def build_qat_model(cfg):
    """
    返回 (model, meta)。

    引擎优先级（**基于本机实测证据，不是假设**）：
      1. torchao `QATConfig(step="prepare")` —— 本机在真实 Qwen2.5-0.5B 上实测可用
         （FakeQuantizedLinear=169，2 步真实训练，见 evidence/probe-qat-engine.json）。
      2. torch `prepare_qat_fx` —— **实测在真实 HF CausalLM 上失败**
         （`ValueError: code: co_varnames is too small`，见 evidence/probe-fxqat.json）。
      3. torch eager `prepare_qat` —— 实测可用（qat.Linear=169 / FakeQuantize=338）。
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = cfg["baseModel"]
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.float32, attn_implementation="eager")
    model.config.use_cache = False
    model.train()

    engine = cfg.get("engine", "torchao")
    meta = {"engine": engine, "requestedWeightDtype": cfg.get("weightDtype", "int8")}

    errors = []
    if engine in ("torchao", "auto"):
        try:
            import torchao
            from torchao.quantization import quantize_
            from torchao.quantization.qat import QATConfig
            scheme = cfg.get("scheme", "int8-w8a8")
            if scheme == "fp8-dynamic":
                # FP8 假量化：torchao 的 float8 配置在 CPU 上会 assert（requires compute capability ≥8.9），
                # 必须跑在 CUDA 上。选它是因为**只有 FP8 能被本机 vLLM 服务**
                # （实测：INT8 W8A8 的 cutlass 内核在 SM121 上直接 RuntimeError "Int8 not supported on SM121"）。
                if not torch.cuda.is_available():
                    raise RuntimeError("scheme=fp8-dynamic 需要 CUDA（torchao float8 假量化在 CPU 上不可用）")
                # **粒度必须显式对齐部署侧**（D 裁决 / QAT_GRANULARITY_ALIGNMENT.md）：
                # 本包产出的 compressed-tensors 是 `weightStrategy=channel`（**按输出通道**）；
                # 而 `Float8DynamicActivationFloat8WeightConfig()` 不指定 granularity 时默认 **PerTensor**，
                # 实测两者不同 ⇒ QAT 会去适应一个**永远不会被部署**的量化器。
                # torchao 的 `PerRow()` 对 nn.Linear 权重 [out, in] 即"按输出通道"，与 channel 一致。
                base = torchao.quantization.Float8DynamicActivationFloat8WeightConfig(
                    granularity=torchao.quantization.PerRow(),
                )
            else:
                wa = cfg.get("weightActivation", "int8-dynamic")
                if wa == "int8-dynamic":
                    base = torchao.quantization.Int8DynamicActivationIntxWeightConfig(
                        weight_dtype=torch.int8,
                        weight_granularity=torchao.quantization.PerAxis(cfg.get("weightAxis", 0)),
                        intx_packing_format="unpacked_to_int8",
                    )
                else:
                    base = torchao.quantization.IntxWeightOnlyConfig(
                        weight_dtype=torch.int8,
                        granularity=torchao.quantization.PerAxis(cfg.get("weightAxis", 0)),
                        intx_packing_format="unpacked_to_int8",
                    )
            t0 = time.time()
            quantize_(model, QATConfig(base, step="prepare"))
            n_fq = sum(1 for m in model.modules() if type(m).__name__ == "FakeQuantizedLinear")
            if n_fq == 0:
                raise RuntimeError("torchao QATConfig 未插入任何 FakeQuantizedLinear")
            # **假量化器的实际配置**必须现场取证（不是复述我们请求了什么）——
            # 这是"训练侧以为在做 X、实际在做 Y"这类漂移的最早观测点。
            probe = None
            for m in model.modules():
                if type(m).__name__ != "FakeQuantizedLinear":
                    continue
                wfq = getattr(m, "weight_fake_quantizer", None)
                afq = getattr(m, "activation_fake_quantizer", None)
                probe = {
                    "weightFakeQuantizerClass": type(wfq).__name__ if wfq is not None else None,
                    "weightFakeQuantizeConfig": repr(getattr(wfq, "config", None)),
                    "weightFakeQuantDtype": str(getattr(getattr(wfq, "config", None), "dtype", None)),
                    "weightGranularity": repr(getattr(getattr(wfq, "config", None), "granularity", None)),
                    "activationFakeQuantizerClass": type(afq).__name__ if afq is not None else None,
                    "activationFakeQuantizeConfig": repr(getattr(afq, "config", None)),
                    "weightTensorDtypeAfterPrepare": str(type(m.weight).__name__) + "/" + str(m.weight.dtype),
                }
                break
            # **fail-closed：粒度对齐断言**（D 裁决）。
            # 训练的假量化器必须模拟**将要部署**的那个量化器；不一致时**立即失败**，
            # 而不是训完之后才发现"这轮 QAT 适应的是一个不存在的目标"。
            if probe is not None:
                _wg = str(probe.get("weightGranularity"))
                if scheme == "fp8-dynamic" and "PerRow" not in _wg:
                    raise RuntimeError(
                        "粒度未对齐：本包产出并部署的是 compressed-tensors `weightStrategy=channel`（按输出通道），"
                        f"但假量化器实测 granularity={_wg}。QAT 必须模拟将要部署的量化器 —— "
                        "否则训练在适应一个不会被部署的目标（见 QAT_GRANULARITY_ALIGNMENT.md）。"
                    )
                if scheme != "fp8-dynamic" and "PerAxis" not in _wg:
                    raise RuntimeError(
                        f"粒度未对齐：INT8 分支请求的是 PerAxis(0)（按输出通道），但假量化器实测 granularity={_wg}。"
                    )
            meta.update({"torchao": getattr(torchao, "__version__", None), "scheme": scheme, "weightActivation": scheme,
                         "baseConfig": repr(base), "fakeQuantizedLinear": n_fq,
                         "prepareSec": round(time.time() - t0, 2),
                         "fakeQuantMechanism": "torchao QATConfig STE (IntxFakeQuantizer)",
                         "fakeQuantProbe": probe})
            _log(f"QAT 引擎 = torchao QATConfig；FakeQuantizedLinear={n_fq}；prepare {meta['prepareSec']}s")
            return model, tok, meta
        except Exception as e:  # noqa: BLE001
            errors.append(f"torchao: {type(e).__name__}: {e}")
            _log(f"torchao QAT 不可用，回退 eager torch QAT：{e}")

    # ---- fallback: torch eager prepare_qat ----
    import torch.ao.quantization as tq
    import torch.ao.nn.qat as nnqat
    qc = tq.get_default_qat_qconfig(cfg.get("torchEngine", "qnnpack"))
    model.qconfig = qc
    for _n, mod in model.named_modules():
        if isinstance(mod, torch.nn.Embedding):
            mod.qconfig = None          # B3 同类约束：Embedding 不量化
        elif isinstance(mod, torch.nn.Linear):
            mod.qconfig = qc
    if hasattr(model, "lm_head") and model.lm_head is not None:
        model.lm_head.qconfig = None    # 与 llmcompressor 的 ignore=["lm_head"] 对齐
    tq.prepare_qat(model, inplace=True)
    n_qat = sum(1 for m in model.modules() if isinstance(m, nnqat.Linear))
    if n_qat == 0:
        raise RuntimeError(f"所有 QAT 引擎均不可用：{errors}")
    meta.update({"engine": "torch-eager-qat", "torchEngine": cfg.get("torchEngine", "qnnpack"),
                 "qatLinear": n_qat, "priorErrors": errors,
                 "fakeQuantMechanism": "torch.ao.quantization.QATLinear (FusedMovingAvgObsFakeQuantize)"})
    _log(f"QAT 引擎 = torch eager prepare_qat；qat.Linear={n_qat}")
    return model, tok, meta


# ---------------------------------------------------------------- 命令实现

def cmd_init(req):
    cfg = req["config"]
    RT.cfg = cfg
    import torch
    torch.manual_seed(int(cfg.get("seed", 20260929)))
    if cfg.get("threads"):
        torch.set_num_threads(int(cfg["threads"]))

    dev = cfg.get("device", "cpu")
    if dev == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 device=cuda 但 CUDA 不可用")
    RT.device = dev

    model, tok, meta = build_qat_model(cfg)
    if dev == "cuda":
        model = model.to("cuda")
    RT.model, RT.tok, RT.qat_meta = model, tok, meta

    RT.rows = RT.load_rows(cfg["trainFile"])
    if len(RT.rows) == 0:
        raise RuntimeError(f"训练语料为空：{cfg['trainFile']}")
    if len(RT.rows) < 4:
        raise RuntimeError(f"训练语料过少（{len(RT.rows)} 行 < 4），拒绝训练")
    rng = random.Random(int(cfg.get("seed", 20260929)))
    RT.order = list(range(len(RT.rows)))
    rng.shuffle(RT.order)

    trainable = [p for p in model.parameters() if p.requires_grad]
    RT.opt = torch.optim.AdamW(trainable, lr=float(cfg.get("lr", 1e-5)),
                               weight_decay=float(cfg.get("weightDecay", 0.0)))
    RT.init_fingerprint = param_fingerprint()   # 起点指纹（供"相对起点变了没有"的断言）
    return {
        "ok": True, "model": type(model).__name__, "device": dev,
        "params": sum(p.numel() for p in model.parameters()),
        "trainableParams": sum(p.numel() for p in trainable),
        "linear": sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear)),
        "rows": len(RT.rows),
        "qat": meta,
        "torch": torch.__version__,
        "cudaAvailable": torch.cuda.is_available(),
        "threads": torch.get_num_threads(),
    }


def cmd_batch(req):
    """取下一批 token（确定性）。Node 侧只发一个 id，数据体留在容器内。"""
    import torch
    cfg = RT.cfg
    bs = int(req.get("batchSize", cfg.get("batchSize", 1)))
    maxlen = int(cfg.get("maxSeqLen", 256))
    texts = [RT.next_text() for _ in range(bs)]
    enc = RT.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=maxlen)
    ids = enc["input_ids"]
    if RT.device == "cuda":
        ids = ids.to("cuda")
    return {"ok": True, "shape": list(ids.shape), "tokens": int(ids.numel())}


def cmd_step(req):
    """一个**优化器步**（内含 gradAccum 个微批）。"""
    import torch
    cfg = RT.cfg
    accum = max(1, int(cfg.get("gradAccum", 1)))
    maxlen = int(cfg.get("maxSeqLen", 256))
    t0 = time.time()
    total_tokens = 0
    RT.opt.zero_grad(set_to_none=True)
    last_loss = None
    for _ in range(accum):
        texts = [RT.next_text() for _ in range(int(cfg.get("batchSize", 1)))]
        enc = RT.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=maxlen)
        ids = enc["input_ids"]
        if RT.device == "cuda":
            ids = ids.to("cuda")
        total_tokens += int(ids.numel())
        out = RT.model(input_ids=ids, labels=ids)
        loss = out.loss / accum
        loss.backward()
        last_loss = float(out.loss.detach())
    gn = torch.nn.utils.clip_grad_norm_([p for p in RT.model.parameters() if p.requires_grad], 1.0)
    RT.opt.step()
    RT.step_index += 1
    gpu_mem = None
    if RT.device == "cuda":
        gpu_mem = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
    return {"ok": True, "stepIndex": RT.step_index, "loss": last_loss,
            "gradNorm": float(gn), "lr": RT.opt.param_groups[0]["lr"],
            "tokens": total_tokens, "ms": int((time.time() - t0) * 1000),
            "epoch": RT.epoch, "gpuMemGiB": gpu_mem}


def cmd_save_float(req):
    """把 **QAT 训练后的浮点权重**导出为标准 HF 目录（供 convert 与对照使用）。

    ⚠️ **缺陷 16**：QAT 训练在 **F32** 计算精度下进行（CPU/GPU 上的假量化训练需要它），
    但**存储**必须按策略落在 `PRODUCT_DTYPE`（BF16）—— 此前这里直接
    `save_pretrained()`，于是产物把未量化张量存成 F32，而基座/PTQ 存 BF16，
    仅这一项差异就使 serving 测得的 dropPct 变化 0.40pp（见 PRODUCT_DTYPE 处注释）。
    计算精度与存储精度是两件事：**只有存储精度会跟着产物走进测量**。
    """
    out = req["dir"]
    os.makedirs(out, exist_ok=True)
    RT.model.config.use_cache = True
    RT.model.save_pretrained(out, safe_serialization=True)
    # 导出后**把存储 dtype 钉到策略值**，并读回核对（对不上即抛错，不交付）
    dtype_report = enforce_product_dtype(out)
    _log(f"[dtype] save_float 产物 dtype={dtype_report['floatDtype']} "
         f"convertedFrom={dtype_report['convertedFrom']} policy={dtype_report['policy']}")
    files = sorted(os.listdir(out))
    weights = [f for f in files if f.endswith(".safetensors") or f.endswith(".bin")]
    h = hashlib.sha256()
    total = 0
    for w in sorted(weights):
        with open(os.path.join(out, w), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
                total += len(chunk)
    license_info = carry_license(RT.cfg["baseModel"], out)
    files = sorted(os.listdir(out))
    weights = [f for f in files if f.endswith(".safetensors") or f.endswith(".bin")]
    return {"ok": True, "dir": out, "files": files, "weightFiles": weights,
            "bytes": total, "weightsSha256": h.hexdigest(), "step": RT.step_index,
            "license": license_info,
            "productDtype": dtype_report,
            "qat": RT.qat_meta}


def cmd_convert(req):
    """
    QAT 的 convert 步：对 **QAT 训练后的权重** 施加真实的 INT8 量化。

    刻意用仓库自己的 `llmcompressor` INT8 实现（与 `inference-quantization`
    的 `int8-w8a8` 档位同实现、同 scheme），因为 torchao 的 convert 产出
    **无法 save_pretrained**（见文件头）。产出是 compressed-tensors 目录。
    """
    src = req["src"]
    out = req["dir"]
    samples = int(req.get("samples", 64))
    maxlen = int(req.get("maxSeqLen", 256))
    calib_file = req.get("calibFile") or RT.cfg["trainFile"]
    os.makedirs(out, exist_ok=True)

    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    import llmcompressor
    import compressed_tensors

    rows = []
    with open(calib_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
            if len(rows) >= samples:
                break
    if len(rows) < 16:
        raise RuntimeError(f"校准样本 {len(rows)} 条 < 16，拒绝量化（与 inference-quantization 同一条门禁）")
    ds = Dataset.from_list([{"text": r["t"]} for r in rows])

    scheme = req.get("scheme", "FP8_DYNAMIC")
    ignore = list(req.get("ignore", ["lm_head"]))
    t0 = time.time()
    oneshot(model=src, dataset=ds,
            recipe=[QuantizationModifier(targets="Linear", scheme=scheme, ignore=ignore)],
            max_seq_length=maxlen, num_calibration_samples=len(rows),
            output_dir=out, save_compressed=True)
    seconds = round(time.time() - t0, 1)

    # ⚠️ **缺陷 16**：convert 的产出把**未量化**张量（embedding / lm_head / norm / 量化 scale）
    # 按 **源产物（src）的存储 dtype** 落下 —— src 是 F32 时这里就是 F32，而基座与 PTQ 都是 BF16。
    # 于是"同一份权重、只因存储精度不同"就测出不同的 Δ（实测 0.40pp）。这里显式钉到策略 dtype
    # 并**读回核对**；核对不过 ⇒ E_PRODUCT_DTYPE_VIOLATION（拒绝交付）。
    dtype_report = enforce_product_dtype(out)
    _log(f"[dtype] convert 产物 dtype={dtype_report['floatDtype']} "
         f"convertedFrom={dtype_report['convertedFrom']} quantized={dtype_report['quantizedCounts']} "
         f"policy={dtype_report['policy']}")

    cfg_path = os.path.join(out, "config.json")
    if not os.path.exists(cfg_path):
        raise RuntimeError("convert 产出缺 config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        out_cfg = json.load(f)
    qc = out_cfg.get("quantization_config")
    if not qc:
        raise RuntimeError("convert 产出的 config.json 没有 quantization_config —— 说明没真的量化")
    # **不声明，只读回**：实际 scheme 取自产出自身
    g0 = (qc.get("config_groups") or {}).get("group_0") or {}
    w = g0.get("weights") or {}
    actual = {"quantMethod": qc.get("quant_method"), "format": qc.get("format"),
              "numBits": w.get("num_bits"), "strategy": w.get("strategy"),
              "symmetric": w.get("symmetric"), "groupSize": w.get("group_size"),
              "type": w.get("type"),
              "inputActivations": None if not g0.get("input_activations") else {
                  "numBits": g0["input_activations"].get("num_bits"),
                  "strategy": g0["input_activations"].get("strategy"),
                  "dynamic": g0["input_activations"].get("dynamic")},
              "ignore": qc.get("ignore")}

    want_bits = int(req.get("expectBits", 8))
    if actual["numBits"] != want_bits:
        raise RuntimeError(f"E_SCHEME_DRIFT: 请求 {want_bits}bit，产出实际 {actual['numBits']}bit —— 拒绝冒名上报")
    want_type = req.get("expectType")
    if want_type and actual.get("type") != want_type:
        raise RuntimeError(f"E_SCHEME_DRIFT: 请求 weight type={want_type}，产出实际 {actual.get('type')} —— 拒绝冒名上报")

    # 派生模型必须带 license —— 否则下游 serving 的 license 闸门会拒绝部署（本机实测踩到）
    license_info = carry_license(RT.cfg["baseModel"], out)
    # 附件（tokenizer / chat template）：llmcompressor 会写桩文件，必须用基座的原文覆盖
    ancillary = carry_ancillary(RT.cfg["baseModel"], out)
    tokenizer_check = verify_tokenizer(out)
    files = sorted(os.listdir(out))
    weights = [f for f in files if f.endswith(".safetensors") or f.endswith(".bin")]
    return {"ok": True, "dir": out, "seconds": seconds, "calibrationSamples": len(rows),
            "license": license_info, "ancillary": ancillary, "tokenizerCheck": tokenizer_check,
            "files": files, "weightFiles": weights,
            "bytes": sum(os.path.getsize(os.path.join(out, w2)) for w2 in weights),
            "quantizationConfig": qc, "scheme": f"{actual['numBits']}bit/{actual['strategy']}",
            "actual": actual,
            # 缺陷 16：产物的存储 dtype 是**测量变量** ⇒ 随产物一起上报（读回的实际值，不是声明值）
            "productDtype": dtype_report,
            "toolchain": {"llmcompressor": llmcompressor.__version__,
                          "compressed_tensors": compressed_tensors.__version__}}


def param_fingerprint() -> str:
    """**权重指纹**：取前若干**可训练**参数的各 1024 个元素做哈希。

    为什么需要它（沿用 pretrain 的原始事故）：计数类断言（步号/墙面/上限）**全部为真**，
    也可能**权重一步没动**（实测：`lr=0` 让 500 步全是空操作）。
    必须有一条**落在权重上**的绝对量断言。

    与 pretrain 版本的**关键差别**：只哈希 `requires_grad=True` 的参数。
    理由（用户裁决 H5 明确要求）：若模型有冻结部分（LoRA 场景），
    哈希"前 8 个具名参数"会哈希到**冻结的基座** ⇒ `weightsChanged` **恒为 False**
    ⇒ 断言变成永远报错的假警报。QAT 目前是全参训练，但这条规则**不依赖当前形态**。

    为什么是**内存指纹**而不是哈希磁盘目录：QAT 的 checkpoint 在训练**过程中**写，
    而此刻 `qat-float/` 还**没有落盘** ⇒ 哈希目录恒为 `null`（这正是本缺口的根因）。
    内存指纹在**任何时刻**都可算。
    """
    import torch  # 与同文件其它 cmd_* 一致：torch 是**函数内惰性导入**（模块顶层不 import torch）
    h = hashlib.sha256()
    with torch.no_grad():
        ps = [(n, p) for n, p in RT.model.named_parameters() if getattr(p, "requires_grad", False)]
        if not ps:
            ps = list(RT.model.named_parameters())
        for name, prm in ps[:8]:
            v = prm.detach().float().flatten()[:1024].cpu().numpy().tobytes()
            h.update(name.encode("utf-8"))
            h.update(v)
    return "sha256:" + h.hexdigest()


def cmd_fingerprint(req):
    """当前权重指纹（内存）——checkpoint payload 在任何步都可以拿到它。"""
    return {"ok": True, "hash": param_fingerprint(), "initHash": getattr(RT, "init_fingerprint", None)}


def cmd_health(req):
    import torch
    out = {"ok": True, "torch": torch.__version__, "cuda": torch.cuda.is_available(), "python": sys.version.split()[0]}
    try:
        import torchao
        out["torchao"] = torchao.__version__
        from torchao.quantization.qat import QATConfig  # noqa: F401
        out["torchaoQat"] = True
    except Exception as e:  # noqa: BLE001
        out["torchao"] = None
        out["torchaoQat"] = False
        out["torchaoError"] = f"{type(e).__name__}: {e}"
    try:
        import llmcompressor
        import compressed_tensors
        out["llmcompressor"] = llmcompressor.__version__
        out["compressedTensors"] = compressed_tensors.__version__
    except Exception as e:  # noqa: BLE001
        out["llmcompressor"] = None
        out["llmcompressorError"] = f"{type(e).__name__}: {e}"
    try:
        import transformers
        out["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001
        out["transformers"] = None
    return out


COMMANDS = {"init": cmd_init, "batch": cmd_batch, "step": cmd_step,
            "save_float": cmd_save_float, "convert": cmd_convert, "health": cmd_health,
            "fingerprint": cmd_fingerprint}


def main():
    _log(f"qat_trainer 启动 pid={os.getpid()} python={sys.version.split()[0]}")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            _log(f"非协议行（忽略）：{line[:120]}")
            continue
        mid = req.get("id")
        cmd = req.get("cmd")
        if cmd in (None, ""):
            continue
        if cmd == "shutdown":
            _emit({"id": mid, "ok": True, "cmd": "shutdown"})
            break
        fn = COMMANDS.get(cmd)
        if fn is None:
            _emit({"id": mid, "ok": False, "cmd": cmd, "stage": "dispatch", "error": f"未知命令 {cmd}"})
            continue
        try:
            payload = fn(req)
            payload["id"] = mid
            payload["cmd"] = cmd
            _emit(payload)
        except Exception as e:  # noqa: BLE001
            _err(mid, cmd, e, "execute")
    _log("qat_trainer 退出")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.parse_args()
    main()
