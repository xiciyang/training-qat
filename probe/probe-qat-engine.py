#!/usr/bin/env python3
"""
Probe 2 (CPU only): decide the QAT engine + the real-quantization (convert) step.

Body of evidence targeted:
  1. torchao QATConfig prepare on the REAL Qwen2.5-0.5B-Instruct for the int8
     candidates that best match what llmcompressor's INT8 scheme actually emits.
  2. does the torchao-QAT-prepared model still `save_pretrained()` (float export)?
  3. does llmcompressor `oneshot(QuantizationModifier(scheme="INT8"))` run **on CPU**
     on those QAT-trained weights, and what `quantization_config` does it emit?
  4. eager `torch.ao.quantization.prepare_qat` fallback: does it insert QAT modules
     (counted with isinstance, NOT type().__name__ -- see probe 1's known trap)?

Writes only to --out. <MODELS_DIR> is mounted read-only.
"""
import argparse
import json
import os
import sys
import time
import traceback


def log(msg):
    print(f"[probe2] {msg}", flush=True)


CANDIDATES = [
    ("Int8DynamicActivationIntxWeight(int8 perAxis)", "torchao.quantization.Int8DynamicActivationIntxWeightConfig",
     {"weight_dtype": "int8", "weight_granularity_peraxis": 0,
      "intx_packing_format": "unpacked_to_int8"}),
    ("IntxWeightOnly(int8 perAxis)", "torchao.quantization.IntxWeightOnlyConfig",
     {"weight_dtype": "int8", "granularity_peraxis": 0, "intx_packing_format": "unpacked_to_int8"}),
    ("Int8WeightOnly", "torchao.quantization.Int8WeightOnlyConfig", {}),
]


def build_base_config(name, kwargs):
    import torch
    import torchao.quantization as tq
    if name.startswith("Int8DynamicActivationIntxWeight"):
        return tq.Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int8,
            weight_granularity=tq.PerAxis(kwargs["weight_granularity_peraxis"]),
            intx_packing_format=kwargs["intx_packing_format"],
        )
    if name.startswith("IntxWeightOnly"):
        return tq.IntxWeightOnlyConfig(
            weight_dtype=torch.int8,
            granularity=tq.PerAxis(kwargs["granularity_peraxis"]),
            intx_packing_format=kwargs["intx_packing_format"],
        )
    if name.startswith("Int8WeightOnly"):
        return tq.Int8WeightOnlyConfig()
    raise ValueError(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="<BASE_MODEL>")
    ap.add_argument("--out", default="/out")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--calib", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    report = {"probe": "torchao-qat + llmcompressor-int8-convert", "candidates": []}

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    report["torch"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    import torchao
    report["torchao"] = torchao.__version__
    log(f"torch {torch.__version__} torchao {torchao.__version__} cuda={torch.cuda.is_available()}")

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    text = ("The quick brown fox jumps over the lazy dog. In a distant country, the population "
            "of the province had grown steadily for a decade, and the roads were crowded. ")
    enc = tok(text * 8, return_tensors="pt", truncation=True, max_length=args.seq_len)
    ids = enc["input_ids"]
    log(f"input_ids {tuple(ids.shape)}")

    saved_dir = None
    for label, cls, kw in CANDIDATES:
        rec = {"label": label, "cls": cls}
        log(f"--- candidate: {label} ---")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, local_files_only=True, dtype=torch.float32, attn_implementation="eager")
            model.config.use_cache = False
            model.train()
            from torchao.quantization import quantize_
            from torchao.quantization.qat import QATConfig
            base = build_base_config(cls.split(".")[-1], kw)
            rec["base_config"] = repr(base)
            import torchao.quantization.qat as _q  # noqa
            t0 = time.time()
            quantize_(model, QATConfig(base, step="prepare"))
            rec["prepare_sec"] = round(time.time() - t0, 1)

            n_fq = sum(1 for m in model.modules() if type(m).__name__ == "FakeQuantizedLinear")
            n_wfq = sum(1 for m in model.modules() if hasattr(m, "weight_fake_quantizer"))
            rec["fake_quantized_linear"] = n_fq
            rec["with_weight_fake_quantizer"] = n_wfq
            log(f"  prepare OK {rec['prepare_sec']}s FakeQuantizedLinear={n_fq} with_wfq={n_wfq}")

            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
            times, losses = [], []
            for s in range(args.steps):
                ts = time.time()
                out = model(input_ids=ids, labels=ids)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                times.append(round(time.time() - ts, 2))
                losses.append(round(float(out.loss), 6))
                log(f"  step {s} loss={losses[-1]} {times[-1]}s")
            rec["step_sec"] = times
            rec["losses"] = losses

            # float export of the QAT-prepared (still float) model
            d = os.path.join(args.out, f"qat-float-{cls.split('.')[-1]}")
            try:
                model.config.use_cache = True
                model.save_pretrained(d, safe_serialization=True)
                files = sorted(os.listdir(d))
                rec["float_export"] = {"ok": True, "dir": d, "files": files,
                                       "bytes": sum(os.path.getsize(os.path.join(d, f)) for f in files if f.endswith(".safetensors"))}
                log(f"  float save_pretrained OK -> {files}")
                if saved_dir is None:
                    saved_dir = d
            except Exception as e:
                rec["float_export"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                log(f"  float save_pretrained FAILED: {e}")
            rec["ok"] = True
        except Exception as e:
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()[-1200:]
            log(f"  FAILED: {e}")
        report["candidates"].append(rec)
        if rec.get("ok"):
            break  # first working candidate is enough for the decision

    # ---- eager torch QAT fallback, counted correctly ----
    eager = {}
    try:
        import copy
        import torch.ao.quantization as tq2
        import torch.ao.nn.qat as nnqat
        em = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.float32,
                                                 attn_implementation="eager")
        em.config.use_cache = False
        em.train()
        qc = tq2.get_default_qat_qconfig("qnnpack")
        em.qconfig = qc
        for name, mod in em.named_modules():
            if isinstance(mod, torch.nn.Embedding):
                mod.qconfig = None
            elif isinstance(mod, torch.nn.Linear):
                mod.qconfig = qc
        tq2.prepare_qat(em, inplace=True)
        eager["n_qat_linear_isinstance"] = sum(1 for m in em.modules() if isinstance(m, nnqat.Linear))
        eager["n_fakequant"] = sum(1 for m in em.modules() if isinstance(m, tq2.FakeQuantize))
        eager["ok"] = eager["n_qat_linear_isinstance"] > 0
        log(f"eager prepare_qat: qat.Linear={eager['n_qat_linear_isinstance']} FakeQuantize={eager['n_fakequant']}")
    except Exception as e:
        eager = {"ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-1200:]}
        log(f"eager prepare_qat FAILED: {e}")
    report["eager_qat"] = eager

    # ---- llmcompressor INT8 oneshot on the QAT-trained float weights (CPU) ----
    conv = {"ran": False}
    if saved_dir:
        conv["source"] = saved_dir
        # calibration jsonl from the same corpus (tiny, deterministic)
        calib_path = os.path.join(args.out, "calib.jsonl")
        with open(calib_path, "w") as f:
            for i in range(args.calib):
                f.write(json.dumps({"t": text * (2 + (i % 3))}) + "\n")
        conv["calib"] = calib_path
        try:
            from datasets import Dataset
            from llmcompressor import oneshot
            from llmcompressor.modifiers.quantization import QuantizationModifier
            rows = [json.loads(l) for l in open(calib_path)]
            ds = Dataset.from_list([{"text": r["t"]} for r in rows])
            out_dir = os.path.join(args.out, "int8-converted")
            t0 = time.time()
            oneshot(model=saved_dir, dataset=ds,
                    recipe=[QuantizationModifier(targets="Linear", scheme="INT8", ignore=["lm_head"])],
                    max_seq_length=args.seq_len, num_calibration_samples=len(rows),
                    output_dir=out_dir, save_compressed=True)
            conv["seconds"] = round(time.time() - t0, 1)
            cfg = json.load(open(os.path.join(out_dir, "config.json")))
            qc = cfg.get("quantization_config")
            conv["quantization_config"] = qc
            conv["scheme_string_verifyQuantOutput_style"] = (
                f"{qc['config_groups']['group_0']['weights']['num_bits']}bit/"
                f"{qc['config_groups']['group_0']['weights']['strategy']}" if qc else None)
            conv["files"] = sorted(os.listdir(out_dir))[:12]
            conv["ok"] = qc is not None
            log(f"llmcompressor INT8 oneshot OK {conv['seconds']}s scheme={conv['scheme_string_verifyQuantOutput_style']}")
            log(f"  files={conv['files']}")
        except Exception as e:
            conv["ok"] = False
            conv["error"] = f"{type(e).__name__}: {e}"
            conv["traceback"] = traceback.format_exc()[-2500:]
            log(f"llmcompressor INT8 oneshot FAILED: {e}")
    report["convert"] = conv

    with open(os.path.join(args.out, "probe-qat-engine.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    log("PROBE2 DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
