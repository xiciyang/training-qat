# training-qat toolchain
#
# FROM the quantization toolchain (NOT the SFT one) because the QAT "convert" step
# must apply the *repository's own* INT8 real-quantization (llmcompressor +
# compressed-tensors), and those live only in the quantization image.
#
# What this base gives us [verified: out/qat-probe/step2-import.txt, build.log]:
#   torch 2.11.0+cu130, transformers 5.17.0, datasets, llmcompressor 0.14.0,
#   compressed_tensors, accelerate  -- and NO peft, NO torchao.
#
# Added here: torchao 0.18.0 (pure-python wheel, 1.4 MB) -- the QAT fake-quant engine.
# Pinned because the QAT entry points (`QATConfig`, `Int8DynamicActivationIntxWeightConfig`)
# are version-sensitive and the version was probed.
FROM inference-quantization/toolchain:0.1.0

RUN pip install --no-cache-dir torchao==0.18.0
