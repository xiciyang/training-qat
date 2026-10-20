#!/usr/bin/env bash
# 等 GPU 串行化标记出现 → 立刻跑真机 validate（serving 仪器 + 跨插件对账）。
#
# 为什么用 watcher 而不是手工轮询：**不占 GPU 的等待不该占人的注意力**。
# 标记出现前本脚本不碰 GPU；出现后它跑一次 validate 并把原始输出落盘。
# 上限 --max-wait 秒；超时即如实记录"未等到"，不无限挂。
set -uo pipefail
MARKER=/home/xiciyang/deepseek_harness/inference-training/out/lt4-seed-retrospect-raw.json
PKG=/home/xiciyang/deepseek_harness/inference-training/training-qat
RUN_ID="${1:?usage: watch-and-validate.sh <runId> [maxWaitSec]}"
MAX_WAIT="${2:-3600}"
OUT="$PKG/out/realmachine/watch-validate-${RUN_ID}.log"
mkdir -p "$(dirname "$OUT")"

echo "[$(date -Is)] watcher 启动：等 $MARKER（最多 ${MAX_WAIT}s），runId=$RUN_ID" | tee -a "$OUT"
waited=0
while [ ! -f "$MARKER" ]; do
  if [ "$waited" -ge "$MAX_WAIT" ]; then
    echo "[$(date -Is)] 超时：$MARKER 未出现 ⇒ **未做任何 GPU 工作**（串行化纪律）" | tee -a "$OUT"
    exit 3
  fi
  sleep 30
  waited=$((waited + 30))
  if [ $((waited % 300)) -eq 0 ]; then echo "[$(date -Is)] 已等 ${waited}s…" | tee -a "$OUT"; fi
done

echo "[$(date -Is)] 标记出现（waited=${waited}s），开始 validate" | tee -a "$OUT"
cd "$PKG"
node tests/realmachine/qat-validate.mjs --runId "$RUN_ID" --max-drop 2.0 --nTrain 32 --nVal 32 2>&1 | tee -a "$OUT"
code=${PIPESTATUS[0]}
echo "[$(date -Is)] validate 退出码=$code" | tee -a "$OUT"
exit "$code"
