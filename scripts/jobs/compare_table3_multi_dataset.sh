#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?请先 export DATA_ROOT=/你的/数据根目录}"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NGPU="${NGPU:-$NPROC_PER_NODE}"
SEED="${SEED:-2021}"
DATASETS=(ecl etth1 traffic weather solar)

REPRO_SCRIPT="scripts/jobs/table3_96pred96_paper_multigpu.sh"
EXP_SCRIPT="scripts/jobs/table3_96pred96.sh"
EXP_VARIANT="${EXP_VARIANT:-baseline}"

OUT_DIR="logs/table3_compare"
mkdir -p "$OUT_DIR"
OUT_CSV="$OUT_DIR/compare_${SEED}_$(date +%F_%H-%M-%S).csv"

echo "dataset,paper_literal_mse,paper_literal_mae,table3_matched_mse,table3_matched_mae,exp_baseline_mse,exp_baseline_mae" > "$OUT_CSV"

extract_metric () {
  local f="$1"
  python - "$f" <<'PY'
import re,sys
text=open(sys.argv[1],encoding='utf-8',errors='ignore').read()
matches=re.findall(r"mse:([0-9.eE+-]+),\s*mae:([0-9.eE+-]+)", text)
if not matches:
    print('NA,NA')
else:
    mse, mae = matches[-1]
    print(f"{mse},{mae}")
PY
}

run_one () {
  local name="$1"; shift
  local log="$OUT_DIR/${name}.log"
  echo "[RUN] $name"
  "$@" 2>&1 | tee "$log" >/dev/null
  extract_metric "$log"
}

for ds in "${DATASETS[@]}"; do
  lit=$(run_one "${ds}_paper_literal" env DATA_ROOT="$DATA_ROOT" SEED="$SEED" NPROC_PER_NODE="$NPROC_PER_NODE" MODE=paper_literal ONLY_DATASET="$ds" bash "$REPRO_SCRIPT")
  mat=$(run_one "${ds}_table3_matched" env DATA_ROOT="$DATA_ROOT" SEED="$SEED" NPROC_PER_NODE="$NPROC_PER_NODE" MODE=table3_matched ONLY_DATASET="$ds" bash "$REPRO_SCRIPT")
  exp=$(run_one "${ds}_exp_baseline" env DATA_ROOT="$DATA_ROOT" SEED="$SEED" NGPU="$NGPU" ONLY_DATASET="$ds" ONLY_VARIANT="$EXP_VARIANT" bash "$EXP_SCRIPT")

  IFS=',' read -r lit_mse lit_mae <<< "$lit"
  IFS=',' read -r mat_mse mat_mae <<< "$mat"
  IFS=',' read -r exp_mse exp_mae <<< "$exp"

  echo "$ds,$lit_mse,$lit_mae,$mat_mse,$mat_mae,$exp_mse,$exp_mae" >> "$OUT_CSV"
done

echo "[DONE] 比较结果已保存: $OUT_CSV"
