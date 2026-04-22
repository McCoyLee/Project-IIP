#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?请先 export DATA_ROOT=/你的/数据根目录}"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NGPU="${NGPU:-$NPROC_PER_NODE}"
SEED="${SEED:-2021}"
DATASETS=(ecl etth1 etth2 ettm1 ettm2 weather traffic solar exchange)

# Timer-XL Table 3 targets available for these datasets in paper figure/table.
# other datasets keep NA and only做横向方法比较。
declare -A TGT_MSE=(
  [ecl]=0.138
  [etth1]=0.381
  [traffic]=0.387
  [weather]=0.165
  [solar]=0.200
)
declare -A TGT_MAE=(
  [ecl]=0.233
  [etth1]=0.399
  [traffic]=0.260
  [weather]=0.209
  [solar]=0.229
)

REPRO_SCRIPT="scripts/jobs/table3_96pred96_paper_multigpu.sh"
EXP_SCRIPT="scripts/jobs/table3_96pred96.sh"
EXP_VARIANT="${EXP_VARIANT:-baseline}"

OUT_DIR="logs/table3_compare"
mkdir -p "$OUT_DIR"
OUT_CSV="$OUT_DIR/compare_table3_all9_${SEED}_$(date +%F_%H-%M-%S).csv"

echo "dataset,target_mse,target_mae,paper_literal_mse,paper_literal_mae,paper_literal_gap_mse,paper_literal_gap_mae,table3_matched_mse,table3_matched_mae,table3_matched_gap_mse,table3_matched_gap_mae,orig_baseline_mse,orig_baseline_mae" > "$OUT_CSV"

extract_metric () {
  local f="$1"
  python - "$f" <<'PY'
import re,sys,os
p=sys.argv[1]
if not os.path.exists(p):
    print('NA,NA'); raise SystemExit(0)
text=open(p,encoding='utf-8',errors='ignore').read()
matches=re.findall(r"mse:([0-9.eE+-]+),\s*mae:([0-9.eE+-]+)", text)
if not matches:
    print('NA,NA')
else:
    mse, mae = matches[-1]
    print(f"{mse},{mae}")
PY
}

abs_gap () {
  python - "$1" "$2" <<'PY'
import sys
try:
    a=float(sys.argv[1]); b=float(sys.argv[2])
    print(f"{abs(a-b):.6f}")
except Exception:
    print("NA")
PY
}

run_one () {
  local name="$1"; shift
  local log="$OUT_DIR/${name}.log"
  echo "[RUN] $name"
  set +e
  "$@" 2>&1 | tee "$log" >/dev/null
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[WARN] $name failed (exit=$rc), fill NA" >&2
    echo "NA,NA"
    return 0
  fi
  extract_metric "$log"
}

for ds in "${DATASETS[@]}"; do
  tgt_mse="${TGT_MSE[$ds]:-NA}"
  tgt_mae="${TGT_MAE[$ds]:-NA}"

  lit=$(run_one "${ds}_paper_literal" env DATA_ROOT="$DATA_ROOT" SEED="$SEED" NPROC_PER_NODE="$NPROC_PER_NODE" MODE=paper_literal ONLY_DATASET="$ds" bash "$REPRO_SCRIPT")
  mat=$(run_one "${ds}_table3_matched" env DATA_ROOT="$DATA_ROOT" SEED="$SEED" NPROC_PER_NODE="$NPROC_PER_NODE" MODE=table3_matched ONLY_DATASET="$ds" bash "$REPRO_SCRIPT")
  ori=$(run_one "${ds}_orig_baseline" env DATA_ROOT="$DATA_ROOT" SEED="$SEED" NGPU="$NGPU" ONLY_DATASET="$ds" ONLY_VARIANT="$EXP_VARIANT" bash "$EXP_SCRIPT")

  IFS=',' read -r lit_mse lit_mae <<< "$lit"
  IFS=',' read -r mat_mse mat_mae <<< "$mat"
  IFS=',' read -r ori_mse ori_mae <<< "$ori"

  lit_gap_mse=$(abs_gap "$lit_mse" "$tgt_mse")
  lit_gap_mae=$(abs_gap "$lit_mae" "$tgt_mae")
  mat_gap_mse=$(abs_gap "$mat_mse" "$tgt_mse")
  mat_gap_mae=$(abs_gap "$mat_mae" "$tgt_mae")

  echo "$ds,$tgt_mse,$tgt_mae,$lit_mse,$lit_mae,$lit_gap_mse,$lit_gap_mae,$mat_mse,$mat_mae,$mat_gap_mse,$mat_gap_mae,$ori_mse,$ori_mae" >> "$OUT_CSV"
done

echo "[DONE] 结果已保存: $OUT_CSV"
