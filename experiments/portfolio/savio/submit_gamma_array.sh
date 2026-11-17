#!/bin/bash
#SBATCH --job-name=gamma_sgd
#SBATCH --partition=savio3_htc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --array=0-17
#SBATCH --output=gamma_sgd_%A_task_%a.out
#SBATCH --error=gamma_sgd_%A_task_%a.err
#SBATCH --export=NONE

# Fixed-gamma SAVIO submission. Adaptive gamma has a separate wrapper:
# savio/submit_adaptive_gamma_array.sh

# set -euo pipefail
set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
export MKL_INTERFACE_LAYER=LP64
module load python/3.11.6-gcc-11.4.0
module load ml/pytorch/2.3.1-py3.11.7

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

BASIS="${1:-random_fourier}"
BASIS_DIM="${2:-16}"
BASIS_SEED="${3:-8675309}"
RANDOM_FEATURE_SCALE="${4:-}"
RANDOM_BIAS_SCALE="${5:-1.0}"
WEIGHT_INIT="${6:-random}"
INIT_SCALE="${7:-0.025}" # 0.025 if fourier, 0.05 if relu.
EPOCHS="${8:-}"
BETA_SNAPSHOT_EPOCHS="${9:-}"
GRID_CSV="${10:-}"

comma_epoch_sequence() {
  local end_epoch="$1"
  local step="$2"
  local epochs=""
  local epoch=0
  while [[ "${epoch}" -le "${end_epoch}" ]]; do
    if [[ -z "${epochs}" ]]; then
      epochs="${epoch}"
    else
      epochs="${epochs},${epoch}"
    fi
    epoch=$((epoch + step))
  done
  echo "${epochs}"
}

if [[ -z "${GRID_CSV}" ]]; then
  if [[ "${BASIS}" == "random_fourier" ]]; then
    GRID_CSV="savio/gamma_grid_fourier.csv"
  elif [[ "${BASIS}" == "random_relu" ]]; then
    GRID_CSV="savio/gamma_grid_relu.csv"
  elif [[ "${BASIS}" == "taylor" ]]; then
    GRID_CSV="savio/gamma_grid_taylor.csv"
  else
    GRID_CSV="savio/gamma_grid.csv"
  fi
fi

if [[ -z "${EPOCHS}" ]]; then
  if [[ "${BASIS}" == "random_relu" ]]; then
    EPOCHS="20000"
  elif [[ "${BASIS}" == "random_fourier" ]]; then
    EPOCHS="2000"
  else
    EPOCHS="20000"
  fi
fi

if [[ -z "${BETA_SNAPSHOT_EPOCHS}" ]]; then
  if [[ "${BASIS}" == "random_relu" ]]; then
    BETA_SNAPSHOT_EPOCHS="$(comma_epoch_sequence "${EPOCHS}" 500)"
  elif [[ "${BASIS}" == "random_fourier" ]]; then
    BETA_SNAPSHOT_EPOCHS="$(comma_epoch_sequence "${EPOCHS}" 50)"
  else
    QUARTER=$((EPOCHS / 4))
    HALF=$((EPOCHS / 2))
    THREE_QUARTER=$((3 * EPOCHS / 4))
    BETA_SNAPSHOT_EPOCHS="0,${QUARTER},${HALF},${THREE_QUARTER},${EPOCHS}"
  fi
fi

if [[ -z "${RANDOM_FEATURE_SCALE}" ]]; then
  if [[ "${BASIS}" == "random_fourier" ]]; then
    RANDOM_FEATURE_SCALE="0.25"
  else
    RANDOM_FEATURE_SCALE="1.0"
  fi
fi

OUT_DIR="results/gamma_sweep_sgd_${BASIS}_${WEIGHT_INIT}_init"
if [[ "${BASIS}" == "taylor" ]]; then
  OUT_DIR="results/gamma_sweep_sgd_${WEIGHT_INIT}_init"
fi
mkdir -p "${OUT_DIR}"

CMD=(
  python scripts/run_gamma_experiment.py
  --grid-csv "${GRID_CSV}" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --epochs "${EPOCHS}" \
  --record-frequency 10 \
  --trace-frequency 100 \
  --diagnostic-gamma 10 \
  --diagnostic-start-epoch 12450 \
  --diagnostic-end-epoch 12550 \
  --device cpu \
  --basis "${BASIS}" \
  --basis-seed "${BASIS_SEED}" \
  --random-feature-scale "${RANDOM_FEATURE_SCALE}" \
  --random-bias-scale "${RANDOM_BIAS_SCALE}" \
  --weight-init "${WEIGHT_INIT}" \
  --init-scale "${INIT_SCALE}" \
  --beta-snapshot-epochs "${BETA_SNAPSHOT_EPOCHS}" \
  --out-dir "${OUT_DIR}"
)

if [[ -n "${BASIS_DIM}" ]]; then
  CMD+=(--basis-dim "${BASIS_DIM}")
fi

"${CMD[@]}"
