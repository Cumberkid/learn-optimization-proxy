#!/bin/bash
#SBATCH --job-name=adaptive_gamma_sgd
#SBATCH --partition=savio3_htc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --array=0-0
#SBATCH --output=adaptive_gamma_sgd_%A_task_%a.out
#SBATCH --error=adaptive_gamma_sgd_%A_task_%a.err
#SBATCH --export=NONE

# Adaptive-gamma SAVIO submission. Fixed gamma has a separate wrapper:
# savio/submit_gamma_array.sh

# set -euo pipefail
set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
export MKL_INTERFACE_LAYER=LP64
module load python/3.11.6-gcc-11.4.0
module load ml/pytorch/2.3.1-py3.11.7

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

BASIS="${1:-random_fourier}"
BASIS_DIM="${2:-64}" # 64 if fourier, 16 if relu.
BASIS_SEED="${3:-8675309}"
RANDOM_FEATURE_SCALE="${4:-}"
RANDOM_BIAS_SCALE="${5:-1.0}"
WEIGHT_INIT="${6:-random}"
INIT_SCALE="${7:-0.025}" # 0.025 if fourier, 0.05 if relu.
EPOCHS="${8:-}"
BETA_SNAPSHOT_EPOCHS="${9:-}"
GAMMA_START="${10:-368.15625}"
GAMMA_SHRINK="${11:-0.1}"
SATURATION_METRIC="${12:-gradient}"
SATURATION_THRESHOLD_GRID_CSV="${13:-savio/adaptive_gamma_threshold_grid.csv}"
SATURATION_WINDOW="${14:-3}"
THRESHOLD_ALPHA="${15:-0.5}"
THRESHOLD_SCALE_MIN="${16:-0.01}"
THRESHOLD_SCALE_MAX="${17:-100.0}"
ADAPTIVE_LR_C="${18:-}"
MAX_EPOCHS_PER_GAMMA="${19:-1000}"

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

SATURATION_THRESHOLD="$(
  awk -F, -v row="$((SLURM_ARRAY_TASK_ID + 2))" 'NR == row {gsub(/^[ \t]+|[ \t]+$/, "", $1); print $1}' "${SATURATION_THRESHOLD_GRID_CSV}"
)"
if [[ -z "${SATURATION_THRESHOLD}" ]]; then
  echo "No saturation threshold found for array task ${SLURM_ARRAY_TASK_ID} in ${SATURATION_THRESHOLD_GRID_CSV}" >&2
  exit 1
fi

OUT_DIR="results/adaptive_gamma_sgd_${BASIS}_${WEIGHT_INIT}_init"
if [[ "${BASIS}" == "taylor" ]]; then
  OUT_DIR="results/adaptive_gamma_sgd_${WEIGHT_INIT}_init"
fi
mkdir -p "${OUT_DIR}"

CMD=(
  python scripts/run_gamma_experiment.py
  --gamma "${GAMMA_START}" \
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
  --out-dir "${OUT_DIR}" \
  --adaptive-gamma \
  --adaptive-gamma-start "${GAMMA_START}" \
  --adaptive-gamma-shrink "${GAMMA_SHRINK}" \
  --adaptive-gamma-saturation-metric "${SATURATION_METRIC}" \
  --adaptive-gamma-saturation-threshold "${SATURATION_THRESHOLD}" \
  --adaptive-gamma-window "${SATURATION_WINDOW}" \
  --adaptive-gamma-threshold-alpha "${THRESHOLD_ALPHA}" \
  --adaptive-gamma-threshold-scale-min "${THRESHOLD_SCALE_MIN}" \
  --adaptive-gamma-threshold-scale-max "${THRESHOLD_SCALE_MAX}"
)

if [[ -n "${BASIS_DIM}" ]]; then
  CMD+=(--basis-dim "${BASIS_DIM}")
fi

if [[ -n "${ADAPTIVE_LR_C}" ]]; then
  CMD+=(--adaptive-lr-c "${ADAPTIVE_LR_C}")
fi

if [[ -n "${MAX_EPOCHS_PER_GAMMA}" ]]; then
  CMD+=(--adaptive-gamma-max-epochs-per-value "${MAX_EPOCHS_PER_GAMMA}")
fi

"${CMD[@]}"

