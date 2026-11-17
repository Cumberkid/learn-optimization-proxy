#!/bin/bash
#SBATCH --job-name=wc_gap
#SBATCH --partition=savio3_htc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=0-17
#SBATCH --output=wc_gap_%A_task_%a.out
#SBATCH --error=wc_gap_%A_task_%a.err
#SBATCH --export=NONE

# One task computes all requested snapshot epochs for one gamma.
# The default grids currently have 18 gamma rows, so use --array=0-17.
# If the gamma count changes, update the SBATCH array range or pass
# a different range at submission time with: sbatch --array=0-N ...

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
export MKL_INTERFACE_LAYER=LP64
module load python/3.11.6-gcc-11.4.0
module load ml/pytorch/2.3.1-py3.11.7

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

BASIS="${1:-random_relu}"
OUT_DIR="${2:-}"
GRID_CSV="${3:-}"
SNAPSHOT_EPOCHS="${4:-}"
NUM_RANDOM_STARTS="${5:-2048}"
NUM_LOCAL_STARTS="${6:-16}"
USE_DIFFERENTIAL_EVOLUTION="${7:-0}"
SEED="${8:-8675309}"

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
  else
    GRID_CSV="savio/gamma_grid.csv"
  fi
fi

if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="results/gamma_sweep_sgd_${BASIS}_random_init"
fi

if [[ -z "${SNAPSHOT_EPOCHS}" ]]; then
  if [[ "${BASIS}" == "random_fourier" ]]; then
    SNAPSHOT_EPOCHS="$(comma_epoch_sequence 2000 50)"
  elif [[ "${BASIS}" == "random_relu" ]]; then
    SNAPSHOT_EPOCHS="$(comma_epoch_sequence 20000 500)"
  else
    echo "SNAPSHOT_EPOCHS must be provided for basis ${BASIS}" >&2
    exit 1
  fi
fi

CMD=(
  python scripts/compute_worst_case_gap.py
  --grid-csv "${GRID_CSV}"
  --task-id "${SLURM_ARRAY_TASK_ID}"
  --snapshot-epochs "${SNAPSHOT_EPOCHS}"
  --out-dir "${OUT_DIR}"
  --num-random-starts "${NUM_RANDOM_STARTS}"
  --num-local-starts "${NUM_LOCAL_STARTS}"
  --seed "${SEED}"
)

if [[ "${USE_DIFFERENTIAL_EVOLUTION}" == "1" ]]; then
  CMD+=(--use-differential-evolution)
  CMD+=(--optimizer-method-label "random_search_differential_evolution_lbfgsb")
fi

"${CMD[@]}"
