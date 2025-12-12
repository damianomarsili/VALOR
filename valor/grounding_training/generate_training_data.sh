#!/usr/bin/env bash
set -euo pipefail


DATASETS=${DATASETS:-"omni3d-bench, vsr"} # Comma-separated list of datasets to process
START_INDEX=${START_INDEX:-0}
MAX_SAMPLES=${MAX_SAMPLES:-""} # empty means no limit
SEED=${SEED:-2727}

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-"${PROJECT_ROOT}/.venv"}
export UV_PROJECT_ENVIRONMENT

VALOR_DATA_ROOT=${VALOR_DATA_ROOT:-"${SCRIPT_DIR}/data"}
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/data/detector_outputs"}
ODVG_DIR=${ODVG_DIR:-"${SCRIPT_DIR}/data/odvg"}
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-8B"}
GROUNDING_CONFIG=${GROUNDING_CONFIG:-"${PROJECT_ROOT}/modules/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"}
GROUNDING_WEIGHTS=${GROUNDING_WEIGHTS:-"${PROJECT_ROOT}/modules/GroundingDINO/weights/groundingdino_swint_ogc.pth"}
APPEND_ORDER=${APPEND_ORDER:-"sorted"}

# Allow optional positional override for datasets
if [[ $# -ge 1 ]]; then
  DATASETS=$1
fi

read -ra DATASET_ARGS <<<"${DATASETS}"

echo "
DATASETS = ${DATASETS}
VALOR_DATA_ROOT = ${VALOR_DATA_ROOT}
OUTPUT_DIR = ${OUTPUT_DIR}
ODVG_DIR = ${ODVG_DIR}
MODEL_NAME = ${MODEL_NAME}
GROUNDING_CONFIG = ${GROUNDING_CONFIG}
GROUNDING_WEIGHTS = ${GROUNDING_WEIGHTS}
APPEND_ORDER = ${APPEND_ORDER}
SEED = ${SEED}
START_INDEX = ${START_INDEX}
MAX_SAMPLES = ${MAX_SAMPLES:-none}
UV_PROJECT_ENVIRONMENT = ${UV_PROJECT_ENVIRONMENT}
"

cmd=(
  uv run --active -- python valor/grounding_training/generate_training_data.py
    --datasets "${DATASET_ARGS[@]}"
    --data-root "${VALOR_DATA_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --odvg-dir "${ODVG_DIR}"
    --model-name "${MODEL_NAME}"
    --seed "${SEED}"
    --start-index "${START_INDEX}"
    --grounding-config "${GROUNDING_CONFIG}"
    --grounding-weights "${GROUNDING_WEIGHTS}"
    --append-order "${APPEND_ORDER}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  cmd+=(--max-samples "${MAX_SAMPLES}")
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
