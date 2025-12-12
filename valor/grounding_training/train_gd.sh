#!/usr/bin/env bash
set -euo pipefail

GPU_NUM=${1:-1}
CFG=${2:-"valor/grounding_training/Open-GroundingDino/config/cfg_odvg.py"}
DATASETS=${3:-"valor/grounding_training/data/odvg/merged_dataset.json"}
OUTPUT_DIR=${4:-"valor/grounding_training/checkpoints"}

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29499}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.2"}

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
OGD_ROOT="${SCRIPT_DIR}/Open-GroundingDino"

cd "${PROJECT_ROOT}"

UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-"${PROJECT_ROOT}/.venv"}
export UV_PROJECT_ENVIRONMENT

PRETRAIN_MODEL_PATH=${PRETRAIN_MODEL_PATH:-"modules/GroundingDINO/weights/groundingdino_swint_ogc.pth"}
TEXT_ENCODER_TYPE=${TEXT_ENCODER_TYPE:-"bert-base-uncased"}

echo "
GPU_NUM = ${GPU_NUM}
CFG = ${CFG}
DATASETS = ${DATASETS}
OUTPUT_DIR = ${OUTPUT_DIR}
NNODES = ${NNODES}
NODE_RANK = ${NODE_RANK}
PORT = ${PORT}
MASTER_ADDR = ${MASTER_ADDR}
PRETRAIN_MODEL_PATH = ${PRETRAIN_MODEL_PATH}
TEXT_ENCODER_TYPE = ${TEXT_ENCODER_TYPE}
UV_PROJECT_ENVIRONMENT = ${UV_PROJECT_ENVIRONMENT}
"


mkdir -p "${OUTPUT_DIR}"

uv run --active -- python -m torch.distributed.launch \
  --nproc_per_node="${GPU_NUM}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${PORT}" \
  "${OGD_ROOT}/main.py" \
    --output_dir "${OUTPUT_DIR}" \
    -c "${CFG}" \
    --datasets "${DATASETS}" \
    --pretrain_model_path "${PRETRAIN_MODEL_PATH}" \
    --options text_encoder_type="${TEXT_ENCODER_TYPE}"
