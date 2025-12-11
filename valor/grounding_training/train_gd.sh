#!/bin/bash

GPU_NUM=$1
CFG=$2
DATASETS=$3
OUTPUT_DIR=$4

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29499}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.2"}

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OGD_ROOT="${SCRIPT_DIR}/Open-GroundingDino"

PRETRAIN_MODEL_PATH=${PRETRAIN_MODEL_PATH:-"path-to-groundingdino_swint_ogc.pth"}
TEXT_ENCODER_TYPE=${TEXT_ENCODER_TYPE:-"bert-base-uncased"}

echo "
GPU_NUM = $GPU_NUM
CFG = $CFG
DATASETS = $DATASETS
OUTPUT_DIR = $OUTPUT_DIR
NNODES = $NNODES
NODE_RANK = $NODE_RANK
PORT = $PORT
MASTER_ADDR = $MASTER_ADDR
PRETRAIN_MODEL_PATH = $PRETRAIN_MODEL_PATH
TEXT_ENCODER_TYPE = $TEXT_ENCODER_TYPE
"

python -m torch.distributed.launch \
  --nproc_per_node="${GPU_NUM}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${PORT}" \
  "${OGD_ROOT}/main.py" \
    --output_dir "${OUTPUT_DIR}" \
    -c "${CFG}" \
    --datasets "${DATASETS}" \
    --pretrain_model_path "${PRETRAIN_MODEL_PATH}" \
    --options text_encoder_type="$TEXT_ENCODER_TYPE"
