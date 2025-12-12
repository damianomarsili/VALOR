#!/usr/bin/env bash

# Environment
CUDA_VISIBLE_DEVICES="0,1"
VLLM_USE_FLASHINFER_SAMPLER=0
PYTHONUNBUFFERED=1
GENAI_CREDS_PATH="PATH/TO/AUTH.json"
GENAI_PROJECT_ID="project-name"

# Data
TRAIN_FILE="valor/reasoning_training/data/reasoning_data.jsonl"
VAL_FILE="valor/reasoning_training/data/reasoning_data.jsonl"
TRAIN_BATCH_SIZE=64
MAX_PROMPT_LEN=6000
MAX_RESPONSE_LEN=2000
SYSTEM_PROMPT="valor/prompts/system_prompt.jinja"

# Model / optimization
MODEL_PATH="Qwen/Qwen3-8B"
LR=1e-6
PPO_MINI_BATCH_SIZE=64
PPO_MICRO_BATCH_SIZE_PER_GPU=1
GPU_MEMORY_UTILIZATION=0.7
ROLLOUT_N=5

# Trainer / logging
PROJECT_NAME="VALOR_Training"
EXPERIMENT_NAME="valor_llm_reasoning"
N_GPUS_PER_NODE=2
# N_GPUS_PER_NODE=4
NNODES=1
DEFAULT_LOCAL_DIR="valor/reasoning_training/checkpoints/llm_reasoning"
SAVE_FREQ=10
TEST_FREQ=10
TOTAL_EPOCHS=4

# Reward
CUSTOM_REWARD_PATH="valor/reasoning_training/verifier_reward.py"
CUSTOM_REWARD_NAME="compute_score_batched"

#########################
# Exports
#########################

export CUDA_VISIBLE_DEVICES
export VLLM_USE_FLASHINFER_SAMPLER
export PYTHONUNBUFFERED
export GENAI_CREDS_PATH
export GENAI_PROJECT_ID

#########################
# Command
#########################

uv run -- python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.max_response_length="${MAX_RESPONSE_LEN}" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.prompt_key="problem" \
    data.shuffle=True \
    +data.system_prompt="${SYSTEM_PROMPT}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=batch \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.default_local_dir="${DEFAULT_LOCAL_DIR}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.val_before_train=False \
    custom_reward_function.path="${CUSTOM_REWARD_PATH}" \
    custom_reward_function.name="${CUSTOM_REWARD_NAME}" "$@"
