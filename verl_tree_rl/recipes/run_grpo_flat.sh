#!/bin/bash
# Part 4 Phase 1: flat pass-through via TreeFaithfulRollout.
#
# This should produce IDENTICAL results to rollout.name=vllm.
# Purpose: verify the wrapper mechanism works before adding tree engines.
#
# Usage:
#   cd /mnt/weka/home/jianshu.she/MCTS
#   bash verl_tree_rl/recipes/run_grpo_flat.sh
#
# To switch to tree methods (Phase 3+), change TREE_METHOD below.

set -xeuo pipefail

# ---- Tree method ----
TREE_METHOD="${TREE_METHOD:-flat}"          # flat / bfs / negbin / deepsearch
INNER_ROLLOUT="${INNER_ROLLOUT:-vllm}"      # underlying engine

# ---- Project ----
project_name='TreeRL'
exp_name="grpo-${TREE_METHOD}"

# ---- Algorithm ----
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode="token-mean"

# ---- Data ----
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 8))
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

# ---- Batch (small for smoke test; increase for full run) ----
train_prompt_bsz="${TRAIN_BSZ:-64}"
n_resp_per_prompt="${N_RESP:-8}"
train_prompt_mini_bsz="${MINI_BSZ:-16}"
total_steps="${TOTAL_STEPS:-200}"

# ---- Hardware ----
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
gen_tp=${GEN_TP:-2}
fsdp_size=${FSDP_SIZE:-4}
sp_size=1

# ---- Paths ----
COPUS_HOME="/mnt/weka/home/jianshu.she/copus"
MODEL_PATH="${MODEL_PATH:-${COPUS_HOME}/verl/models/Qwen2.5-Math-7B}"
CKPTS_DIR="${COPUS_HOME}/verl/ckpts/${project_name}/${exp_name}"
TRAIN_FILE="${TRAIN_FILE:-${COPUS_HOME}/verl/data/dapo-math-17k.parquet}"
TEST_FILE="${TEST_FILE:-${COPUS_HOME}/verl/data/dapo-math-500-test.parquet}"

# ---- Sampling ----
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

# ---- Performance ----
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
offload=True

# ---- Setup PYTHONPATH for verl_tree_rl ----
MCTS_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${MCTS_ROOT}:${PYTHONPATH:-}"
VERL_PYTHON="/mnt/weka/home/jianshu.she/miniconda3/envs/verl/bin/python"

echo "========================================"
echo "TreeRL GRPO Training"
echo "  Method:     ${TREE_METHOD}"
echo "  Inner:      ${INNER_ROLLOUT}"
echo "  Model:      ${MODEL_PATH}"
echo "  Steps:      ${total_steps}"
echo "  Batch:      ${train_prompt_bsz} × ${n_resp_per_prompt}"
echo "  GPUs:       ${NNODES} × ${NGPUS_PER_NODE}"
echo "  PYTHONPATH: ${MCTS_ROOT}"
echo "========================================"

# Pre-flight: verify registration
${VERL_PYTHON} -c "import verl_tree_rl.register; print('[OK] tree_faithful registered')"

# ---- Launch verl trainer ----
# We inject our register import via a small wrapper that imports
# verl_tree_rl.register before calling verl's main_ppo.
${VERL_PYTHON} -c "
import verl_tree_rl.register   # hook tree_faithful into registry
from verl.trainer.main_ppo import main
main()
" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=tree_faithful \
    actor_rollout_ref.rollout.mode=sync \
    +actor_rollout_ref.rollout.engine_kwargs.tree_method=${TREE_METHOD} \
    +actor_rollout_ref.rollout.engine_kwargs.inner_rollout_name=${INNER_ROLLOUT} \
    +actor_rollout_ref.rollout.engine_kwargs.inner_mode=sync \
    +actor_rollout_ref.rollout.engine_kwargs.max_depth=${BFS_MAX_DEPTH:-3} \
    +actor_rollout_ref.rollout.engine_kwargs.tokens_per_step=${BFS_TOKENS:-128} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N:-4} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger="${LOGGER:-['console']}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-True} \
    trainer.test_freq=${TEST_FREQ:-10} \
    trainer.save_freq=${SAVE_FREQ:-10} \
    +trainer.max_actor_ckpt_to_keep=1 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=${total_steps} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto
