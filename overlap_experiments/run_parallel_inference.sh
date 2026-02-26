#!/bin/bash
# Run inference across 6 GPUs in parallel (GPUs 2-7 have full memory)
# Each GPU handles a shard of the 500 problems

set -e

SCRIPT="overlap_analysis/run_qwen_inference.py"
MODEL="Qwen/Qwen3-30B-A3B-Thinking-2507"
TOTAL_PROBLEMS=500
NUM_GPUS=6
PROBLEMS_PER_GPU=$((TOTAL_PROBLEMS / NUM_GPUS))  # 83 each, last gets remainder
OUTPUT_BASE="overlap_analysis/qwen_results"

mkdir -p "$OUTPUT_BASE"

echo "Starting parallel inference: $TOTAL_PROBLEMS problems across $NUM_GPUS GPUs"
echo "Problems per GPU: ~$PROBLEMS_PER_GPU"

PIDS=()

for i in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ID=$((i + 2))  # GPUs 2-7
    START=$((i * PROBLEMS_PER_GPU))

    if [ $i -eq $((NUM_GPUS - 1)) ]; then
        # Last GPU gets the remainder
        SHARD_SIZE=$((TOTAL_PROBLEMS - START))
    else
        SHARD_SIZE=$PROBLEMS_PER_GPU
    fi

    SHARD_DIR="${OUTPUT_BASE}/shard_${i}"
    mkdir -p "$SHARD_DIR"

    echo "GPU $GPU_ID: problems $START-$((START + SHARD_SIZE - 1)) ($SHARD_SIZE problems) -> $SHARD_DIR"

    CUDA_VISIBLE_DEVICES=$GPU_ID python $SCRIPT \
        --model "$MODEL" \
        --num_problems $SHARD_SIZE \
        --num_rollouts 12 \
        --max_tokens 4096 \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.85 \
        --output_dir "$SHARD_DIR" \
        --problem_offset $START \
        > "${SHARD_DIR}/log.txt" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} workers. PIDs: ${PIDS[*]}"
echo "Waiting for all workers to complete..."

FAILED=0
for i in $(seq 0 $((NUM_GPUS - 1))); do
    wait ${PIDS[$i]} || { echo "Worker $i (PID ${PIDS[$i]}) FAILED"; FAILED=$((FAILED + 1)); }
done

if [ $FAILED -gt 0 ]; then
    echo "WARNING: $FAILED workers failed!"
else
    echo "All workers completed successfully!"
fi

# Merge shards
echo "Merging shards..."
python overlap_analysis/merge_shards.py \
    --shard_dirs $(for i in $(seq 0 $((NUM_GPUS - 1))); do echo "${OUTPUT_BASE}/shard_${i}"; done) \
    --output "${OUTPUT_BASE}/inference_results.json"

echo "Done! Results at ${OUTPUT_BASE}/inference_results.json"
