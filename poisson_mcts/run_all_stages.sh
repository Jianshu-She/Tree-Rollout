#!/bin/bash
# Serial run of 4-method advantage comparison across step_40, step_80, step_120.
# step_0 already done. Runs serially to avoid GPU conflicts (each job needs TP=2).

set -e
cd "$(dirname "$0")/.."

PROBLEMS=$(python -c "print(','.join(str(i) for i in range(100)))")
GPU_IDS="0,1"

for STAGE in step_40 step_80 step_120; do
    OUTPUT_DIR="poisson_mcts/results/advantage_comparison/${STAGE}"
    mkdir -p "$OUTPUT_DIR"
    echo "========================================"
    echo "STAGE: $STAGE  ($(date))"
    echo "========================================"
    bash poisson_mcts/run_advantage_comparison.sh \
        --stage "$STAGE" \
        --problems "$PROBLEMS" \
        --gpu_ids "$GPU_IDS" \
        --output_dir "$OUTPUT_DIR"

    # Regenerate plots for this stage immediately
    python poisson_mcts/plot_advantage_comparison.py \
        "$OUTPUT_DIR/comparison_${STAGE}.json"

    python poisson_mcts/compute_matched_analysis.py \
        "$OUTPUT_DIR/comparison_${STAGE}.json"

    echo "==== $STAGE done at $(date) ===="
done

echo "ALL STAGES COMPLETE at $(date)"
