#!/bin/bash

# Run the improved GSM8K entropy evaluation
echo "Starting GSM8K entropy evaluation..."
echo "Time: $(date)"

python improved_gsm8k_entropy_eval.py \
    --max_questions 50 \
    --samples_per_question 16 \
    --output_dir ./improved_gsm8k_entropy_results \
    > entropy_eval_log.txt 2>&1 &

PID=$!
echo "Started evaluation with PID: $PID"
echo "Monitor progress with: tail -f entropy_eval_log.txt"
echo "Check if running with: ps -p $PID"

# Save PID for later reference
echo $PID > entropy_eval.pid