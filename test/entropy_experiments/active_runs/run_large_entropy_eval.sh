#!/bin/bash

# Run large-scale entropy evaluation for GSM8K
echo "Starting large-scale GSM8K entropy evaluation..."
echo "Time: $(date)"
echo "Running with 200 questions and 16 samples per question"

# Run GSM8K evaluation with more samples
python ../../multi_dataset_entropy_eval.py \
    --datasets gsm8k \
    --max_questions 200 \
    --samples_per_question 16 \
    --output_dir ../../large_scale_entropy_results \
    > large_entropy_eval_log.txt 2>&1 &

PID=$!
echo "Started evaluation with PID: $PID"
echo "Monitor progress with: tail -f large_entropy_eval_log.txt"
echo "Check if running with: ps -p $PID"

# Save PID for later reference
echo $PID > large_entropy_eval.pid

echo ""
echo "This evaluation will:"
echo "- Process 200 GSM8K questions"
echo "- Generate 16 samples per question (3,200 total samples)"
echo "- Track entropy at 4 different token ranges"
echo "- Analyze correlation between entropy and accuracy"
echo ""
echo "Estimated runtime: 2-3 hours"