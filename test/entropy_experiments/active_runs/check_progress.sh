#!/bin/bash

echo "=== Entropy Evaluation Progress Check ==="
echo "Time: $(date)"
echo ""

# Check if process is running
PID=$(cat large_entropy_eval.pid 2>/dev/null || echo "unknown")
if ps -p $PID > /dev/null 2>&1; then
    echo "✅ Evaluation is RUNNING (PID: $PID)"
    
    # Get runtime
    RUNTIME=$(ps -o etime= -p $PID 2>/dev/null | tr -d ' ')
    echo "⏱️  Runtime: $RUNTIME"
else
    echo "❌ Evaluation is NOT running"
fi

echo ""

# Get latest progress
echo "📊 Latest Progress:"
LATEST_Q=$(grep -E "gsm8k Q[0-9]+:" large_entropy_eval_log.txt 2>/dev/null | tail -1)
if [ ! -z "$LATEST_Q" ]; then
    echo "$LATEST_Q"
else
    echo "No questions processed yet or log file not found"
fi

echo ""

# Get overall accuracy trend
echo "📈 Accuracy Trend (last 5 questions):"
grep "Overall:" large_entropy_eval_log.txt 2>/dev/null | tail -5 | sed 's/.*Overall: //' || echo "No accuracy data yet"

echo ""

# Count total questions processed
TOTAL_PROCESSED=$(grep -c "gsm8k Q[0-9]+:" large_entropy_eval_log.txt 2>/dev/null || echo "0")
echo "📋 Questions processed: $TOTAL_PROCESSED / 200 ($(echo "scale=1; $TOTAL_PROCESSED * 100 / 200" | bc 2>/dev/null || echo "0")%)"

echo ""

# Estimate completion time if we have data
if [ "$TOTAL_PROCESSED" -gt 5 ] && ps -p $PID > /dev/null 2>&1; then
    RUNTIME_SECONDS=$(ps -o etimes= -p $PID 2>/dev/null | tr -d ' ')
    if [ ! -z "$RUNTIME_SECONDS" ] && [ "$RUNTIME_SECONDS" -gt 0 ]; then
        AVG_TIME_PER_Q=$(echo "scale=1; $RUNTIME_SECONDS / $TOTAL_PROCESSED" | bc 2>/dev/null)
        REMAINING_Q=$((200 - TOTAL_PROCESSED))
        ETA_SECONDS=$(echo "$REMAINING_Q * $AVG_TIME_PER_Q" | bc 2>/dev/null)
        ETA_MINUTES=$(echo "scale=0; $ETA_SECONDS / 60" | bc 2>/dev/null)
        echo "⏰ Estimated completion: ${ETA_MINUTES} minutes"
    fi
fi

echo ""
echo "To monitor live: tail -f large_entropy_eval_log.txt"
echo "To check this again: ./check_progress.sh"