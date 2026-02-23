#!/bin/bash

echo "🚀 Starting GPU-optimized entropy evaluation with 8x H200 GPUs"
echo "================================================================"

# Setup environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate entropy_eval

# Setup GPU environment if not already done
if [ ! -f ".gpu_setup_done" ]; then
    echo "Setting up GPU environment..."
    bash setup_gpu_env.sh
    touch .gpu_setup_done
fi

# Set environment variables for optimal GPU performance
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Create results directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="./entropy_results_gpu_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

echo "🔧 Configuration:"
echo "  Model: Qwen/Qwen2.5-7B-Instruct"
echo "  GPUs: 8x H200 ($(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1))"
echo "  Output: $OUTPUT_DIR"
echo "  Max samples per dataset: 500"

# Run the evaluation
echo ""
echo "🧮 Starting entropy evaluation..."
python entropy_eval_gpu.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --device_map "auto" \
    --max_samples 500 \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "✅ Evaluation complete!"
echo "📊 Results available in: $OUTPUT_DIR"
echo ""
echo "📋 Summary of outputs:"
echo "  - math_500_results.json: MATH-500 dataset results"
echo "  - polymath_results.json: PolyMath-en dataset results"
echo "  - all_results.json: Combined results"
echo "  - analysis.json: Statistical analysis"
echo "  - entropy_analysis_gpu.png: Comprehensive visualizations"
echo ""
echo "🔍 To view results:"
echo "  cd $OUTPUT_DIR"
echo "  cat analysis.json | python -m json.tool"
echo "  # Or open entropy_analysis_gpu.png to see visualizations"