# Entropy vs Correctness Evaluation

This project evaluates the relationship between model entropy and answer correctness using Qwen2.5-7B-Instruct on math datasets (MATH-500 and PolyMath-en).

## Setup

### Environment Creation
A conda environment `entropy_eval` has been created with all necessary dependencies:

```bash
# Activate environment
conda activate entropy_eval

# The environment includes:
# - PyTorch (CPU version installed)
# - Transformers 4.54.1
# - Datasets 4.0.0
# - NumPy, pandas, scipy, matplotlib
# - tqdm for progress tracking
```

### GPU Support (Optional)
For faster inference, install PyTorch with CUDA:

```bash
conda activate entropy_eval
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Files Overview

### Main Scripts

1. **`entropy_eval.py`** - Main evaluation script
   - Loads Qwen2.5-7B-Instruct model
   - Evaluates on MATH-500 and PolyMath-en datasets
   - Tracks entropy for first 100 tokens of each generation
   - Computes answer correctness
   - Generates correlation analysis and visualizations

2. **`test_entropy_setup.py`** - Environment verification
   - Tests all package imports
   - Verifies entropy calculation works
   - Quick environment health check

3. **`test_small_eval.py`** - Small model test
   - Uses DialoGPT-small for quick testing
   - Demonstrates entropy tracking functionality

4. **`qwen_entropy_test.py`** - Qwen model test
   - Tests Qwen2.5-7B-Instruct on 5 sample problems
   - Smaller scale demonstration of full evaluation

### Helper Scripts

- **`setup_env.sh`** - Environment setup script
- **`run_entropy_eval.sh`** - Main evaluation runner
- **`README_entropy_eval.md`** - This documentation

## Running the Evaluation

### Quick Test (Recommended First)
```bash
conda activate entropy_eval
python test_entropy_setup.py
```

### Small Scale Test with Qwen
```bash
conda activate entropy_eval
python qwen_entropy_test.py
```

### Full Evaluation
```bash
conda activate entropy_eval
bash run_entropy_eval.sh
```

Or run directly:
```bash
python entropy_eval.py --model "Qwen/Qwen2.5-7B-Instruct" --device "cpu" --max_samples 100 --output_dir "./entropy_results"
```

### Parameters
- `--model`: Model name (default: "Qwen/Qwen2.5-7B-Instruct")
- `--device`: Device to use ("cpu", "cuda", or "auto")
- `--max_samples`: Maximum samples per dataset (default: 100)
- `--output_dir`: Output directory for results (default: "./entropy_results")

## Expected Output

The evaluation will generate:

1. **JSON Results Files**:
   - `math_500_results.json` - MATH-500 dataset results
   - `polymath_results.json` - PolyMath-en dataset results  
   - `all_results.json` - Combined results
   - `analysis.json` - Statistical analysis

2. **Visualizations**:
   - `entropy_analysis.png` - Multi-panel visualization showing:
     - Entropy distribution by correctness
     - Scatter plot of entropy vs correctness
     - Box plots comparing correct/incorrect entropy
     - Token-level entropy evolution

3. **Console Output**:
   - Progress tracking during evaluation
   - Statistical summary including:
     - Total problems evaluated
     - Accuracy rate
     - Average entropy for correct vs incorrect answers
     - Correlation coefficients (Pearson & Spearman)
     - Statistical significance tests

## Key Features

### Entropy Calculation
- Calculates Shannon entropy at each token generation step
- Tracks entropy for first 100 tokens of each response
- Uses softmax probabilities over vocabulary

### Answer Extraction
- Multiple regex patterns for math answer extraction
- Handles various formats: "The answer is X", "Answer: X", boxed notation
- Numerical normalization and comparison

### Correctness Evaluation
- String matching and numerical comparison
- Handles approximate matches for floating point numbers
- Partial credit for substring matching

### Statistical Analysis
- Pearson and Spearman correlation coefficients
- Statistical significance testing
- Entropy statistics (mean, std, min, max) by correctness group

## Research Questions Addressed

1. **Is there a correlation between model entropy and answer correctness?**
   - Measured via Pearson/Spearman correlation
   - Hypothesis: Higher entropy → lower confidence → more errors

2. **Do correct and incorrect answers have different entropy distributions?**
   - Compared via statistical tests and visualizations
   - Entropy patterns across token positions

3. **How does entropy evolve during generation?**
   - Token-by-token entropy tracking
   - Different patterns for correct vs incorrect responses

## Memory and Performance Notes

- **CPU Mode**: Works but slow (~1-2 problems/minute)
- **GPU Mode**: Much faster (~10-20 problems/minute)
- **Memory**: Qwen2.5-7B requires ~15GB RAM minimum
- **Disk Space**: Model downloads ~14GB

## Potential Extensions

1. **More Datasets**: Add other math/reasoning datasets
2. **Different Models**: Compare across model sizes/families  
3. **Sampling Strategies**: Test different decoding methods
4. **Confidence Calibration**: Relate entropy to calibrated confidence
5. **Error Analysis**: Categorize error types by entropy patterns

## Troubleshooting

### Common Issues

1. **Out of Memory**: Use smaller batch size or CPU mode
2. **Model Download Fails**: Check internet connection, try resume
3. **Dataset Loading Issues**: Some datasets may not be available
4. **Import Errors**: Verify conda environment activation

### Solutions
```bash
# Reset environment if needed
conda deactivate
conda remove -n entropy_eval --all
# Then re-run setup

# For memory issues
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# For dataset issues
huggingface-cli login  # May need authentication
```

## Results Interpretation

### Correlation Values
- **r > 0.3**: Moderate positive correlation (higher entropy → more errors)  
- **r < -0.3**: Moderate negative correlation (higher entropy → fewer errors)
- **|r| < 0.1**: Weak/no correlation
- **p < 0.05**: Statistically significant

### Practical Implications
- Strong correlations suggest entropy could be used as uncertainty estimate
- Weak correlations might indicate model is well-calibrated
- Patterns could inform early stopping or confidence thresholding strategies

This evaluation framework provides a comprehensive analysis of the relationship between model entropy and mathematical reasoning accuracy.