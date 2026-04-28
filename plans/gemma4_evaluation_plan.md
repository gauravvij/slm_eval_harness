# Gemma 4 26B A4B Evaluation Plan

## Goal
Complete benchmark evaluation of Gemma 4 26B A4B model across Q4_K_M, Q8_0, and BF16 quantization variants. Add results to model_comparison.json and update dashboard.

## Research Summary
- **Model ID**: `unsloth/gemma-4-26B-A4B-it-GGUF`
- **Available Variants**:
  - Q4_K_M: `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`
  - Q8_0: `gemma-4-26B-A4B-it-Q8_0.gguf`
  - BF16: `mmproj-BF16.gguf` (or full BF16 variant)
- **Architecture**: Gemma 4 with 26B parameters, A4B (4-bit activation) variant
- **Base URL**: https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF

## Subtasks

1. **Download Gemma 4 GGUF Models**
   - Download Q4_K_M, Q8_0, and BF16 variants
   - Verify file integrity and sizes
   - Expected output: 3 GGUF files in models/ directory

2. **Run HumanEval Benchmark (Q4_K_M)**
   - Command: `python cli.py --model gguf --model_path models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --tasks humaneval --output reports/humaneval_gemma4_q4.json`
   - Expected: 164 samples, pass@1 metric
   - Verify: JSON output with pass@k scores

3. **Run HumanEval Benchmark (Q8_0)**
   - Command: `python cli.py --model gguf --model_path models/gemma-4-26B-A4B-it-Q8_0.gguf --tasks humaneval --output reports/humaneval_gemma4_q8.json`
   - Expected: 164 samples, pass@1 metric

4. **Run HumanEval Benchmark (BF16)**
   - Command: `python cli.py --model gguf --model_path models/gemma-4-26B-A4B-it-BF16.gguf --tasks humaneval --output reports/humaneval_gemma4_bf16.json`
   - Note: BF16 requires n_ctx=32768 fix (already in gguf_adapter.py)

5. **Run HellaSwag Benchmark (Q4_K_M)**
   - Command: `python cli.py --model gguf --model_path models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --tasks hellaswag --output reports/hellaswag_gemma4_q4.json --max_samples 100`
   - Expected: 100 samples (normalized subset)

6. **Run HellaSwag Benchmark (Q8_0)**
   - Command: Similar to Q4 but with Q8_0 model path
   - Expected: 100 samples

7. **Run HellaSwag Benchmark (BF16)**
   - Command: Similar to Q4 but with BF16 model path
   - Expected: 100 samples

8. **Run BFCL Benchmark (Q4_K_M)**
   - Command: `python cli.py --model gguf --model_path models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --tasks bfcl --output reports/bfcl_gemma4_q4.json`
   - Expected: 400 samples

9. **Run BFCL Benchmark (Q8_0)**
   - Command: Similar to Q4 but with Q8_0 model path
   - Expected: 400 samples

10. **Run BFCL Benchmark (BF16)**
    - Command: Similar to Q4 but with BF16 model path
    - Expected: 400 samples

11. **Measure Inference Metrics (All Variants)**
    - Use bench_inference.py to measure TTFT, throughput, and peak RAM
    - Command: `python bench_inference.py --model_path models/<variant> --output reports/inference_gemma4_<variant>.json`
    - Run for Q4_K_M, Q8_0, and BF16

12. **Update model_comparison.json**
    - Aggregate all benchmark results into model_comparison.json
    - Add Gemma 4 entries with metadata, benchmark scores, and inference metrics
    - Verify: All 3 variants present with complete data

13. **Verify Dashboard Integration**
    - Start server and verify Gemma 4 models appear in leaderboard
    - Check RAM vs Accuracy scatter plot includes Gemma 4 points
    - Verify all metrics display correctly

## Deliverables

| File Path | Description |
|-----------|-------------|
| models/gemma-4-26B-A4B-it-*.gguf | Downloaded GGUF model files |
| reports/humaneval_gemma4_*.json | HumanEval results for all 3 variants |
| reports/hellaswag_gemma4_*.json | HellaSwag results for all 3 variants |
| reports/bfcl_gemma4_*.json | BFCL results for all 3 variants |
| reports/inference_gemma4_*.json | Inference metrics for all 3 variants |
| reports/model_comparison.json | Updated with Gemma 4 results |

## Evaluation Criteria
- All 3 Gemma 4 variants evaluated on HumanEval (164 samples each)
- All 3 variants evaluated on HellaSwag (100 samples each)
- All 3 variants evaluated on BFCL (400 samples each)
- Inference metrics captured: TTFT, throughput (tok/s), peak RAM (GB)
- model_comparison.json updated with complete Gemma 4 entries
- Dashboard displays Gemma 4 models correctly

## Notes
- BF16 variant requires n_ctx=32768 (auto-set in gguf_adapter.py line 89-91)
- Each benchmark may take 2-6 hours depending on quantization level
- Checkpoint system will resume if interrupted
- Hardware: 32-core AMD EPYC, 125GB RAM (no GPU)
