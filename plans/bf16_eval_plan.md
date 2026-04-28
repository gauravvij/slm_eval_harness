# Qwen 3.6 35B A3B BF16 Evaluation Plan

## Goal
Evaluate the BF16 (16-bit) version of Qwen 3.6 35B A3B to complete the holistic comparison across Q4, Q8, and BF16 variants.

## Research Summary
- **Model ID**: `unsloth/Qwen3.6-35B-A3B-GGUF`
- **Quantization Files**: 
  - `BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf`
  - `BF16/Qwen3.6-35B-A3B-BF16-00002-of-00002.gguf`
- **Hardware Constraint**: BF16 weights will occupy ~70GB of RAM. The 32-core EPYC setup has 125GB RAM, which is sufficient. Inference speed is expected to be significantly lower than Q4/Q8.

## Approach
1. **Model Acquisition**: Download the BF16 GGUF shards using `huggingface_hub`.
2. **Inference Benchmarking**: Measure TTFT and throughput on the 32-core EPYC hardware.
3. **Accuracy Evaluation**: Run HumanEval, HellaSwag (mini), and BFCL benchmarks.
4. **Comparative Analysis**: Update the dashboard and `model_comparison.json` to show the full spectrum: Q4 vs Q8 vs BF16.

## Subtasks
1. Download BF16 GGUF shards from HF, expected output: shards in `BF16/` directory.
2. Run inference performance test (TTFT, throughput), expected output: `reports/inference_metrics_bf16.json`.
3. Run HumanEval benchmark, expected output: accuracy score in `reports/humaneval_bf16.json`.
4. Run HellaSwag benchmark (mini), expected output: accuracy score in `reports/hellaswag_bf16.json`.
5. Run BFCL benchmark with `function_call_match` metric, expected output: accuracy score in `reports/bfcl_bf16.json`.
6. Update `reports/model_comparison.json` and dashboard with BF16 results and 3-way comparison view.

## Deliverables
| File Path | Description |
|-----------|-------------|
| `/root/slm_eval_harness/reports/inference_metrics_bf16.json` | Performance data for BF16 variant |
| `/root/slm_eval_harness/reports/qwen3.6_35b_bf16_summary.json` | Combined accuracy/performance report |
| `/root/slm_eval_harness/app.py` | Updated dashboard with 3-way comparison (Q4 vs Q8 vs BF16) |

## Evaluation Criteria
- Successful completion of all 3 benchmarks for the BF16 variant.
- Comparison dashboard showing the "Diminishing Returns" curve (Accuracy vs. RAM/Latency).
- Verification that BF16 provides the upper bound for accuracy in this series.
