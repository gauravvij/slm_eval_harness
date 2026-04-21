# Qwen 3.6 35B A3B Q8_0 Evaluation Plan

## Goal
Evaluate the Q8_0 (8-bit) quantized version of Qwen 3.6 35B A3B to compare accuracy and inference performance against the Q4_K_M baseline.

## Research Summary
- **Model ID**: `unsloth/Qwen3.6-35B-A3B-GGUF`
- **Quantization File**: `Qwen3.6-35B-A3B-Q8_0.gguf`
- **Context**: The Q4_K_M version achieved ~22 tokens/sec. Q8_0 is expected to have higher accuracy but lower throughput and higher RAM usage (~37GB for weights).

## Approach
1. **Model Acquisition**: Download the Q8_0 GGUF file using `huggingface-cli`.
2. **Inference Benchmarking**: Measure TTFT and throughput on the 32-core EPYC hardware.
3. **Accuracy Evaluation**: Run HumanEval, HellaSwag (mini/full), and BFCL benchmarks.
4. **Comparative Analysis**: Update the dashboard and `model_comparison.json` to show Q4 vs Q8 trade-offs.

## Subtasks
1. Download `Qwen3.6-35B-A3B-Q8_0.gguf` from HF, expected output: `.gguf` file in project root.
2. Run inference performance test (TTFT, throughput), expected output: `reports/inference_metrics_q8.json`.
3. Run HumanEval benchmark, expected output: accuracy score in `reports/humaneval_q8.json`.
4. Run HellaSwag benchmark (checkpoint-enabled), expected output: accuracy score in `reports/hellaswag_q8.json`.
5. Run BFCL benchmark with `function_call_match` metric, expected output: accuracy score in `reports/bfcl_q8.json`.
6. Update `reports/model_comparison.json` and dashboard with Q8 results.

## Deliverables
| File Path | Description |
|-----------|-------------|
| `/root/slm_eval_harness/reports/inference_metrics_q8.json` | Performance data for Q8 variant |
| `/root/slm_eval_harness/reports/qwen3.6_35b_q8_summary.json` | Combined accuracy/performance report |
| `/root/slm_eval_harness/app.py` | Updated dashboard with Q4 vs Q8 toggle/comparison |

## Evaluation Criteria
- Successful completion of all 3 benchmarks for the Q8 variant.
- Comparison table generated showing Accuracy vs. Tokens/sec for Q4 vs Q8.
- RAM usage monitored to ensure it stays within the 125GB limit.
