# BFCL Complete Evaluation + Model Comparison Plan

## Goal
Run complete BFCL evaluation on gemma4:e4b and qwen3.5:9b, plus full HumanEval and HellaSwag on qwen3.5:9b for comprehensive model comparison.

## Research Summary
- **Available Models**: gemma4:e4b (9.6GB), qwen3.5:9b (6.6GB) both available via Ollama
- **BFCL Dataset**: 17+ subsets available (simple, multiple, parallel, exec_*, live_*, java, javascript, sql, etc.)
- **Current Status**: BFCL tiny (20 samples) completed with 95% accuracy on gemma4:e4b
- **HumanEval Full**: Already completed for gemma4:e4b (95.73% accuracy)
- **HellaSwag Full**: Already completed for gemma4:e4b (0% - model token limit)

## Approach
1. Create full BFCL task config (all subsets, no sample limit)
2. Run full BFCL on gemma4:e4b
3. Run full BFCL on qwen3.5:9b
4. Run full HumanEval on qwen3.5:9b
5. Run full HellaSwag on qwen3.5:9b
6. Generate comparison report

## Subtasks

### 1. Create Full BFCL Task Configuration
- Create tasks/function_call/bfcl_full.yaml
- Include all major subsets (simple, multiple, parallel, exec_simple, live_simple)
- Set num_samples: null (all samples)
- Estimated samples: 400+ in simple subset alone

### 2. Run Full BFCL - gemma4:e4b
- Execute: python cli.py eval bfcl_full --model gemma4:e4b
- Expected: 2-4 hours (400+ samples × ~15-30s each)
- Checkpoint: .checkpoints/bfcl_full_gemma4_e4b.json

### 3. Run Full BFCL - qwen3.5:9b
- Execute: python cli.py eval bfcl_full --model qwen3.5:9b
- Expected: 2-4 hours
- Checkpoint: .checkpoints/bfcl_full_qwen3.5_9b.json

### 4. Run Full HumanEval - qwen3.5:9b
- Execute: python cli.py eval humaneval_full --model qwen3.5:9b
- Expected: 2-3 hours (164 samples)
- Checkpoint: .checkpoints/humaneval_full_qwen3.5_9b.json

### 5. Run Full HellaSwag - qwen3.5:9b
- Execute: python cli.py eval hellaswag_full --model qwen3.5:9b
- Expected: 1-2 hours (1000 samples, faster per sample)
- Checkpoint: .checkpoints/hellaswag_full_qwen3.5_9b.json

### 6. Generate Comparison Report
- Create reports/model_comparison.json
- Compare gemma4:e4b vs qwen3.5:9b across all benchmarks
- Include per-task accuracy, pass rates, timing

## Deliverables

| File Path | Description |
|-----------|-------------|
| tasks/function_call/bfcl_full.yaml | Full BFCL task configuration |
| .checkpoints/bfcl_full_gemma4_e4b.json | BFCL results for gemma4:e4b |
| .checkpoints/bfcl_full_qwen3.5_9b.json | BFCL results for qwen3.5:9b |
| .checkpoints/humaneval_full_qwen3.5_9b.json | HumanEval results for qwen3.5:9b |
| .checkpoints/hellaswag_full_qwen3.5_9b.json | HellaSwag results for qwen3.5:9b |
| reports/model_comparison.json | Side-by-side model comparison |

## Evaluation Criteria
- BFCL full completes without errors for both models
- All checkpoints created successfully
- Comparison report generated with clear metrics
- qwen3.5:9b HellaSwag accuracy > 0% (testing if token limit is gemma-specific)

## Notes
- qwen3.5:9b is 6.6GB vs gemma4:e4b 9.6GB - may have different capabilities
- BFCL has 17+ subsets; we'll focus on the main ones (simple, multiple, parallel variants)
- Each model evaluation can run independently
- Total estimated time: 8-12 hours for complete suite
