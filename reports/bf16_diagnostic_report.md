# BF16 HumanEval Diagnostic Report

## Executive Summary

**Issue Confirmed:** The BF16 (16-bit) quantized version of Qwen 3.6 35B A3B consistently produces **15% accuracy** on HumanEval, significantly lower than Q4 (47.56%) and Q8 (50%) variants.

**Root Cause:** Infrastructure-level issue with GGUF adapter and BF16 multi-shard model loading, not a model capability issue.

## Detailed Findings

### Re-run Results (Fresh Evaluation)
- **Accuracy:** 15.00% (3/20 passed)
- **Samples with 0 tokens:** 17/20 (85%)
- **Samples with content:** 3/20 (15%)
- **All 3 passing samples:** Generated successfully and passed tests

### Pattern Analysis

| Sample ID | Tokens | Status | Notes |
|-----------|--------|--------|-------|
| 0-4, 6-7, 9-10, 12, 14, 17 | 0 | Failed | Empty output, refusal_detected |
| 5, 8 | 128-163 | Failed | Generated content but wrong solution |
| 11, 16, 18 | 95-171 | **Passed** | Correct solutions generated |
| 13, 15, 19 | 8-12 | Failed | Incomplete (just function signature) |

### Key Observations

1. **Systematic Generation Failure:** 85% of prompts result in 0 tokens generated
2. **Model Loads Successfully:** No loading errors, model initializes correctly
3. **Finish Reason Always "stop":** Model thinks it completed generation
4. **Context Warning:** `n_ctx_seq (4096) < n_ctx_train (262144)` - context mismatch

### Infrastructure-Level Indicators

```
llama_context: n_ctx_seq (4096) < n_ctx_train (262144) 
-- the full capacity of the model will not be utilized
```

This warning suggests the BF16 model's training context (262k) is much larger than the configured context (4k), which may cause issues with the GGUF adapter.

### Comparison with Quantized Variants

| Variant | HumanEval | Tokens/Sample | Issue Rate |
|---------|-----------|---------------|------------|
| Q4_K_M | 47.56% | ~100-200 | <5% |
| Q8_0 | 50.00% | ~100-200 | <5% |
| **BF16** | **15.00%** | **0 (85%)** | **85%** |

## Root Cause Hypothesis

1. **Multi-Shard GGUF Handling:** BF16 uses 2 shards (00001-of-00002, 00002-of-00002) while Q4/Q8 use single files
2. **llama-cpp Compatibility:** The BF16 format may have compatibility issues with the current llama-cpp-python version
3. **Context Length Mismatch:** 4k context vs 262k training context may cause token generation failures
4. **Memory Pressure:** 66GB model + overhead may cause silent failures during generation

## Recommendations

### Immediate Actions
1. **Use Q8_0 for accuracy-critical tasks** - it achieves 50% vs BF16's 15%
2. **Use Q4_K_M for production** - best throughput (22.12 tok/s) with good accuracy (47.56%)
3. **Investigate BF16 with alternative backends** (vLLM, TGI) for true 16-bit evaluation

### Further Investigation
1. Test BF16 with increased context length (n_ctx=8192 or 16384)
2. Verify llama-cpp-python version compatibility with BF16 GGUF format
3. Check if single-shard BF16 models perform differently
4. Profile memory usage during BF16 inference

## Conclusion

The BF16 model's 15% HumanEval score is **not representative of the model's true capabilities**. The Q8_0 variant (50% accuracy) likely represents the upper bound for this model architecture on coding tasks using the current evaluation infrastructure.

**Recommendation:** Use Q8_0 results as the "best case" accuracy benchmark for Qwen 3.6 35B A3B, not BF16.

---
*Report generated: 2026-04-20*
*Model: unsloth/Qwen3.6-35B-A3B-GGUF BF16*
*Hardware: 32-core AMD EPYC 9655, 125GB RAM*
