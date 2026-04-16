# Gap Analysis & Risk Assessment

## Critical Gaps Identified

### 1. Chat Template Complexity (HIGH RISK)

**Gap**: The plan assumes uniform chat template handling, but different model families have radically different formats.

**Specific Issues**:
- Llama-3: `<|begin_of_text|><|start_header_id|>user<|end_header_id|>...<|eot_id|>`
- Qwen: `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`
- Gemma: `<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n`
- Base models (no chat template): Raw prompt only
- Custom templates (fine-tuned models): Unknown format

**Inconsistency Risk**: If we apply the wrong template, model outputs become garbage, making parsing impossible.

**Mitigation Required**:
- Auto-detect tokenizer.chat_template from HF config
- Fallback to manual templates for GGUF/API models
- Validation: Generate a test prompt and verify model responds sensibly
- Support override via config for custom models

---

### 2. Parsing Edge Cases (HIGH RISK)

**Gap**: Multi-stage parsing is good, but edge cases will break it.

**Specific Issues**:
```python
# Case 1: Multiple code blocks
"""Here's the solution:
```python
def helper():
    pass
```

And the main function:
```python
def solve():
    helper()
```
"""

# Case 2: Explanation + code
"""I need to solve this step by step.
First, let me analyze the problem...
[500 words of explanation]
```python
def solution():
    pass
```
"""

# Case 3: Refusal/uncertainty
"""I'm not sure how to solve this problem. 
It seems to require domain knowledge I don't have.
Here's my best attempt: ..."""

# Case 4: Hallucinated tools (function calling)
"""I'll use the `calculate_sum` tool.
<tool>nonexistent_tool</tool>"""

**Inconsistency Risk**: Parser extracts wrong code block → execution fails → false negative.

**Mitigation Required**:
- Priority-based extraction (last code block for solutions)
- Refusal detection patterns ("I cannot", "I'm not sure", etc.)
- Confidence scoring per parse attempt
- Debug logging of all parsing attempts

---

### 3. Code Execution Safety (CRITICAL)

**Gap**: "Sandboxed execution" is mentioned but not specified.

**Specific Issues**:
```python
# Infinite loop
while True:
    pass

# Resource exhaustion
import os
os.fork()  # Fork bomb

# Malicious code
import shutil
shutil.rmtree("/")  # If not containerized

# Network access
import requests
requests.get("http://malicious.com")

# File system access
open("/etc/passwd").read()
```

**Inconsistency Risk**: One malicious sample can crash the harness or compromise the system.

**Mitigation Required**:
- Subprocess with timeout (2-5 seconds max)
- Resource limits (RLIMIT_CPU, RLIMIT_AS)
- Restricted Python environment (no imports except stdlib + whitelist)
- Container/isolated execution for production
- Separate process per test case (prevents cross-contamination)

---

### 4. Dataset Format Variability (MEDIUM RISK)

**Gap**: Assumes consistent dataset schemas across benchmarks.

**Specific Issues**:

HumanEval:
```json
{
  "task_id": "HumanEval/0",
  "prompt": "def has_close_elements(numbers:list, threshold: float):\n    ...",
  "entry_point": "has_close_elements",
  "canonical_solution": "...",
  "test": "def check(candidate): ..."
}
```

GSM8K:
```json
{
  "question": "Janet buys...",
  "answer": "Janet buys 5 * 3 = 15...\n#### 15"
}
```

MBPP (sanitized vs original):
- Sanitized: `{
  "text": "Write a function...", 
  "code": "def solution():...",
  "test_list": [...]
}`
- Original: Different field names, includes test setup code

**Inconsistency Risk**: Dataset loader fails → task cannot run.

**Mitigation Required**:
- Per-task dataset loader classes
- Schema validation on load
- Clear error messages for missing fields
- Support for dataset variants (sanitized/unsanitized)

---

### 5. Few-Shot Prompting Complexity (MEDIUM RISK)

**Gap**: Plan mentions few-shot but not implementation details.

**Specific Issues**:
- Context length: Few-shot examples + prompt may exceed model's context
- Example selection: Random vs fixed vs diverse selection affects reproducibility
- Formatting: Different models expect few-shot in different formats
- Answer inclusion: Some tasks include answer in prompt, others don't

**Inconsistency Risk**: Different few-shot configurations = incomparable results.

**Mitigation Required**:
- Token counting before generation
- Automatic truncation with warning
- Configurable example selection strategy
- Separate prompt template for few-shot vs zero-shot

---

### 6. Metric Calculation for pass@k (MEDIUM RISK)

**Gap**: pass@k metric requires multiple samples per problem.

**Specific Issues**:
- Temperature > 0 needed for diversity
- k=100 means 100 generations per problem × 164 problems = 16,400 calls
- Memory: Storing all outputs
- Aggregation: pass@k formula is non-trivial
  ```
  pass@k = 1 - C(n-c, k) / C(n, k)
  where n=total samples, c=correct samples
  ```

**Inconsistency Risk**: Wrong formula = wrong scores.

**Mitigation Required**:
- Streaming evaluation (don't store all in memory)
- Correct pass@k formula implementation
- Temperature control in generation config
- Checkpointing for resume capability

---

### 7. LLM-as-Judge Circular Dependency (MEDIUM RISK)

**Gap**: Using LLM-as-judge for parsing fallback creates issues.

**Specific Issues**:
- If we're evaluating SLMs, using a larger LLM as judge defeats the purpose
- API costs for large-scale evaluation
- Judge model bias (prefers outputs similar to its training)
- Consistency: Same input may get different judge scores

**Inconsistency Risk**: Evaluation becomes dependent on judge model quality.

**Mitigation Required**:
- Make LLM-as-judge optional (configurable)
- Support local judge models (smaller, faster)
- Judge calibration on subset of data
- Clear documentation of judge model used

---

### 8. Configuration Validation (MEDIUM RISK)

**Gap**: YAML configs can have errors that aren't caught until runtime.

**Specific Issues**:
- Missing required fields (dataset path, metric type)
- Invalid combinations (code task with exact_match metric)
- Typos in extraction patterns (invalid regex)
- Wrong dataset split names

**Inconsistency Risk**: Config error discovered mid-evaluation = wasted compute.

**Mitigation Required**:
- JSON Schema validation for YAML configs
- Dry-run mode (validate without executing)
- Config linting tool
- Clear error messages with line numbers

---

### 9. Checkpointing & Resume (LOW RISK but HIGH IMPACT)

**Gap**: Long evaluations need checkpointing.

**Specific Issues**:
- Evaluation of 1B+ model on full HumanEval + MBPP + GSM8K could take hours
- Crash mid-evaluation loses all progress
- Partial results need to be valid and resumable

**Inconsistency Risk**: Restarting from scratch = wasted time and money.

**Mitigation Required**:
- Periodic checkpointing (every N samples)
- Resume from checkpoint on restart
- Atomic writes (write to temp, then rename)
- Validation that checkpoint matches current config

---

### 10. Batch Size & Memory Management (LOW RISK)

**Gap**: Batch inference assumed but not specified.

**Specific Issues**:
- Different models have different optimal batch sizes
- Memory usage scales with batch_size × sequence_length
- Padding strategies affect results (left vs right pad)
- Attention mask handling

**Inconsistency Risk**: OOM errors on large batches.

**Mitigation Required**:
- Auto batch size detection (start high, reduce on OOM)
- Memory profiling utilities
- Clear documentation of memory requirements
- Gradient checkpointing for large models

---

## Implementation Order Adjustments

Based on this analysis, the original plan needs reordering:

### Revised Priority

**Phase 1: Foundation (REORDERED)**
1. Core base classes ✅
2. **NEW: Configuration validation system** (catch errors early)
3. **NEW: Chat template detection and validation**
4. YAML task loader ✅
5. **NEW: Dataset loader abstraction** (handle format variations)

**Phase 2: Safety & Robustness (NEW PHASE)**
6. Sandboxed code execution (CRITICAL - do before any code tasks)
7. Robust parser with edge case handling
8. Error handling and logging framework

**Phase 3: Evaluation Engine**
9. Model adapter system
10. Checkpointing and resume capability
11. Metric calculators (with correct pass@k formula)

**Phase 4: Tasks & CLI**
12. Task implementations
13. CLI interface
14. Additional adapters

---

## Additional Requirements Not in Original Plan

1. **Test Suite**: Unit tests for parsers, validators, and metrics
2. **Integration Tests**: End-to-end evaluation on small dataset samples
3. **Documentation**: Troubleshooting guide for common errors
4. **Benchmarking**: Performance profiling for batch sizes
5. **Security Audit**: Code execution sandbox review

---

## Success Criteria Revision

Add to evaluation criteria:
- [ ] Configuration validation catches 95%+ of config errors before runtime
- [ ] Parser handles 90%+ of outputs without falling back to raw
- [ ] Code execution is fully sandboxed (no FS/network access)
- [ ] Evaluation can resume from checkpoint without data loss
- [ ] Chat template auto-detection works for top 10 HF model families
