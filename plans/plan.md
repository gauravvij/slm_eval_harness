# SLM Evaluation Harness - Final Design Plan

## Goal
Build a modular, extensible evaluation harness for Small Language Models (SLMs) that provides **actionable diagnostics** in **under 1 hour** using the "Core-3" benchmark suite (HumanEval, HellaSwag, Mini-BFCL).

## Philosophy: Depth Over Breadth

Instead of 25,000 shallow tests across many benchmarks, we run **1,264 focused tests** with rich error analysis:
- **HumanEval** (164 tests): Code generation capability
- **HellaSwag** (1,000 tests): Commonsense reasoning
- **Mini-BFCL** (100 tests): Function calling / structured output

**Total runtime: ~50 minutes** (not 10+ hours)

**Why these three:**
1. **HumanEval**: Essential for developers - can it write correct code?
2. **HellaSwag**: Universal reasoning - commonsense inference without math
3. **Mini-BFCL**: Agentic capability - structured output and tool use

**Note on GSM8K**: Available as optional task (`tasks/reasoning/gsm8k.yaml`) for users who need math evaluation.

**Note on Multi-Modal**: Excluded from Core-3 due to complexity (vision encoders, architecture differences). Can be added as Tier 3 extension later.

---

## Research Summary

### Key Findings from Existing Frameworks:

**EleutherAI lm-eval-harness:**
- Uses YAML-based task definitions with Jinja2 prompt templating
- Model-agnostic interface supporting HF Transformers, vLLM, GGUF, APIs
- Config-based task creation for reproducibility

**OpenCompass:**
- Modular design with clear separation between models, datasets, and evaluators
- Supports diverse evaluation paradigms (zero-shot, few-shot, chain-of-thought)

**Common Patterns:**
- Task = Dataset + Prompt Template + Metric + Parser
- Model Adapter = Input Formatter + Output Parser + Special Token Handler
- Evaluation = Load Task → Format Prompts → Run Inference → Parse Outputs → Score → Analyze

---

## Gap Analysis Summary

### Critical Issues Addressed

| Issue | Risk Level | Mitigation Strategy |
|-------|------------|---------------------|
| Chat Template Complexity | HIGH | Auto-detection from tokenizer + manual override + validation |
| Parsing Edge Cases | HIGH | Priority-based extraction + refusal detection + confidence scoring |
| Code Execution Safety | CRITICAL | Subprocess with timeout + resource limits + restricted environment |
| Dataset Format Variability | MEDIUM | Per-task dataset loaders with schema validation |
| Configuration Validation | MEDIUM | JSON Schema validation + dry-run mode + config linting |
| Checkpointing & Resume | LOW | Periodic checkpointing + atomic writes + resume validation |

---

## Core Architectural Principles

### 1. Separation of Concerns
```
Task (Static)          Model Adapter (Dynamic)        Evaluation Engine
-----------------      -----------------------        -----------------
- Dataset              - Input formatting             - Orchestration
- Prompt template      - Chat template application    - Batching
- Expected format      - Output parsing               - Result aggregation
- Metric definition    - Answer extraction            - Reporting
```

### 2. Robust Parsing Strategy (Multi-Stage with Confidence)
```
Raw Output → Stage 1: Pattern Match → Stage 2: JSON Extraction → Stage 3: Raw Fallback
                ↓                        ↓                           ↓
           Exact match              Regex/JSON parsing            Last resort
           (confidence=1.0)        (confidence=0.9)              (confidence=0.5)
```

### 3. Chat Template Handling
```python
# Auto-detection hierarchy:
1. tokenizer.chat_template (from HF config)
2. Model family detection (llama, qwen, gemma, etc.)
3. Manual template from config
4. Base model fallback (no template)

# Validation:
- Generate test prompt
- Verify model produces coherent output
- Warn if template mismatch detected
```

### 4. Two-Tier Evaluation Approach

**Tier 1: Smoke Test (10 minutes, 60 tests)**
- HumanEval-Mini: 20 tests
- HellaSwag-Mini: 20 tests
- BFCL-Tiny: 20 tests
- **Purpose**: Validate harness works, catch major issues

**Tier 2: Core-3 Suite (50 minutes, 1,264 tests)**
- HumanEval: 164 tests (pass@1, pass@10)
- HellaSwag: 1,000 tests (subset of full 10K)
- Mini-BFCL: 100 tests (function calling)
- **Purpose**: Full diagnostic analysis with actionable insights

---

## The Core-3 Benchmark Suite (Final)

### ⚠️ IMPORTANT: Core-3 Definition

**Core-3 = HumanEval + HellaSwag + Mini-BFCL ONLY**

**NOT included in Core-3:**
- ❌ GSM8K (math reasoning) - available as OPTIONAL task only
- ❌ Multi-modal evaluation - excluded entirely

### Task Overview

| Task | Samples | Inference Calls | Time | Key Insight |
|------|---------|-----------------|------|-------------|
| **HumanEval** | 164 | 1,640 (pass@10) | ~30 min | Can it write correct Python? |
| **HellaSwag** | 1,000 | 1,000 (greedy) | ~15 min | Can it reason with commonsense? |
| **Mini-BFCL** | 100 | 100 (greedy) | ~5 min | Can it produce valid JSON/tool calls? |
| **Total** | **1,264** | **~2,740** | **~50 min** | Complete capability assessment |

### What Each Task Tests

**1. HumanEval (Coding)**
- Syntactically correct Python generation
- Understanding function signatures and docstrings
- Edge case handling (empty inputs, boundaries)
- **Metrics**: pass@1 (first-try quality), pass@10 (with retries)

**2. HellaSwag (Commonsense Reasoning)**
- Commonsense inference from context
- Plausible scenario completion
- No math required - pure reasoning
- **Metrics**: Accuracy (multiple choice A/B/C/D)

**Example:**
```
Context: "Then, the man writes over the snow covering the window of a car, 
and a woman wearing winter clothes smiles. Then"

Options:
A) The man adds snow to his car.
B) The man clears the snow from the car.
C) The woman gets in the car.
D) The woman cleans the window.

Correct: B (commonsense: writing on snow = clearing it)
```

**3. Mini-BFCL (Function Calling)**
- Valid JSON production
- Tool schema understanding
- Natural language to structured calls
- **Metrics**: Tool match rate, JSON validity

### Diagnostic Output Example

```
╔══════════════════════════════════════════════════════════════╗
║           SLM Evaluation Report: Core-3 Suite                ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  OVERALL SCORE: 68/100                                       ║
║                                                              ║
║  ┌─────────────────────────────────────────────────────────┐ ║
║  │ CODE (HumanEval)          ████████████░░░░  52%         │ ║
║  │   - pass@1: 35%  (first try quality)                    │ ║
║  │   - pass@10: 68% (with retries)                         │ ║
║  │   - ⚠️ 8 syntax errors, 15 runtime crashes              │ ║
║  │   - 💡 Suggestion: Add type hints in prompts            │ ║
║  └─────────────────────────────────────────────────────────┘ ║
║                                                              ║
║  ┌─────────────────────────────────────────────────────────┐ ║
║  │ REASONING (HellaSwag)    ██████████████░░  71%         │ ║
║  │   - Physical scenarios: Strong (82%)                    │ ║
║  │   - Social contexts: Moderate (68%)                     │ ║
║  │   - Abstract reasoning: Weak (58%) ⚠️                   │ ║
║  │   - 💡 Suggestion: Improve on abstract contexts           │ ║
║  └─────────────────────────────────────────────────────────┘ ║
║                                                              ║
║  ┌─────────────────────────────────────────────────────────┐ ║
║  │ FUNCTION CALL (Mini-BFCL) ███████████░░░░░  81%         │ ║
║  │   - JSON validity: Good (94%)                           │ ║
║  │   - Schema compliance: Moderate (81%)                   │ ║
║  │   - Nested args: Weak (54%) ⚠️                          │ ║
║  │   - 💡 Suggestion: Add schema in context examples       │ ║
║  └─────────────────────────────────────────────────────────┘ ║
║                                                              ║
║  RECOMMENDATION: Strong coding and reasoning. Ready for     ║
║  production use in code assistants. Work on abstract        ║
║  reasoning and nested function arguments.                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Approach

### Architecture Overview
```
slm_eval_harness/
├── core/
│   ├── base.py              # Abstract base classes
│   ├── task.py              # Task definition and loading
│   ├── evaluator.py         # Evaluation orchestration
│   ├── registry.py          # Task and adapter registry
│   ├── config_validator.py  # YAML validation with JSON Schema
│   └── checkpoint.py        # Save/resume functionality
├── adapters/
│   ├── base_adapter.py      # Abstract model adapter
│   ├── hf_adapter.py        # HuggingFace Transformers with chat template
│   ├── chat_templates.py    # Template registry per model family
│   ├── vllm_adapter.py      # vLLM backend
│   ├── api_adapter.py       # OpenAI/Anthropic APIs
│   └── gguf_adapter.py      # llama.cpp/GGUF
├── datasets/
│   ├── base_loader.py       # Abstract dataset loader
│   ├── humaneval_loader.py  # HumanEval-specific schema
│   ├── hellaswag_loader.py  # HellaSwag-specific schema
│   ├── gsm8k_loader.py      # GSM8K-specific schema (optional)
│   └── bfcl_loader.py       # BFCL-specific schema
├── tasks/
│   ├── coding/
│   │   ├── humaneval.yaml       # Full HumanEval (164 samples)
│   │   └── humaneval_mini.yaml  # Smoke test (20 samples)
│   ├── reasoning/
│   │   ├── hellaswag.yaml       # Full HellaSwag subset (1000 samples)
│   │   ├── hellaswag_mini.yaml  # Smoke test (20 samples)
│   │   └── gsm8k.yaml           # Optional: GSM8K (1319 samples)
│   └── function_call/
│       ├── bfcl_mini.yaml       # Mini-BFCL (100 samples)
│       └── bfcl_tiny.yaml       # Smoke test (20 samples)
├── parsers/
│   ├── base_parser.py       # Parser interface with confidence
│   ├── code_parser.py       # Code extraction with priority logic
│   ├── json_parser.py       # JSON extraction with error tolerance
│   ├── mc_parser.py         # Multiple choice extraction (A/B/C/D)
│   ├── refusal_detector.py  # Detect model refusals
│   └── robust_parser.py     # Multi-stage orchestrator
├── execution/
│   ├── sandbox.py           # Sandboxed code execution
│   ├── resource_limits.py   # CPU/memory/time limits
│   └── safety.py            # Import restrictions
├── metrics/
│   ├── pass_at_k.py         # Code correctness with correct formula
│   ├── exact_match.py       # String matching
│   ├── multiple_choice.py   # MC accuracy
│   └── diagnostic.py        # Error pattern analysis
├── reporting/
│   ├── console.py           # Rich terminal output
│   ├── json_export.py       # Machine-readable results
│   └── dashboard.py         # ASCII dashboard
├── tests/
│   ├── test_parsers.py      # Parser unit tests
│   ├── test_sandbox.py      # Safety tests
│   └── test_validation.py   # Config validation tests
├── cli.py                   # Command-line interface
└── README.md                # Documentation
```

### Key Design Decisions

**1. Task Definition (YAML with Validation)**
```yaml
# tasks/reasoning/hellaswag.yaml
schema_version: "1.0"
task_name: hellaswag
category: reasoning
dataset:
  loader: hellaswag
  path: Rowan/hellaswag
  split: validation
  subset:
    strategy: first_n
    count: 1000  # Subset of 10K for Core-3
  schema:
    required_fields: [ctx, endings, label]
    
prompt_template: |
  Complete the following scenario:
  
  {{ ctx }}
  
  Options:
  A) {{ endings[0] }}
  B) {{ endings[1] }}
  C) {{ endings[2] }}
  D) {{ endings[3] }}
  
  Answer with the letter (A, B, C, or D):

expected_output:
  type: multiple_choice
  options: ["A", "B", "C", "D"]
  extraction:
    priority: first_match
    patterns:
      primary: "^([A-D])"  # Single letter at start
      secondary: "[answer|option|choice].*?([A-D])"
  refusal_patterns:
    - "I cannot"
    - "I'm not sure"

metrics:
  - exact_match:
      normalize: true  # Uppercase, strip whitespace
      
generation:
  max_tokens: 10  # Short - just need the letter
  temperature: 0.0  # Greedy for reproducibility
```

**2. Smoke Test Configuration**
```yaml
# tasks/reasoning/hellaswag_mini.yaml
task_name: hellaswag_mini
category: reasoning
dataset:
  loader: hellaswag
  path: Rowan/hellaswag
  split: validation
  subset:
    strategy: first_n
    count: 20

# Same metrics as full, just fewer samples
```

**3. Configuration Validation (JSON Schema)**
```python
class ConfigValidator:
    SCHEMA = {
        "type": "object",
        "required": ["task_name", "category", "dataset", "expected_output"],
        "properties": {
            "task_name": {"type": "string"},
            "category": {"enum": ["coding", "reasoning", "function_call"]},
            "dataset": {
                "type": "object",
                "required": ["loader", "path"],
                "properties": {
                    "loader": {"type": "string"},
                    "path": {"type": "string"},
                    "split": {"type": "string"},
                    "subset": {
                        "type": "object",
                        "properties": {
                            "strategy": {"enum": ["first_n", "random", "specific_ids"]},
                            "count": {"type": "integer"},
                            "specific_ids": {"type": "array", "items": {"type": "integer"}}
                        }
                    }
                }
            }
        }
    }
    
    def validate(self, config_path: str) -> ValidationResult:
        # Returns detailed errors with line numbers
```

**4. Model Adapter with Chat Template Detection**
```python
class HuggingFaceAdapter(ModelAdapter):
    def __init__(self, model_path: str, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.chat_template = self._detect_chat_template()
        
    def _detect_chat_template(self) -> str:
        # Hierarchy: tokenizer config -> model family -> manual -> none
        if self.tokenizer.chat_template:
            return self.tokenizer.chat_template
        
        model_family = self._detect_model_family()
        if model_family in CHAT_TEMPLATES:
            return CHAT_TEMPLATES[model_family]
        
        return None  # Base model
    
    def format_prompt(self, task: Task, example: dict) -> str:
        prompt = task.render_prompt(example)
        if self.chat_template:
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt
```

**5. Robust Parser with Confidence**
```python
@dataclass
class ParsedOutput:
    value: Any
    confidence: float  # 0.0-1.0
    method: str        # "pattern", "json", "mc", "raw", "refusal"
    raw_output: str    # Original for debugging

class RobustParser:
    def parse(self, raw_output: str, task: Task) -> ParsedOutput:
        # Stage 1: Check for refusal
        if self._is_refusal(raw_output, task.refusal_patterns):
            return ParsedOutput(None, 0.0, "refusal", raw_output)
        
        # Stage 2: Task-specific parsing
        if task.expected_output.type == "multiple_choice":
            match = self._extract_multiple_choice(raw_output, task.expected_output.options)
            if match:
                return ParsedOutput(match, 1.0, "mc", raw_output)
        
        elif task.expected_output.type == "code":
            match = self._extract_code(raw_output, task.extraction.patterns)
            if match:
                return ParsedOutput(match, 1.0, "pattern", raw_output)
        
        elif task.expected_output.type == "json":
            json_result = self._try_json_extraction(raw_output)
            if json_result:
                return ParsedOutput(json_result, 0.9, "json", raw_output)
        
        # Stage 3: Raw fallback
        return ParsedOutput(raw_output.strip(), 0.5, "raw", raw_output)
```

**6. Sandboxed Code Execution**
```python
class CodeSandbox:
    """Secure code execution with resource limits"""
    
    def execute(self, code: str, test_code: str, timeout: float = 2.0) -> ExecutionResult:
        result = subprocess.run(
            [sys.executable, "-c", self._wrap_code(code, test_code)],
            timeout=timeout,
            capture_output=True,
            text=True,
            preexec_fn=self._set_resource_limits,
            env=self._restricted_env()
        )
        return ExecutionResult(
            success=result.returncode == 0,
            output=result.stdout,
            error=result.stderr
        )
    
    def _set_resource_limits(self):
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (self.timeout, self.timeout))
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
```

**7. Diagnostic Analysis**
```python
class DiagnosticAnalyzer:
    def analyze_humaneval(self, results: list) -> dict:
        return {
            "syntax_errors": sum(1 for r in results if r.error_type == "syntax"),
            "runtime_errors": sum(1 for r in results if r.error_type == "runtime"),
            "wrong_logic": sum(1 for r in results if r.error_type == "logic"),
            "common_errors": self._extract_error_patterns(results),
            "hardest_problems": self._find_hard_problems(results),
            "easiest_problems": self._find_easy_problems(results)
        }
    
    def analyze_hellaswag(self, results: list) -> dict:
        return {
            "by_category": self._breakdown_by_category(results),
            "physical_scenarios": self._score_physical(results),
            "social_contexts": self._score_social(results),
            "abstract_reasoning": self._score_abstract(results)
        }
```

---

## Subtasks (Final Implementation Plan)

### Phase 1: Foundation & Validation (Critical Path)

1. **Create project structure and base classes**
   - Set up directory structure
   - Implement `Task`, `ModelAdapter`, `Parser`, `DatasetLoader` ABCs
   - Create registry system for components
   - Expected output: `core/base.py`, `core/registry.py`, `datasets/base_loader.py`

2. **Implement configuration validation system**
   - JSON Schema for task YAML validation
   - Dry-run mode (validate without executing)
   - Detailed error messages with line numbers
   - Config linting CLI command
   - Expected output: `core/config_validator.py`, schemas/

3. **Build chat template detection and validation**
   - Auto-detect from tokenizer.chat_template
   - Model family detection (llama, qwen, gemma, etc.)
   - Template registry with known formats
   - Validation: test generation and coherence check
   - Expected output: `adapters/chat_templates.py`, template tests

4. **Implement dataset loader abstraction**
   - Abstract base class for dataset loaders
   - Schema validation per dataset
   - HumanEval loader with field mapping
   - HellaSwag loader with MC options
   - BFCL loader with JSON schema handling
   - Expected output: `datasets/` module

### Phase 2: Safety & Robustness (Critical for Code Tasks)

5. **Implement sandboxed code execution**
   - Subprocess-based execution with timeout
   - Resource limits (CPU, memory, file size)
   - Restricted Python environment (import whitelist)
   - Security: no network, no file writes outside temp
   - Expected output: `execution/sandbox.py`, `execution/safety.py`

6. **Build robust parser framework**
   - Parser interface with confidence scoring
   - Priority-based extraction (first vs last block)
   - Refusal detection patterns
   - Code block extraction with language detection
   - JSON extraction with error tolerance
   - Multiple choice extraction (A/B/C/D)
   - Expected output: `parsers/` module with full test coverage

7. **Implement checkpointing system**
   - Periodic checkpointing every N samples
   - Atomic writes (temp file + rename)
   - Config hash validation on resume
   - Resume CLI flag support
   - Expected output: `core/checkpoint.py`

### Phase 3: Evaluation Engine

8. **Implement HuggingFace model adapter**
   - Full chat template integration
   - Token counting and context management
   - Batch inference support
   - Generation parameter handling
   - Expected output: `adapters/hf_adapter.py`

9. **Implement metric calculators**
   - pass@k with correct combinatorial formula
   - Exact match with normalization options
   - Multiple choice accuracy
   - Diagnostic error pattern analysis
   - Expected output: `metrics/` module

10. **Build evaluation orchestrator**
    - Task loading and validation
    - Model adapter initialization
    - Batch processing with progress tracking
    - Result aggregation and reporting
    - Expected output: `core/evaluator.py`

### Phase 4: Core-3 Tasks & CLI

11. **Implement HumanEval tasks**
    - Full HumanEval config (164 samples)
    - HumanEval-Mini config (20 samples) for smoke test
    - Sandboxed execution integration
    - pass@1 and pass@10 metrics
    - Expected output: `tasks/coding/humaneval.yaml`, `tasks/coding/humaneval_mini.yaml`

12. **Implement HellaSwag tasks**
    - Full HellaSwag config (1,000 sample subset)
    - HellaSwag-Mini config (20 samples) for smoke test
    - Multiple choice extraction
    - Per-category diagnostics
    - Expected output: `tasks/reasoning/hellaswag.yaml`, `tasks/reasoning/hellaswag_mini.yaml`

13. **Implement Mini-BFCL task**
    - BFCL-Mini config (100 samples)
    - BFCL-Tiny config (20 samples) for smoke test
    - JSON schema validation
    - Tool call extraction and matching
    - Expected output: `tasks/function_call/bfcl_mini.yaml`, `tasks/function_call/bfcl_tiny.yaml`

14. **Build CLI interface**
    - Task listing and discovery
    - Model configuration (path, backend, params)
    - Smoke test mode (`--smoke-test`)
    - Full suite mode (default)
    - Evaluation runner with progress bar
    - Result export (JSON, console dashboard)
    - Checkpoint resume support
    - Expected output: `cli.py`, config system

15. **Build reporting system**
    - Rich console output with progress bars
    - ASCII dashboard with visual indicators
    - Error pattern analysis
    - Actionable suggestions
    - Expected output: `reporting/` module

### Phase 5: Optional Tasks & Additional Adapters

**⚠️ NOTE: GSM8K is OPTIONAL - NOT part of Core-3**

16. **Add GSM8K as OPTIONAL task (NOT Core-3)**
    - Full GSM8K config (1,319 samples) - for users who need math evaluation
    - GSM8K-Mini config (50 samples)
    - Chain-of-thought extraction
    - Numeric answer parsing
    - Expected output: `tasks/reasoning/gsm8k.yaml`, `tasks/reasoning/gsm8k_mini.yaml`
    - **Usage**: `--tasks humaneval,hellaswag,gsm8k,bfcl_mini` (NOT default)

17. **Add additional model adapters**
    - vLLM adapter for fast inference
    - API adapter (OpenAI-compatible)
    - GGUF/llama.cpp adapter
    - Expected output: `adapters/vllm_adapter.py`, `adapters/api_adapter.py`, `adapters/gguf_adapter.py`

### Phase 6: Testing & Documentation

18. **Create comprehensive test suite**
    - Unit tests for parsers (edge cases, refusals)
    - Unit tests for sandbox (safety, timeouts)
    - Unit tests for validation (schema errors)
    - Integration tests (end-to-end on smoke test)
    - Expected output: `tests/` directory

19. **Create documentation**
    - README with architecture overview
    - Usage examples for smoke test and full suite
    - Task creation guide
    - Troubleshooting guide (common errors)
    - Expected output: `README.md`, `examples/`, `docs/`

---

## Deliverables

| File Path | Description |
|-----------|-------------|
| `/home/azureuser/slm_eval_harness/core/base.py` | Abstract base classes |
| `/home/azureuser/slm_eval_harness/core/config_validator.py` | YAML validation with JSON Schema |
| `/home/azureuser/slm_eval_harness/core/checkpoint.py` | Save/resume functionality |
| `/home/azureuser/slm_eval_harness/core/task.py` | Task loading and management |
| `/home/azureuser/slm_eval_harness/core/evaluator.py` | Evaluation orchestration |
| `/home/azureuser/slm_eval_harness/core/registry.py` | Component registry |
| `/home/azureuser/slm_eval_harness/adapters/base_adapter.py` | Adapter interface |
| `/home/azureuser/slm_eval_harness/adapters/hf_adapter.py` | HuggingFace with chat template |
| `/home/azureuser/slm_eval_harness/adapters/chat_templates.py` | Template registry |
| `/home/azureuser/slm_eval_harness/adapters/vllm_adapter.py` | vLLM backend |
| `/home/azureuser/slm_eval_harness/adapters/api_adapter.py` | OpenAI/Anthropic APIs |
| `/home/azureuser/slm_eval_harness/datasets/base_loader.py` | Dataset loader interface |
| `/home/azureuser/slm_eval_harness/datasets/humaneval_loader.py` | HumanEval loader |
| `/home/azureuser/slm_eval_harness/datasets/hellaswag_loader.py` | HellaSwag loader |
| `/home/azureuser/slm_eval_harness/datasets/bfcl_loader.py` | BFCL loader |
| `/home/azureuser/slm_eval_harness/parsers/base_parser.py` | Parser interface with confidence |
| `/home/azureuser/slm_eval_harness/parsers/code_parser.py` | Code extraction with priority |
| `/home/azureuser/slm_eval_harness/parsers/json_parser.py` | JSON extraction |
| `/home/azureuser/slm_eval_harness/parsers/mc_parser.py` | Multiple choice extraction |
| `/home/azureuser/slm_eval_harness/parsers/robust_parser.py` | Multi-stage orchestrator |
| `/home/azureuser/slm_eval_harness/parsers/refusal_detector.py` | Refusal pattern detection |
| `/home/azureuser/slm_eval_harness/execution/sandbox.py` | Sandboxed code execution |
| `/home/azureuser/slm_eval_harness/execution/safety.py` | Import restrictions |
| `/home/azureuser/slm_eval_harness/metrics/pass_at_k.py` | Code correctness |
| `/home/azureuser/slm_eval_harness/metrics/exact_match.py` | String matching |
| `/home/azureuser/slm_eval_harness/metrics/multiple_choice.py` | MC accuracy |
| `/home/azureuser/slm_eval_harness/metrics/diagnostic.py` | Error pattern analysis |
| `/home/azureuser/slm_eval_harness/reporting/console.py` | Rich terminal output |
| `/home/azureuser/slm_eval_harness/reporting/dashboard.py` | ASCII dashboard |
| `/home/azureuser/slm_eval_harness/tasks/coding/humaneval.yaml` | Full HumanEval config |
| `/home/azureuser/slm_eval_harness/tasks/coding/humaneval_mini.yaml` | Smoke test config |
| `/home/azureuser/slm_eval_harness/tasks/reasoning/hellaswag.yaml` | Full HellaSwag config |
| `/home/azureuser/slm_eval_harness/tasks/reasoning/hellaswag_mini.yaml` | Smoke test config |
| `/home/azureuser/slm_eval_harness/tasks/reasoning/gsm8k.yaml` | **OPTIONAL** - NOT Core-3 |
| `/home/azureuser/slm_eval_harness/tasks/function_call/bfcl_mini.yaml` | Mini-BFCL config |
| `/home/azureuser/slm_eval_harness/tasks/function_call/bfcl_tiny.yaml` | Smoke test config |
| `/home/azureuser/slm_eval_harness/cli.py` | Command-line interface |
| `/home/azureuser/slm_eval_harness/tests/test_parsers.py` | Parser unit tests |
| `/home/azureuser/slm_eval_harness/tests/test_sandbox.py` | Safety tests |
| `/home/azureuser/slm_eval_harness/tests/test_validation.py` | Config validation tests |
| `/home/azureuser/slm_eval_harness/README.md` | Documentation |

---

## Evaluation Criteria (Final)

- [ ] **Architecture**: Clean separation between tasks, adapters, and parsers
- [ ] **Extensibility**: New tasks can be added via YAML without code changes
- [ ] **Robustness**: Parser handles 90%+ of outputs without raw fallback
- [ ] **Safety**: Code execution is fully sandboxed (no FS/network access)
- [ ] **Correctness**: Evaluation results match established benchmarks
- [ ] **Validation**: Configuration validation catches 95%+ of config errors before runtime
- [ ] **Resilience**: Evaluation can resume from checkpoint without data loss
- [ ] **Correctness**: Chat template auto-detection works for top 10 HF model families
- [ ] **Usability**: Simple CLI for smoke test (`--smoke-test`) and full suite
- [ ] **Actionable**: Reports provide clear diagnostics and suggestions
- [ ] **Speed**: Full Core-3 suite completes in under 1 hour on single GPU
- [ ] **Test Coverage**: Unit tests for parsers, sandbox, and validation

---

## Usage Examples (Final)

### Smoke Test (10 minutes)
```bash
# Quick validation - 60 tests
python -m slm_eval_harness \
  --model hf \
  --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct \
  --smoke-test \
  --output results_smoke.json
```

### Full Core-3 Suite (50 minutes)
```bash
# Complete evaluation - 1,264 tests
python -m slm_eval_harness \
  --model hf \
  --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct \
  --tasks humaneval,hellaswag,bfcl_mini \
  --output results_full.json \
  --checkpoint checkpoints/
```

### Optional: Include GSM8K (NOT Core-3)
```bash
# Core-3 + OPTIONAL math reasoning - 2,583 tests, ~75 min
python -m slm_eval_harness \
  --model hf \
  --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct \
  --tasks humaneval,hellaswag,gsm8k,bfcl_mini \
  --output results_with_math.json
```

### Resume from Checkpoint
```bash
# Continue interrupted evaluation
python -m slm_eval_harness \
  --model hf \
  --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct \
  --tasks humaneval,hellaswag,bfcl_mini \
  --resume checkpoints/last_checkpoint.json
```

### Custom Task Configuration
```bash
# Use custom task YAML
python -m slm_eval_harness \
  --model hf \
  --model_args pretrained=my-model \
  --task_config my_custom_task.yaml \
  --output results.json
```

---

## Key Design Patterns

### 1. Adapter Pattern for Models
Each model family has specific input/output formatting requirements. The adapter pattern encapsulates these differences behind a common interface.

### 2. Strategy Pattern for Parsing
Different tasks require different parsing strategies. The strategy pattern allows flexible parsing with fallback chains and confidence scoring.

### 3. Template Method for Evaluation
The evaluation workflow is standardized, but specific steps (parsing, metric calculation) are customizable per task.

### 4. Registry Pattern for Extensibility
Tasks, adapters, and dataset loaders are registered dynamically, allowing plugins and custom components.

### 5. Factory Pattern for Dataset Loaders
Per-task dataset loaders handle schema variations without polluting the core task logic.

### 6. Two-Tier Evaluation Pattern
Smoke test for quick validation, full suite for comprehensive analysis.

---

## Risk Mitigations Summary

| Risk | Mitigation | Implementation |
|------|------------|----------------|
| Wrong chat template | Auto-detect + validate + manual override | `adapters/chat_templates.py` |
| Parser extracts wrong content | Priority-based + refusal detection + confidence | `parsers/robust_parser.py` |
| Malicious code execution | Subprocess + resource limits + restricted env | `execution/sandbox.py` |
| Config errors mid-run | JSON Schema validation + dry-run mode | `core/config_validator.py` |
| Crash loses progress | Periodic checkpointing + atomic writes | `core/checkpoint.py` |
| Dataset format mismatch | Per-task loaders with schema validation | `datasets/` module |
| Context length exceeded | Token counting + auto-truncation | `utils/token_counter.py` |
| Evaluation takes too long | Focused Core-3 suite (~1 hour) | Task selection + smoke test mode |

---

## Notes

- **Safety First**: Sandboxed execution implemented before any code evaluation tasks
- **Reproducibility**: All random seeds configurable, deterministic sampling, checkpoint resume
- **Efficiency**: Support batching, caching, and streaming for large evaluations
- **Debugging**: Rich logging of parsing decisions, chat template detection, and execution
- **Standards**: Follow patterns from lm-eval-harness where applicable
- **Testing**: Comprehensive test suite before considering implementation complete
- **Focus**: Core-3 suite provides actionable insights without 10+ hour runtimes
- **Flexibility**: GSM8K available as optional task for users needing math evaluation
- **Future**: Multi-modal support can be added as Tier 3 extension

---

## Final Checklist

**Core-3 Suite:**
- [x] HumanEval (code generation) - 164 tests, ~30 min
- [x] HellaSwag (commonsense reasoning) - 1,000 tests, ~15 min
- [x] Mini-BFCL (function calling) - 100 tests, ~5 min
- [x] Total: 1,264 tests, ~50 minutes

**Optional:**
- [ ] GSM8K (math reasoning) - available but not in Core-3
- [ ] Multi-modal - future extension

**Key Features:**
- [x] Smoke test mode (10 min)
- [x] Full suite mode (50 min)
- [x] Checkpoint/resume
- [x] Rich diagnostic reporting
- [x] Sandboxed code execution
- [x] Robust multi-stage parsing
- [x] Chat template auto-detection
- [x] Configuration validation
