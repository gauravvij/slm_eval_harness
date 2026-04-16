# SLM Evaluation Harness Architecture Plan

## Goal
Create a robust, extensible evaluation harness for Small Language Models (SLMs) that can:
1. Evaluate models on multiple tasks (coding/SWE, reasoning, function calling, agentic capabilities)
2. Support different model backends (local via Ollama, HuggingFace, OpenRouter, etc.)
3. Handle varying input/output schemas across model families (Gemma, Llama, Qwen, Phi, etc.)
4. Provide robust parsing with confidence scoring to avoid false negatives
5. Follow established patterns from EleutherAI lm-eval-harness

## Research Summary

### Key Findings

**1. EleutherAI lm-eval-harness Architecture:**
- Task configuration via YAML files with standardized schema
- Model adapters abstract different backends (HF, OpenAI, local)
- Task registry for dynamic task loading
- Metric computation separate from evaluation logic
- Checkpoint/resume support for long-running evaluations

**2. Chat Template Handling:**
- **Gemma**: Uses `<start_of_turn>user`, `<end_of_turn>`, `<start_of_turn>model` tokens
- **Llama/Qwen**: Uses ChatML format with `system`, `user`, `assistant` roles
- **Critical**: Each model family has unique formatting requirements
- **Solution**: Auto-detect model family and apply appropriate template

**3. Robust Parsing Best Practices:**
- Multi-stage fallback chain: strict → lenient → heuristic
- Confidence scoring for each extraction attempt
- Refusal detection (model declines to answer)
- Handle markdown fences, JSON arrays, trailing commas, single quotes
- Never fail silently - always return structured result with metadata

## Architecture Design

### Core Principles

1. **Separation of Concerns**: Tasks, adapters, parsers, metrics are independent
2. **Configuration-Driven**: Tasks defined in YAML, models in config files
3. **Extensibility**: New tasks/adapters/parsers via registration pattern
4. **Robustness**: Graceful degradation, confidence scoring, error recovery
5. **Reproducibility**: Checkpoints, deterministic sampling, versioned configs

### Directory Structure

```
slm_eval_harness/
├── core/                       # Core framework
│   ├── base.py                # Abstract base classes
│   ├── config_validator.py    # YAML schema validation
│   ├── registry.py            # Task/adapter/parser registration
│   ├── evaluator.py           # Main evaluation orchestrator
│   ├── checkpoint.py          # Resume capability
│   └── task_config.py         # TaskConfig dataclass
├── adapters/                   # Model backend adapters
│   ├── base_adapter.py        # Abstract ModelAdapter
│   ├── hf_adapter.py          # HuggingFace integration
│   ├── ollama_adapter.py      # Local Ollama models
│   ├── openrouter_adapter.py  # OpenRouter API
│   └── chat_templates.py      # Model-family templates
├── tasks/                      # Task definitions
│   ├── coding/                # Code generation tasks
│   │   ├── humaneval.yaml
│   │   └── mbpp.yaml
│   ├── reasoning/             # Reasoning tasks
│   │   ├── hellaswag.yaml
│   │   └── gsm8k.yaml
│   ├── function_call/         # Function calling
│   │   └── bfcl.yaml
│   └── agentic/               # Agent capabilities
│       └── webarena.yaml
├── dataset_loaders/            # Dataset loading
│   ├── base_loader.py
│   ├── humaneval_loader.py
│   ├── hellaswag_loader.py
│   └── bfcl_loader.py
├── parsers/                    # Output parsing
│   ├── base_parser.py         # Abstract Parser
│   ├── code_parser.py         # Code extraction
│   ├── json_parser.py         # JSON parsing
│   ├── mc_parser.py           # Multiple choice
│   ├── robust_parser.py       # Multi-stage fallback
│   └── refusal_detector.py    # Refusal identification
├── metrics/                    # Metric computation
│   ├── base_metric.py
│   ├── pass_at_k.py           # Code correctness
│   ├── exact_match.py
│   ├── multiple_choice.py
│   └── diagnostic.py          # Error analysis
├── execution/                  # Code execution
│   └── sandbox.py             # Sandboxed execution
├── reporting/                  # Results reporting
│   ├── console.py             # Rich console output
│   ├── json_reporter.py       # JSON results
│   └── dashboard.py           # HTML dashboard
└── cli.py                     # Command-line interface
```

## Key Design Decisions

### 1. Task Configuration Schema (YAML)

Each task is defined by a YAML file with standardized schema:

```yaml
name: "task_name"
task_type: "coding|reasoning|function_call|agentic"
description: "Human-readable description"

dataset:
  path: "dataset/path"           # HuggingFace or local
  loader: "loader_name"          # Registered loader
  split: "test"
  subset: null                   # Dataset subset if applicable
  num_samples: 100              # Limit for testing

prompt:
  template: "{prompt}"           # Jinja2 or simple format
  system_message: "..."          # System prompt
  few_shot: []                   # Few-shot examples
  chat_template: "auto"         # auto|none|custom

generation:
  max_tokens: 512
  temperature: 0.0
  top_p: 1.0
  stop_sequences: []

parsing:
  parser: "code|json|mc|robust"
  extraction_mode: "last_block"
  language: "python"             # For code parser

evaluation:
  metric: "pass_at_k|exact_match|multiple_choice"
  k: 1                           # For pass@k
  timeout: 5.0                  # Execution timeout
  sandbox: true                  # Use sandbox

metadata:
  version: "1.0"
  tags: ["coding", "python"]
```

### 2. Model Adapter Pattern

Adapters abstract different model backends:

```python
class BaseModelAdapter(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512, ...) -> GenerationResult:
        pass
    
    def apply_chat_template(self, messages: List[Dict], model_family: str) -> str:
        # Auto-detect and apply appropriate template
        pass
```

**Key Insight**: Chat templates are applied at the adapter level, not the task level. Each adapter knows how to format prompts for its specific model family.

### 3. Robust Parser Design

Multi-stage parsing with confidence scoring:

```python
class RobustParser(BaseParser):
    def parse(self, text: str, **kwargs) -> ParsedResult:
        # Stage 1: Try primary parser
        result = self._try_parse(text, self.primary_parser)
        if result.confidence > 0.9:
            return result
        
        # Stage 2: Try specialized parsers
        for parser in self.fallback_parsers:
            result = self._try_parse(text, parser)
            if result.confidence > 0.7:
                return result
        
        # Stage 3: Check for refusal
        if self._is_refusal(text):
            return ParsedResult(
                content=None,
                confidence=1.0,
                refusal_detected=True
            )
        
        # Stage 4: Return raw with low confidence
        return ParsedResult(
            content=text.strip(),
            confidence=0.0,
            raw_output=text
        )
```

**Confidence Scoring**:
- 1.0: Perfect match (e.g., exact JSON parse)
- 0.7-0.9: Good match (e.g., extracted with regex)
- 0.3-0.6: Partial match (e.g., found content but format uncertain)
- 0.0: No extraction possible, returning raw

### 4. Task Registry Pattern

Dynamic registration for extensibility:

```python
# Registry decorators
@register_task("humaneval")
class HumanEvalTask(Task):
    pass

@register_adapter("ollama")
class OllamaAdapter(BaseModelAdapter):
    pass

@register_parser("code")
class CodeParser(BaseParser):
    pass
```

### 5. Evaluation Flow

```
1. Load Task Config (YAML)
   ↓
2. Create Components (Adapter, Loader, Parser, Task)
   ↓
3. Load Dataset → List[Sample]
   ↓
4. For each Sample:
   a. Format Prompt (Task.get_prompt_template)
   b. Apply Chat Template (Adapter.apply_chat_template)
   c. Generate (Adapter.generate)
   d. Parse Output (Parser.parse)
   e. Check Refusal (RefusalDetector)
   f. Evaluate (Task.evaluate_prediction)
   g. Store Result
   ↓
5. Compute Metrics
   ↓
6. Generate Report
```

## Implementation Plan

### Phase 1: Core Framework (Foundation)
1. **Base Classes**: Task, ModelAdapter, Parser, DatasetLoader, Metric
2. **Config Validation**: JSON Schema for YAML task definitions
3. **Registry System**: Decorator-based registration
4. **Checkpoint System**: Atomic writes, resume capability

### Phase 2: Adapters (Model Backends)
1. **HuggingFace Adapter**: Transformers integration, device management
2. **Ollama Adapter**: HTTP API, local model support
3. **OpenRouter Adapter**: OpenAI-compatible API, multi-provider
4. **Chat Templates**: Auto-detection for major model families

### Phase 3: Tasks & Loaders (Benchmarks)
1. **HumanEval**: Code generation with sandboxed execution
2. **HellaSwag**: Commonsense reasoning, multiple choice
3. **BFCL**: Function calling with JSON parsing
4. **Dataset Loaders**: HuggingFace integration, caching

### Phase 4: Parsers (Robust Extraction)
1. **Code Parser**: Multi-language, fenced/unfenced extraction
2. **JSON Parser**: Schema validation, error recovery
3. **MC Parser**: Multiple choice answer extraction
4. **Robust Parser**: Multi-stage fallback chain
5. **Refusal Detector**: Pattern matching for model refusals

### Phase 5: Metrics & Reporting
1. **Pass@k**: Code correctness metric
2. **Exact Match**: String comparison
3. **Multiple Choice**: Accuracy computation
4. **Console Reporter**: Rich formatted output
5. **JSON Reporter**: Structured results
6. **Diagnostic Analyzer**: Error categorization

### Phase 6: CLI & Integration
1. **CLI Interface**: argparse, subcommands
2. **Smoke Tests**: Quick validation suite
3. **Full Evaluation**: Production runs
4. **Report Generation**: Multi-format output

## Handling Different Model Schemas

### Problem
Different models expect different input formats:
- **Gemma**: `<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n`
- **Llama-3**: `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n`
- **Qwen**: `<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`

### Solution: Adapter-Level Template Application

1. **Auto-Detection**: Infer model family from model name
   ```python
   def detect_model_family(model_name: str) -> str:
       if "gemma" in model_name.lower():
           return "gemma"
       elif "llama" in model_name.lower():
           return "llama"
       elif "qwen" in model_name.lower():
           return "qwen"
       # ... etc
   ```

2. **Template Application**: Each adapter applies its own template
   ```python
   class OllamaAdapter(BaseModelAdapter):
       def generate(self, prompt: str, system: str = None, **kwargs):
           model_family = detect_model_family(self.model_name)
           formatted = apply_chat_template(prompt, system, model_family)
           return self._call_api(formatted, **kwargs)
   ```

3. **Task-Agnostic**: Tasks define logical prompts, adapters handle formatting

## Robust Parsing Strategy

### Problem
LLM outputs vary widely:
- Code: fenced (```python) or unfenced
- JSON: valid, malformed, with markdown fences
- Multiple choice: "A", "Option A", "The answer is A", etc.
- Refusals: "I cannot", "I'm not able to", etc.

### Solution: Confidence-Based Multi-Stage Parsing

```python
class ParsedResult:
    content: Any           # Extracted content
    confidence: float      # 0.0-1.0
    parser_used: str       # Which parser succeeded
    raw_output: str        # Original LLM output
    refusal_detected: bool # Model refused
    error: Optional[str]   # Parsing error if any
```

**Parsing Pipeline**:
1. **Strict Parse**: Try exact format (e.g., JSON.loads)
2. **Lenient Parse**: Handle common variations (e.g., regex extraction)
3. **Heuristic Parse**: Pattern matching for edge cases
4. **Refusal Check**: Detect if model declined
5. **Raw Fallback**: Return original with confidence=0

**False Negative Prevention**:
- Never return empty without checking refusal
- Always preserve raw_output for debugging
- Confidence scoring enables threshold tuning
- Diagnostic logging for parser failures

## Testing Strategy

### Unit Tests
- Parser edge cases (40+ test cases)
- Adapter mock tests
- Config validation tests

### Integration Tests
- End-to-end with small models
- Checkpoint resume verification
- Multi-adapter consistency

### Smoke Tests
- 20 samples per task
- ~10 minute runtime
- Quick validation before full runs

## Success Criteria

1. **Correctness**: No false negatives from parsing
2. **Coverage**: Support Core-3 tasks (HumanEval, HellaSwag, BFCL)
3. **Extensibility**: New task/adapter/parser in <50 lines
4. **Robustness**: >95% successful extractions
5. **Reproducibility**: Checkpoint resume, deterministic sampling

## References

- EleutherAI lm-eval-harness: https://github.com/EleutherAI/lm-evaluation-harness
- OpenAI Evals: https://github.com/openai/evals
- BIG-bench: https://github.com/google/BIG-bench
- Chat Templates: https://huggingface.co/docs/transformers/main/chat_templating
