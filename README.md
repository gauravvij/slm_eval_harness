# SLM Evaluation Harness

A modular, extensible evaluation framework for Small Language Models (SLMs) and Large Language Models (LLMs). Supports multiple benchmarks, model adapters, and output formats.

## Features

- **Multiple Benchmarks**: HumanEval (code), HellaSwag (reasoning), BFCL (function calling)
- **Model Adapters**: Ollama (local), OpenRouter (cloud API), HuggingFace (direct)
- **Robust Parsing**: Multi-stage fallback chain for reliable output extraction
- **Checkpointing**: Resume interrupted evaluations without losing progress
- **Flexible Metrics**: pass@k, exact_match, multiple_choice accuracy
- **Rich Reporting**: Console output, JSON reports, and comparison dashboards

## Quick Start

### Installation

```bash
# Clone and setup
cd slm_eval_harness
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set Python path
export PYTHONPATH=/home/azureuser/slm_eval_harness:$PYTHONPATH
```

### Run Your First Evaluation

```bash
# Evaluate a local Ollama model on HumanEval
python cli.py \
    --task tasks/coding/humaneval_mini.yaml \
    --model qwen3.5:9b \
    --adapter ollama \
    --output reports/my_first_eval.json
```

## Project Structure

```
slm_eval_harness/
├── cli.py                    # Main CLI entry point
├── adapters/                 # Model connection interfaces
│   ├── ollama_adapter.py    # Local Ollama models
│   ├── openrouter_adapter.py # Cloud API via OpenRouter
│   └── hf_adapter.py        # HuggingFace transformers
├── tasks/                    # Benchmark configurations
│   ├── coding/               # HumanEval configs
│   ├── reasoning/            # HellaSwag configs
│   └── function_call/        # BFCL configs
├── dataset_loaders/          # Dataset loading logic
├── parsers/                  # Model output parsers
├── metrics/                  # Scoring implementations
├── core/                     # Evaluation engine
├── reporting/                # Output formatters
└── reports/                  # Evaluation results
```

## Supported Benchmarks

### HumanEval (Code Generation)
- **Type**: Python code completion
- **Samples**: 164 programming problems
- **Metric**: pass@k (code execution)
- **Task Config**: `tasks/coding/humaneval.yaml`

### HellaSwag (Commonsense Reasoning)
- **Type**: Multiple choice completion
- **Samples**: 10,000+ examples
- **Metric**: Accuracy
- **Task Config**: `tasks/reasoning/hellaswag.yaml`

### BFCL (Function Calling)
- **Type**: Function call generation
- **Samples**: 1,700+ function calling scenarios
- **Metric**: Accuracy
- **Task Config**: `tasks/function_call/bfcl_full.yaml`

## Usage

### Basic Evaluation

```bash
python cli.py \
    --task tasks/reasoning/hellaswag.yaml \
    --model gemma4:e4b \
    --adapter ollama \
    --output reports/results.json \
    --checkpoint-dir .checkpoints
```

### With Options

```bash
python cli.py \
    --task tasks/coding/humaneval.yaml \
    --model qwen3.5:9b \
    --adapter ollama \
    --output reports/humaneval_results.json \
    --max-samples 50 \
    --temperature 0.2 \
    --max-tokens 1024 \
    --checkpoint-dir .checkpoints
```

### Resume Interrupted Evaluation

```bash
# Same command - automatically resumes from checkpoint
python cli.py \
    --task tasks/reasoning/hellaswag.yaml \
    --model qwen3.5:9b \
    --adapter ollama \
    --output reports/results.json \
    --checkpoint-dir .checkpoints
```

## Model Adapters

### Ollama (Local)

```bash
# Requires Ollama running on localhost:11434
ollama serve

# Use in evaluation
python cli.py --adapter ollama --model qwen3.5:9b ...
```

### OpenRouter (Cloud)

```bash
# Set API key
export OPENROUTER_API_KEY="your-key-here"

# Use in evaluation
python cli.py --adapter openrouter --model qwen/qwen3.5-9b ...
```

### HuggingFace (Direct)

```bash
# Use in evaluation
python cli.py --adapter hf --model "Qwen/Qwen2.5-7B-Instruct" ...
```

## Configuration

### Task Config Format

```yaml
name: "hellaswag"
description: "HellaSwag commonsense reasoning"

dataset:
  loader: "hellaswag_loader"
  path: "Rowan/hellaswag"
  split: "validation"
  num_samples: null  # null = full dataset

parsing:
  parser: "mc"  # multiple choice parser

evaluation:
  metric: "multiple_choice"

generation:
  max_tokens: 1024
  temperature: 0.0
  top_p: 1.0
```

### Available Parsers

| Parser | Use Case |
|--------|----------|
| `code` | Extract code blocks |
| `mc` | Parse multiple choice (A/B/C/D) |
| `json` | Parse JSON output |
| `robust` | Multi-stage fallback chain |

### Available Metrics

| Metric | Description |
|--------|-------------|
| `exact_match` | String comparison |
| `pass_at_k` | Code execution pass rate |
| `multiple_choice` | Option selection accuracy |

## Results & Reporting

### Individual Results

Each evaluation produces a JSON file:

```json
{
  "accuracy": 0.7683,
  "samples": 164,
  "passed": 126,
  "failed": 38,
  "results": [...]
}
```

### Model Comparison Report

The master comparison report at `reports/model_comparison.json` aggregates results across models and benchmarks:

```json
{
  "models": {
    "qwen3.5:9b": {
      "humaneval": {"accuracy": 0.768, "samples": 164},
      "hellaswag": {"accuracy": 0.504, "samples": 1000}
    }
  },
  "summary": {
    "total_samples_evaluated": 1164,
    "models_compared": 1
  }
}
```

## Development

### Running Tests

```bash
# All tests
pytest tests/

# Specific test
pytest tests/test_parsers.py
```

### Adding a New Task

1. Create task config in `tasks/<category>/<name>.yaml`
2. Implement dataset loader in `dataset_loaders/`
3. Add to registry in `core/registry.py` (if needed)
4. Test with mini config first

### Adding a New Adapter

1. Implement adapter in `adapters/<name>_adapter.py`
2. Inherit from `BaseModelAdapter`
3. Register with `@register_adapter("name")`
4. Add to CLI in `cli.py`

## Requirements

- Python 3.8+
- PyTorch 2.0+ (for HF adapter)
- Ollama (for local models)
- OpenRouter API key (for cloud models)

See `requirements.txt` for full dependency list.

## Troubleshooting

### Import Errors

```bash
export PYTHONPATH=/home/azureuser/slm_eval_harness:$PYTHONPATH
```

### Ollama Connection Issues

```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve
```

### 0% Accuracy on HellaSwag

Usually indicates token limit issues. Check:
- `max_tokens` in task config (should be 1024 for HellaSwag)
- Model's actual output capability
- Checkpoint file for raw model outputs

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest tests/`
4. Submit a pull request

## Acknowledgments

- HumanEval: OpenAI
- HellaSwag: Zellers et al.
- BFCL: Berkeley Function Calling Leaderboard
