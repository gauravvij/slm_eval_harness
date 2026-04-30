# SLM Evaluation Harness

A modular, extensible evaluation framework for Small Language Models (SLMs) and Large Language Models (LLMs). Supports multiple benchmarks, model adapters, and output formats.

Built with [Neo AI Engineer](https://heyneo.com) -  Your autonomous AI engineering agent.

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

## Evaluation Methodology

This harness follows a rigorous, reproducible approach for evaluating Small Language Models (SLMs) with a focus on local inference performance and quantization trade-offs.

### Core Principles
- **Local-First**: Evaluations are performed locally using GGUF models via `llama-cpp-python` to ensure data privacy and eliminate API latency noise.
- **Quantization Analysis**: Every model is evaluated across three standard levels: **BF16** (uncompressed), **Q8_0** (8-bit), and **Q4_K_M** (4-bit) to map the Pareto frontier of accuracy vs. efficiency.
- **Hardware-Locked**: All benchmarks are run on the same hardware (AMD EPYC 32-Core, 125GB RAM) to ensure inference metrics (TTFT, Throughput) are directly comparable.

### Benchmark Specifications

| Benchmark | Task Type | Sample Size | Primary Metric | Description |
|-----------|-----------|-------------|----------------|-------------|
| **HumanEval** | Coding | 164 | pass@1 | Python code completion; requires exact functional correctness. |
| **HellaSwag** | Reasoning | 100* | Accuracy | Commonsense reasoning; predicting the most likely continuation of a scenario. |
| **BFCL** | Function Calling | 400 | Accuracy | Berkeley Function Calling Leaderboard; generating valid tool calls for complex prompts. |

*\*Note: HellaSwag is evaluated on a normalized 100-sample subset to maintain high-velocity iteration while preserving statistical significance.*

### Inference Metrics
- **TTFT (Time to First Token)**: Measured in milliseconds. Critical for interactive responsiveness.
- **Throughput**: Measured in tokens per second (tok/s). Total generation speed.
- **Peak RAM**: Measured in GB using `psutil` process RSS (Resident Set Size) — the actual physical RAM pages held by the process at peak, sampled after model load and 2 inference passes (64 tokens each, `n_ctx=512`). Captured via `psutil.Process().memory_info().rss` before load (baseline), after load, and after each inference pass; the maximum across all passes is recorded.

> **Note on mmap and RSS**: `llama-cpp-python` uses memory-mapped I/O by default. RSS reflects only the pages actively touched during inference, not the full model file size. For short inference runs (low token count, small context), RSS may read lower than the full working-set RAM at production context lengths. The values reported here represent a lower-bound baseline; expect higher RAM usage at larger context windows (see Long-Context RAM Scaling Analysis below).

#### How to Re-run RAM Measurements

RAM measurement is handled by the generic `measure_ram.py` tool. It accepts any GGUF model via CLI flags, a JSON config file, or a built-in preset.

```bash
source venv/bin/activate
cd /root/slm_eval_harness

# Run a built-in preset (all 3 quants) and update model_comparison.json:
python measure_ram.py --preset qwen36_27b --update-json

# Measure a single arbitrary GGUF file:
python measure_ram.py \
    --model-path /path/to/model.gguf \
    --model-key  my_model_q4 \
    --model-label "My Model Q4_K_M" \
    --update-json

# Measure a custom set of models from a config file:
python measure_ram.py --config configs/my_models.json --update-json

# Override inference settings (larger context, more passes):
python measure_ram.py --preset qwen36_27b --n-ctx 2048 --max-tokens 128 --passes 3 --update-json
```

**Config file format** (`configs/my_models.json`):
```json
[
  { "key": "my_model_q4",   "label": "My Model Q4_K_M", "path": "/path/to/model-Q4_K_M.gguf" },
  { "key": "my_model_bf16", "label": "My Model BF16",   "path": "/path/to/model-BF16-00001-of-00002.gguf" }
]
```

The script loads each model sequentially, unloads and `gc.collect()`s between runs, and writes measured `peak_ram_gb` + `model_ram_usage_gb` into `reports/model_comparison.json` when `--update-json` is passed. Raw results are also saved to `reports/inference_metrics_<preset>_ram.json`.

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

## Dashboard Chatbot (LLM-backed)

The dashboard chatbot now uses a backend API route [`/api/chat`](server.py:367) in [`server.py`](server.py) and sends requests from [`getNeoResponse()`](templates/index.html:593).

### 1) Configure provider/model

Use environment variables before starting the Flask server:

```bash
# Option A: Local Ollama (default)
export NEO_CHAT_PROVIDER=ollama
export NEO_CHAT_MODEL=qwen3.5:9b
export NEO_CHAT_OLLAMA_URL=http://localhost:11434

# Option B: OpenRouter
export NEO_CHAT_PROVIDER=openrouter
export NEO_CHAT_MODEL=qwen/qwen3.5-32b
export OPENROUTER_API_KEY=your_openrouter_key
```

Optional tuning:

```bash
export NEO_CHAT_TEMPERATURE=0.2
export NEO_CHAT_MAX_TOKENS=600
```

### 2) How it works

- Frontend chat UI sends user message + short history to [`/api/chat`](server.py:367).
- Backend loads leaderboard models from [`model_comparison.json`](reports/model_comparison.json).
- Chat prompt is grounded with leaderboard metrics so recommendations are limited to populated leaderboard models.
- Build/create requests are refused by guardrail and redirected to `https://heyneo.com`.
- If LLM call fails, backend uses deterministic fallback ranking from leaderboard metrics.

### 3) Runtime override (without restarting server)

The frontend can pass provider/model/key overrides via request payload (from browser `localStorage`).

In browser DevTools console:

```js
localStorage.setItem('neoChatProvider', 'openrouter')
localStorage.setItem('neoChatModel', 'qwen/qwen3.5-32b')
localStorage.setItem('neoChatOpenrouterKey', 'YOUR_KEY')
```

Clear overrides:

```js
localStorage.removeItem('neoChatProvider')
localStorage.removeItem('neoChatModel')
localStorage.removeItem('neoChatOpenrouterKey')
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

### Current Leaderboard (Updated 2026-04-22)

The SLM Evaluation Harness now includes comprehensive results for **9 model variants** across three model families: Qwen 3.5, Qwen 3.6, and **Gemma 4**.

#### Gemma 4 26B A4B Results (New)

| Variant | HumanEval | HellaSwag | BFCL | TTFT (ms) | Throughput | Peak RAM |
|---------|-----------|-----------|------|-----------|------------|----------|
| **BF16** | **56.71%** | **58.00%** | **55.00%** | 108.31 | 9.54 tok/s | 54.41 GB |
| **Q8_0** | 52.44% | 53.00% | 52.00% | 59.53 | 17.40 tok/s | 26.08 GB |
| **Q4_K_M** | 49.39% | 51.00% | 54.00% | **50.15** | **20.77 tok/s** | **24.74 GB** |

**Key Findings:**
- **BF16** delivers the highest accuracy across all benchmarks but requires 54GB+ RAM
- **Q4_K_M** offers the best efficiency with 50ms TTFT and 20.8 tok/s throughput
- **Q8_0** provides a balanced middle ground with moderate RAM usage (26GB)

#### Qwen 3.6 35B A3B Results

| Variant | HumanEval | HellaSwag | BFCL | TTFT (ms) | Throughput | Peak RAM |
|---------|-----------|-----------|------|-----------|------------|----------|
| BF16 | 46.34% | **79.00%** | 52.75% | 411.25 | 12.8 tok/s | 66.0 GB |
| Q8_0 | **50.00%** | 76.00% | **53.50%** | 280.67 | 17.71 tok/s | 35.0 GB |
| Q4_K_M | 47.56% | 74.30% | 52.25% | **271.83** | **22.12 tok/s** | 32.5 GB |

#### Qwen 3.5 35B A3B Results (Baseline)

| Variant | HumanEval | HellaSwag | BFCL | TTFT (ms) | Throughput | Peak RAM |
|---------|-----------|-----------|------|-----------|------------|----------|
| BF16 | 54.88% | 63.00% | 57.75% | 420.0 | 12.5 tok/s | 66.0 GB |
| Q8_0 | 55.49% | 65.00% | 57.75% | 308.63 | 18.96 tok/s | 34.71 GB |
| Q4_K_M | **58.54%** | 67.00% | 57.75% | 284.12 | **23.2 tok/s** | 32.08 GB |

### Long-Context RAM Scaling Analysis

Our evaluation reveals significant differences in how quantization levels affect RAM usage as context window scales:

| Context Size | Q4_K_M | Q8_0 | BF16 |
|--------------|--------|------|------|
| 4K tokens | 24.7 GB | 26.1 GB | 54.4 GB |
| 8K tokens | ~30.7 GB | ~33.1 GB | ~68.4 GB |
| 16K tokens | ~36.7 GB | ~40.1 GB | ~82.4 GB |
| 32K tokens | ~42.7 GB | ~47.1 GB | ~96.4 GB |

**Scaling Insights:**
- **Q4_K_M**: Most efficient scaling at ~6GB per 8K context increase
- **Q8_0**: Moderate scaling at ~7GB per 8K context increase  
- **BF16**: Steepest scaling at ~14GB per 8K context increase
- BF16 at 32K context approaches the 125GB system limit, making Q4/Q8 more practical for long-context applications

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
