# Agent Guide: SLM Evaluation Harness

> **For AI Coding Agents**: This document provides step-by-step instructions for running model evaluations and updating reports.

## Quick Start (Agent Checklist)

When asked to evaluate a model, follow this exact sequence:

1. [ ] Check available adapters (Ollama, OpenRouter, HuggingFace)
2. [ ] Verify model availability on chosen adapter
3. [ ] Select appropriate task config from `tasks/`
4. [ ] Run evaluation via CLI
5. [ ] Update `reports/model_comparison.json` with results

---

## 1. Project Structure Overview

```
slm_eval_harness/
├── cli.py                    # Main entry point - USE THIS
├── adapters/                 # Model connection modules
│   ├── ollama_adapter.py    # Local models via Ollama (localhost:11434)
│   ├── openrouter_adapter.py # Cloud models via OpenRouter API
│   └── hf_adapter.py        # HuggingFace models
├── tasks/                    # Benchmark configurations
│   ├── coding/humaneval.yaml
│   ├── reasoning/hellaswag.yaml
│   └── function_call/bfcl.yaml
├── dataset_loaders/          # Dataset loading logic
├── parsers/                  # Output parsing (code, JSON, multiple choice)
├── metrics/                  # Scoring logic
├── core/                     # Evaluation engine
├── reports/                  # OUTPUT: Store all results here
│   └── model_comparison.json # MASTER REPORT - UPDATE THIS
└── tests/                    # Test scripts
```

---

## 2. Running an Evaluation

### 2.1 Basic Command Structure

```bash
# Activate virtual environment first
source venv/bin/activate

# Run evaluation
python cli.py \
    --task tasks/reasoning/hellaswag.yaml \
    --model gemma4:e4b \
    --adapter ollama \
    --output reports/hellaswag_gemma4_results.json \
    --checkpoint-dir .checkpoints
```

### 2.2 Available Task Configs

| Task | Config Path | Type | Metric |
|------|-------------|------|--------|
| HumanEval | `tasks/coding/humaneval.yaml` | Code generation | pass@k |
| HumanEval Mini | `tasks/coding/humaneval_mini.yaml` | Code generation (20 samples) | pass@k |
| HellaSwag | `tasks/reasoning/hellaswag.yaml` | Multiple choice | accuracy |
| HellaSwag Mini | `tasks/reasoning/hellaswag_mini.yaml` | Multiple choice (20 samples) | accuracy |
| BFCL | `tasks/function_call/bfcl_full.yaml` | Function calling | accuracy |
| BFCL Tiny | `tasks/function_call/bfcl_tiny.yaml` | Function calling (20 samples) | accuracy |

### 2.3 Available Adapters

| Adapter | Flag | Use Case | Requirements |
|---------|------|----------|--------------|
| Ollama | `--adapter ollama` | Local models | Ollama running on localhost:11434 |
| OpenRouter | `--adapter openrouter` | Cloud API models | OPENROUTER_API_KEY env var |
| HuggingFace | `--adapter hf` | Direct HF models | Sufficient GPU/CPU memory |

### 2.4 CLI Options

```bash
python cli.py --help  # Show all options

# Common options:
--task PATH           # Task configuration YAML
--model NAME          # Model name (e.g., "qwen3.5:9b", "gemma4:e4b")
--adapter TYPE        # Adapter type (ollama/openrouter/hf)
--output PATH         # Where to save results JSON
--checkpoint-dir DIR  # Resume from checkpoints if interrupted
--max-samples N       # Limit samples (for testing)
--temperature FLOAT   # Generation temperature (default: 0.0)
--max-tokens INT      # Max tokens to generate
```

---

## 3. Task-Specific Instructions

### 3.1 HumanEval (Code Generation)

```bash
# Full evaluation (164 samples)
python cli.py \
    --task tasks/coding/humaneval.yaml \
    --model qwen3.5:9b \
    --adapter ollama \
    --output reports/humaneval_qwen3.5_9b.json

# Quick test (20 samples)
python cli.py \
    --task tasks/coding/humaneval_mini.yaml \
    --model qwen3.5:9b \
    --adapter ollama \
    --output reports/humaneval_qwen3.5_9b_mini.json
```

**Expected Output Format:**
```json
{
  "accuracy": 0.7683,
  "samples": 164,
  "passed": 126,
  "failed": 38
}
```

### 3.2 HellaSwag (Multiple Choice Reasoning)

```bash
# Full evaluation (1000 samples) - TAKES 2+ HOURS
python cli.py \
    --task tasks/reasoning/hellaswag.yaml \
    --model qwen3.5:9b \
    --adapter ollama \
    --output reports/hellaswag_qwen3.5_9b.json \
    --checkpoint-dir .checkpoints

# Quick smoke test (20 samples, ~20 min)
python cli.py \
    --task tasks/reasoning/hellaswag_mini.yaml \
    --model qwen3.5:9b \
    --adapter ollama \
    --output reports/hellaswag_qwen3.5_9b_smoke.json
```

**CRITICAL:** HellaSwag requires `max_tokens: 1024` in the task config. If using a model with limited output tokens (like gemma4:e4b with 32-token limit), accuracy will be 0%.

**Expected Output Format:**
```json
{
  "accuracy": 0.504,
  "samples": 1000,
  "passed": 504,
  "avg_time_per_sample": 64.5
}
```

### 3.3 BFCL (Function Calling)

```bash
# Full evaluation
python cli.py \
    --task tasks/function_call/bfcl_full.yaml \
    --model qwen3.5:9b \
    --adapter ollama \
    --output reports/bfcl_qwen3.5_9b.json

# Quick test
python cli.py \
    --task tasks/function_call/bfcl_tiny.yaml \
    --model qwen3.5:9b \
    --adapter ollama \
    --output reports/bfcl_qwen3.5_9b_tiny.json
```

---

## 4. Updating the Model Comparison Report

After running evaluations, you MUST update `reports/model_comparison.json`.

### 4.1 Report Structure

```json
{
  "models": {
    "model_name:version": {
      "task_name": {
        "accuracy": 0.0,
        "samples": 0,
        "passed": 0,
        "avg_time_per_sample": 0.0,
        "token_limit": 0,
        "note": "optional note"
      }
    }
  },
  "benchmarks": {},
  "generated_at": "YYYY-MM-DD",
  "summary": {
    "total_samples_evaluated": 0,
    "models_compared": 0,
    "benchmarks_completed": 0,
    "notes": []
  }
}
```

### 4.2 Step-by-Step Update Process

1. **Read existing report:**
   ```python
   import json
   with open('reports/model_comparison.json') as f:
       report = json.load(f)
   ```

2. **Add/update model entry:**
   ```python
   # Model name format: "modelname:version" (e.g., "qwen3.5:9b", "gemma4:e4b")
   model_key = "qwen3.5:9b"
   
   if model_key not in report["models"]:
       report["models"][model_key] = {}
   
   # Add task results
   report["models"][model_key]["hellaswag"] = {
       "accuracy": 0.504,
       "samples": 1000,
       "passed": 504,
       "avg_time_per_sample": 64.5,
       "token_limit": 1024
   }
   ```

3. **Update summary statistics:**
   ```python
   # Recalculate totals
   total_samples = sum(
       task["samples"] 
       for model in report["models"].values() 
       for task in model.values()
   )
   
   report["summary"]["total_samples_evaluated"] = total_samples
   report["summary"]["models_compared"] = len(report["models"])
   report["summary"]["benchmarks_completed"] = len(set(
       task_name 
       for model in report["models"].values() 
       for task_name in model.keys()
   ))
   report["generated_at"] = "2025-04-16"
   ```

4. **Save updated report:**
   ```python
   with open('reports/model_comparison.json', 'w') as f:
       json.dump(report, f, indent=2)
   ```

### 4.3 Example: Complete Update Script

```python
import json
from datetime import datetime

def update_model_comparison(model_name, task_name, results_file):
    """
    Update model_comparison.json with new evaluation results.
    
    Args:
        model_name: e.g., "qwen3.5:9b"
        task_name: e.g., "hellaswag", "humaneval", "bfcl"
        results_file: Path to the JSON results file from CLI
    """
    # Load existing report
    with open('reports/model_comparison.json') as f:
        report = json.load(f)
    
    # Load new results
    with open(results_file) as f:
        results = json.load(f)
    
    # Initialize model entry if needed
    if model_name not in report["models"]:
        report["models"][model_name] = {}
    
    # Extract metrics
    task_results = {
        "accuracy": results.get("accuracy", 0.0),
        "samples": results.get("samples", 0),
        "passed": results.get("passed", 0)
    }
    
    # Add optional fields if present
    if "avg_time_per_sample" in results:
        task_results["avg_time_per_sample"] = results["avg_time_per_sample"]
    if "token_limit" in results:
        task_results["token_limit"] = results["token_limit"]
    
    # Update report
    report["models"][model_name][task_name] = task_results
    
    # Update summary
    total_samples = sum(
        task["samples"]
        for model in report["models"].values()
        for task in model.values()
    )
    report["summary"]["total_samples_evaluated"] = total_samples
    report["summary"]["models_compared"] = len(report["models"])
    report["generated_at"] = datetime.now().strftime("%Y-%m-%d")
    
    # Save
    with open('reports/model_comparison.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Updated report: {model_name} / {task_name}")
    print(f"Total samples in report: {total_samples}")

# Usage example:
# update_model_comparison("qwen3.5:9b", "hellaswag", "reports/hellaswag_qwen3.5_9b.json")
```

---

## 5. Common Issues & Solutions

### 5.1 Import Errors

If you see `ModuleNotFoundError` or import errors:

```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=/home/azureuser/slm_eval_harness:$PYTHONPATH

# Or run with python -m
python -m cli --task tasks/reasoning/hellaswag.yaml ...
```

### 5.2 Ollama Connection Failed

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve

# Pull model if not available
ollama pull qwen3.5:9b
```

### 5.3 Checkpoint Resume

If evaluation was interrupted, resume from checkpoint:

```bash
# Same command - will auto-resume if checkpoint exists
python cli.py \
    --task tasks/reasoning/hellaswag.yaml \
    --model qwen3.5:9b \
    --adapter ollama \
    --output reports/hellaswag_qwen3.5_9b.json \
    --checkpoint-dir .checkpoints
```

### 5.4 0% Accuracy on HellaSwag

This usually means:
1. **Token limit too low** - Model outputs truncated (check `max_tokens` in task config)
2. **Parser mismatch** - Model output format doesn't match expected format
3. **Model limitation** - Some models can't handle multiple choice well

**Debug steps:**
```bash
# Run mini test first
python cli.py --task tasks/reasoning/hellaswag_mini.yaml ...

# Check checkpoint file to see raw outputs
ls -la .checkpoints/
cat .checkpoints/hellaswag_*.json
```

---

## 6. Verification Checklist

Before marking a task complete, verify:

- [ ] Evaluation ran without errors
- [ ] Results JSON file created and non-empty
- [ ] `model_comparison.json` updated with new results
- [ ] Summary statistics recalculated correctly
- [ ] Date updated in report
- [ ] File paths are absolute or relative to project root

---

## 7. Example: Complete Workflow

```bash
# 1. Setup
cd /home/azureuser/slm_eval_harness
source venv/bin/activate
export PYTHONPATH=/home/azureuser/slm_eval_harness:$PYTHONPATH

# 2. Verify Ollama
curl http://localhost:11434/api/tags | grep qwen3.5

# 3. Run evaluation
python cli.py \
    --task tasks/reasoning/hellaswag_mini.yaml \
    --model qwen3.5:9b \
    --adapter ollama \
    --output reports/hellaswag_qwen3.5_9b_smoke.json \
    --checkpoint-dir .checkpoints

# 4. Update comparison report (use Python script from section 4.3)
python << 'EOF'
import json
# ... update script here ...
EOF

# 5. Verify update
cat reports/model_comparison.json | python -m json.tool
```

---

## 8. Adapter-Specific Notes

### 8.1 Ollama Adapter

- **Default host:** `http://localhost:11434`
- **Model names:** Use Ollama model tags (e.g., `qwen3.5:9b`, `gemma4:e4b`, `llama3.1:8b`)
- **Auto-pull:** Enabled by default - will download missing models

### 8.2 OpenRouter Adapter

- **API Key:** Must set `OPENROUTER_API_KEY` environment variable
- **Model names:** Use OpenRouter format (e.g., `qwen/qwen3.5-9b`, `google/gemma-4b`)
- **Rate limits:** Be aware of API rate limits for large evaluations

### 8.3 HuggingFace Adapter

- **Model names:** HuggingFace model IDs (e.g., `Qwen/Qwen2.5-7B-Instruct`)
- **Hardware:** Requires sufficient GPU/CPU memory
- **Authentication:** May need `HF_TOKEN` for gated models

---

## 9. Task Configuration Format

When creating new task configs, use this YAML structure:

```yaml
name: "task_name"
description: "Description of the task"

dataset:
  loader: "loader_name"  # e.g., hellaswag_loader, humaneval_loader
  path: "dataset/path"
  split: "test"
  num_samples: null  # null for full dataset

parsing:
  parser: "parser_name"  # robust, code, mc, json
  language: "python"  # for code parser
  extraction_mode: "last_block"  # for code parser

evaluation:
  metric: "metric_name"  # exact_match, pass_at_k, multiple_choice
  
generation:
  max_tokens: 512
  temperature: 0.0
  top_p: 1.0
```

---

## 10. Metrics Reference

| Metric | Task Type | Description |
|--------|-----------|-------------|
| `exact_match` | Code, QA | String match after normalization |
| `pass_at_k` | Code generation | Code execution passes tests |
| `multiple_choice` | Reasoning | Correct option selected |
| `function_call` | Tool use | Function call format correct |

---

**Last Updated:** 2025-04-16

**Questions?** Check `README.md` for general project overview or examine existing task configs in `tasks/` directory.
