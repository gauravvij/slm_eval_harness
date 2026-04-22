import json
import os

report_path = '/root/slm_eval_harness/reports/model_comparison.json'

with open(report_path, 'r') as f:
    data = json.load(f)

# Update Gemma 4 Q4
data['models']['gemma4-26b-a4b-q4'] = {
    "humaneval": {"accuracy": 0.4939, "passed": 81, "samples": 164, "note": "Q4_K_M quantization - 164 samples full evaluation"},
    "hellaswag": {"accuracy": 0.51, "passed": 51, "samples": 100, "note": "Full 100-sample evaluation"},
    "bfcl": {"accuracy": 0.54, "passed": 54, "samples": 100, "note": "Q4_K_M quantization - 100 sample evaluation"},
    "inference_metrics": {
        "ttft_ms": 50.15,
        "throughput_tok_s": 20.77,
        "peak_ram_gb": 24.74,
        "model_size_gb": 15.71,
        "quantization": "Q4_K_M"
    }
}

# Update Gemma 4 Q8
data['models']['gemma4-26b-a4b-q8'] = {
    "humaneval": {"accuracy": 0.5244, "passed": 86, "samples": 164, "note": "Q8_0 quantization - 164 samples full evaluation"},
    "hellaswag": {"accuracy": 0.53, "passed": 53, "samples": 100, "note": "Full 100-sample evaluation (resumed from checkpoint)"},
    "bfcl": {"accuracy": 0.52, "passed": 52, "samples": 100, "note": "Q8_0 quantization - 100 sample evaluation"},
    "inference_metrics": {
        "ttft_ms": 59.53,
        "throughput_tok_s": 17.40,
        "peak_ram_gb": 26.08,
        "model_size_gb": 26.00,
        "quantization": "Q8_0"
    }
}

# Add Gemma 4 BF16
data['models']['gemma4-26b-a4b-bf16'] = {
    "humaneval": {"accuracy": 0.5671, "passed": 93, "samples": 164, "note": "BF16 quantization - 164 samples full evaluation"},
    "hellaswag": {"accuracy": 0.58, "passed": 58, "samples": 100, "note": "Full 100-sample evaluation"},
    "bfcl": {"accuracy": 0.55, "passed": 55, "samples": 100, "note": "BF16 quantization - 100 sample evaluation"},
    "inference_metrics": {
        "ttft_ms": 108.31,
        "throughput_tok_s": 9.54,
        "peak_ram_gb": 54.41,
        "model_size_gb": 54.00,
        "quantization": "BF16"
    }
}

with open(report_path, 'w') as f:
    json.dump(data, f, indent=4)

print("Successfully updated model_comparison.json with final Gemma 4 results.")
