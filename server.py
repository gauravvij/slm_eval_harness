"""
SLM Evaluation Dashboard - Flask Backend
========================================
Modern minimalist dashboard server for visualizing Small Language Model evaluation results.
Serves model comparison and inference metrics data via REST API.

Hardware Context: 32-core AMD EPYC CPU, 125GB RAM (CPU-only testing environment)
"""

import csv
import io
import json
import os
from pathlib import Path
from datetime import datetime

from flask import Flask, jsonify, render_template, send_from_directory, Response

app = Flask(__name__, template_folder='templates', static_folder='static')

# Configuration
BASE_PATH = Path(__file__).parent
REPORTS_PATH = BASE_PATH / "reports"


def load_json_data(filename):
    """Load JSON data from reports directory."""
    filepath = REPORTS_PATH / filename
    if filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}


def get_system_info():
    """Get system hardware information."""
    # Read memory info from /proc/meminfo
    mem_info = {}
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    mem_info[key.strip()] = value.strip()
    except:
        pass
    
    # Parse memory values
    total_kb = 0
    available_kb = 0
    
    if 'MemTotal' in mem_info:
        total_kb = int(mem_info['MemTotal'].split()[0])
    if 'MemAvailable' in mem_info:
        available_kb = int(mem_info['MemAvailable'].split()[0])
    
    total_gb = total_kb / (1024 * 1024)
    available_gb = available_kb / (1024 * 1024)
    used_gb = total_gb - available_gb
    utilization_pct = (used_gb / total_gb * 100) if total_gb > 0 else 0
    
    # Get CPU info
    cpu_cores = os.cpu_count() or 32
    cpu_model = "AMD EPYC 7B13"
    
    return {
        "cpu": {
            "model": cpu_model,
            "cores": cpu_cores,
            "threads": cpu_cores * 2,
            "architecture": "x86_64"
        },
        "memory": {
            "total_gb": round(total_gb, 1),
            "available_gb": round(available_gb, 1),
            "used_gb": round(used_gb, 1),
            "utilization_pct": round(utilization_pct, 1)
        },
        "gpu": {
            "available": False,
            "count": 0,
            "note": "CPU-only inference environment"
        },
        "timestamp": datetime.now().isoformat()
    }


def process_model_data(comparison_data):
    """Process model comparison data for dashboard display."""
    models = comparison_data.get("models", {})
    processed_models = []
    
    # Include Qwen 3.5, Qwen 3.6, and Gemma 4 variants for comparison
    target_models = [
        # Qwen 3.6 variants
        "qwen3.6-35b-a3b",      # Q4_K_M
        "qwen3.6-35b-a3b-q8",   # Q8_0
        "qwen3.6-35b-a3b-bf16", # BF16
        # Qwen 3.5 variants (older generation)
        "qwen3.5-35b-a3b",      # Q4_K_M
        "qwen3.5-35b-a3b-q8",   # Q8_0
        "qwen3.5-35b-a3b-bf16", # BF16
        # Gemma 4 variants
        "gemma4-26b-a4b-q4",    # Q4_K_M
        "gemma4-26b-a4b-q8",    # Q8_0
        "gemma4-26b-a4b-bf16"   # BF16
    ]
    
    # Define quantization order for ranking
    quant_order = {"BF16": 0, "Q8_0": 1, "Q4_K_M": 2, "Q4": 2}
    
    for model_name, benchmarks in models.items():
        # Filter to only target models
        if model_name not in target_models:
            continue
        
        # Extract quantization level from model name
        quantization = "Q4_K_M"
        if "bf16" in model_name.lower():
            quantization = "BF16"
        elif "q8" in model_name.lower():
            quantization = "Q8_0"
        elif "q4" in model_name.lower():
            quantization = "Q4_K_M"
        
        # Determine model family and generation
        display_name = model_name
        if "qwen3.6" in model_name.lower():
            generation = "3.6"
            display_name = f"Qwen {generation} 35B A3B ({quantization})"
        elif "qwen3.5" in model_name.lower():
            generation = "3.5"
            display_name = f"Qwen {generation} 35B A3B ({quantization})"
        elif "gemma4" in model_name.lower():
            generation = "4"
            display_name = f"Gemma 4 26B A4B ({quantization})"
        
        # Extract benchmark scores
        humaneval = benchmarks.get("humaneval", {})
        hellaswag = benchmarks.get("hellaswag", {})
        bfcl = benchmarks.get("bfcl", {})
        inference = benchmarks.get("inference_metrics", {})
        
        # Calculate average accuracy
        accuracies = []
        if humaneval.get("accuracy", 0) > 0:
            accuracies.append(humaneval.get("accuracy", 0) * 100)
        if hellaswag.get("accuracy", 0) > 0:
            accuracies.append(hellaswag.get("accuracy", 0) * 100)
        if bfcl.get("accuracy", 0) > 0:
            accuracies.append(bfcl.get("accuracy", 0) * 100)
        
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        
        model_data = {
            "name": model_name,
            "quantization": quantization,
            "display_name": display_name,
            "humaneval": {
                "accuracy": round(humaneval.get("accuracy", 0) * 100, 2),
                "samples": humaneval.get("samples", 0),
                "passed": humaneval.get("passed", 0)
            },
            "hellaswag": {
                "accuracy": round(hellaswag.get("accuracy", 0) * 100, 2),
                "samples": hellaswag.get("samples", 0),
                "passed": hellaswag.get("passed", 0)
            },
            "bfcl": {
                "accuracy": round(bfcl.get("accuracy", 0) * 100, 2),
                "samples": bfcl.get("samples", 0),
                "passed": bfcl.get("passed", 0)
            },
            "inference": {
                "ttft_ms": inference.get("ttft_ms", 0),
                "throughput": inference.get("throughput_tok_s", 0),
                "model_size_gb": inference.get("model_size_gb", 35),
                "memory_gb": inference.get("peak_ram_gb", 0)
            },
            "avg_accuracy": round(avg_accuracy, 2),
            "rank": quant_order.get(quantization, 99)
        }
        
        processed_models.append(model_data)
    
    # Sort by rank (BF16 first, then Q8, then Q4)
    processed_models.sort(key=lambda x: x["rank"])
    
    # Assign display ranks
    for i, model in enumerate(processed_models):
        model["display_rank"] = i + 1
    
    return processed_models


@app.route('/')
def index():
    """Serve the main dashboard page."""
    return render_template('index.html')


@app.route('/api/models')
def get_models():
    """API endpoint for model comparison data."""
    comparison_data = load_json_data("model_comparison.json")
    processed_models = process_model_data(comparison_data)
    return jsonify({
        "models": processed_models,
        "generated_at": comparison_data.get("generated_at", datetime.now().isoformat()),
        "total_models": len(processed_models)
    })


@app.route('/api/inference/<quantization>')
def get_inference(quantization):
    """API endpoint for inference metrics by quantization level."""
    # Map quantization to file
    file_map = {
        "q4": "inference_metrics.json",
        "q8": "inference_metrics_q8.json",
        "bf16": "inference_metrics_bf16.json"
    }
    
    filename = file_map.get(quantization.lower(), "inference_metrics.json")
    inference_data = load_json_data(filename)
    
    return jsonify({
        "quantization": quantization.upper(),
        "metrics": inference_data.get("overall_metrics", {}),
        "metadata": inference_data.get("metadata", {})
    })


@app.route('/api/system')
def get_system():
    """API endpoint for system information."""
    return jsonify(get_system_info())


@app.route('/api/export')
def export_csv():
    """API endpoint to export model comparison data as CSV."""
    comparison_data = load_json_data("model_comparison.json")
    processed_models = process_model_data(comparison_data)
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow([
        'Model Name', 'Quantization', 'HumanEval Accuracy (%)', 'HellaSwag Accuracy (%)',
        'BFCL Accuracy (%)', 'Average Accuracy (%)', 'TTFT (ms)', 'Throughput (tok/s)',
        'Peak RAM (GB)', 'Model Size (GB)'
    ])
    
    # Write data rows
    for model in processed_models:
        writer.writerow([
            model['display_name'],
            model['quantization'],
            model['humaneval']['accuracy'],
            model['hellaswag']['accuracy'],
            model['bfcl']['accuracy'],
            model['avg_accuracy'],
            model['inference']['ttft_ms'],
            model['inference']['throughput'],
            model['inference']['memory_gb'],
            model['inference']['model_size_gb']
        ])
    
    # Get the CSV content
    csv_content = output.getvalue()
    output.close()
    
    # Return as downloadable file
    return Response(
        csv_content,
        mimetype='text/csv',
        headers={
            'Content-Disposition': 'attachment; filename=slm_model_comparison.csv'
        }
    )


@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files."""
    return send_from_directory('static', path)


if __name__ == '__main__':
    # Ensure templates directory exists
    (BASE_PATH / "templates").mkdir(exist_ok=True)
    (BASE_PATH / "static").mkdir(exist_ok=True)
    
    print("=" * 60)
    print("SLM Evaluation Dashboard Server")
    print("=" * 60)
    print(f"Server starting on http://0.0.0.0:5000")
    print(f"Press Ctrl+C to stop")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False)
