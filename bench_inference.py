#!/usr/bin/env python3
"""
Inference Benchmarking Script for SLM Evaluation Harness
Measures TTFT, Throughput, and Peak RAM usage for GGUF models using llama-cpp-python
"""

import os
import sys
import time
import json
import statistics
import tracemalloc
import psutil
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adapters.gguf_adapter import GGUFAdapter


# Standard test prompt for consistent benchmarking
TEST_PROMPT = """Write a Python function to calculate the factorial of a number using recursion.

Example:
def factorial(n):
    # Your code here
    pass

print(factorial(5))  # Should output 120

Provide the complete implementation:"""


def get_peak_ram_usage() -> float:
    """Get current peak RAM usage in GB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 3)  # Convert bytes to GB


def measure_inference_metrics(
    model_path: str,
    model_name: str,
    max_tokens: int = 50,
    temperature: float = 0.7,
    n_ctx: int = 4096,
    warmup_runs: int = 1,
    measurement_runs: int = 5,
) -> Dict[str, Any]:
    """
    Measure TTFT (Time To First Token), Throughput, and Peak RAM for a GGUF model.
    
    Args:
        model_path: Path to the GGUF model file
        model_name: Display name for the model
        max_tokens: Maximum tokens to generate (default 50 for quick benchmark)
        temperature: Sampling temperature
        n_ctx: Context window size
        warmup_runs: Number of warmup runs before measurement
        measurement_runs: Number of measurement runs
    
    Returns:
        Dictionary with TTFT, throughput, and RAM metrics
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking: {model_name}")
    print(f"Model Path: {model_path}")
    print(f"{'='*70}")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return None
    
    # Get initial RAM baseline
    initial_ram = get_peak_ram_usage()
    print(f"Initial RAM usage: {initial_ram:.2f} GB")
    
    # Initialize adapter (CPU-only, no GPU layers)
    print(f"\nLoading model...")
    start_load = time.perf_counter()
    
    try:
        adapter = GGUFAdapter(
            model_name=model_path,
            device="cpu",
            n_ctx=n_ctx,
            n_gpu_layers=0,  # CPU only
            verbose=False,
        )
        load_time = time.perf_counter() - start_load
        print(f"Model loaded in {load_time:.2f}s")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None
    
    # Get RAM after loading
    post_load_ram = get_peak_ram_usage()
    print(f"RAM after loading: {post_load_ram:.2f} GB")
    
    # Storage for measurements
    ttft_ms_list = []
    throughput_list = []
    tokens_list = []
    
    # Warmup runs
    print(f"\nRunning {warmup_runs} warmup(s)...")
    for w in range(warmup_runs):
        print(f"  Warmup {w + 1}/{warmup_runs}...", end=" ", flush=True)
        try:
            _ = adapter.generate(
                prompt=TEST_PROMPT,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            print("done")
        except Exception as e:
            print(f"ERROR: {e}")
            return None
    
    # Measurement runs
    print(f"\nRunning {measurement_runs} measurement(s)...")
    for r in range(measurement_runs):
        print(f"  Run {r + 1}/{measurement_runs}...", end=" ", flush=True)
        
        try:
            # Use streaming to capture TTFT accurately
            stream = adapter._model.create_completion(
                prompt=TEST_PROMPT,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            
            # Process streaming output
            start_time = time.perf_counter()
            first_token_time = None
            full_text = ""
            token_count = 0
            
            for chunk in stream:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    text = chunk['choices'][0].get('text', '')
                    full_text += text
                    if text:
                        token_count += 1
            
            end_time = time.perf_counter()
            
            # Calculate metrics
            ttft_ms = (first_token_time - start_time) * 1000
            total_time_s = end_time - start_time
            
            # More accurate token count
            if adapter._model and full_text:
                try:
                    token_count = len(adapter._model.tokenize(full_text.encode('utf-8')))
                except:
                    pass  # Use the count from streaming
            
            throughput = token_count / total_time_s if total_time_s > 0 else 0
            
            ttft_ms_list.append(ttft_ms)
            throughput_list.append(throughput)
            tokens_list.append(token_count)
            
            print(f"TTFT={ttft_ms:.2f}ms, Throughput={throughput:.2f}tok/s, Tokens={token_count}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    # Get peak RAM after generation
    peak_ram = get_peak_ram_usage()
    print(f"\nPeak RAM usage: {peak_ram:.2f} GB")
    
    # Calculate statistics
    if len(ttft_ms_list) == 0:
        print("ERROR: No successful measurements")
        return None
    
    avg_ttft = statistics.mean(ttft_ms_list)
    std_ttft = statistics.stdev(ttft_ms_list) if len(ttft_ms_list) > 1 else 0
    avg_throughput = statistics.mean(throughput_list)
    std_throughput = statistics.stdev(throughput_list) if len(throughput_list) > 1 else 0
    
    # Model size estimation based on file size
    model_size_gb = os.path.getsize(model_path) / (1024 ** 3)
    
    results = {
        "model_name": model_name,
        "model_path": model_path,
        "model_size_gb": round(model_size_gb, 2),
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n_ctx": n_ctx,
            "warmup_runs": warmup_runs,
            "measurement_runs": measurement_runs,
        },
        "metrics": {
            "ttft_ms": {
                "mean": round(avg_ttft, 2),
                "std": round(std_ttft, 2),
                "min": round(min(ttft_ms_list), 2),
                "max": round(max(ttft_ms_list), 2),
                "raw": [round(x, 2) for x in ttft_ms_list],
            },
            "throughput_tokens_per_sec": {
                "mean": round(avg_throughput, 2),
                "std": round(std_throughput, 2),
                "min": round(min(throughput_list), 2),
                "max": round(max(throughput_list), 2),
                "raw": [round(x, 2) for x in throughput_list],
            },
            "memory": {
                "initial_ram_gb": round(initial_ram, 2),
                "post_load_ram_gb": round(post_load_ram, 2),
                "peak_ram_gb": round(peak_ram, 2),
                "model_ram_usage_gb": round(peak_ram - initial_ram, 2),
            },
            "tokens_generated": {
                "mean": round(statistics.mean(tokens_list), 1),
                "total": sum(tokens_list),
            },
        },
        "summary": {
            "ttft_ms": round(avg_ttft, 2),
            "throughput_tok_s": round(avg_throughput, 2),
            "peak_ram_gb": round(peak_ram, 2),
            "model_size_gb": round(model_size_gb, 2),
        }
    }
    
    return results


def print_summary(results: Dict[str, Any]):
    """Print a formatted summary of results."""
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    print(f"\nModel: {results['model_name']}")
    print(f"Model Size: {results['model_size_gb']:.2f} GB")
    
    summary = results['summary']
    print(f"\n--- KEY METRICS ---")
    print(f"TTFT (Time To First Token): {summary['ttft_ms']:.2f} ms")
    print(f"Throughput: {summary['throughput_tok_s']:.2f} tokens/sec")
    print(f"Peak RAM Usage: {summary['peak_ram_gb']:.2f} GB")
    
    mem = results['metrics']['memory']
    print(f"\n--- MEMORY BREAKDOWN ---")
    print(f"Initial RAM: {mem['initial_ram_gb']:.2f} GB")
    print(f"Post-Load RAM: {mem['post_load_ram_gb']:.2f} GB")
    print(f"Model RAM Usage: {mem['model_ram_usage_gb']:.2f} GB")
    
    print("\n" + "="*70)


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    """Main entry point - benchmark all available models."""
    
    # Define models to benchmark
    # Format: (model_key, model_path, display_name, n_ctx_override)
    models_to_benchmark = [
        # Gemma 4 26B A4B - Q4_K_M
        ("gemma4-26b-a4b-q4", "/root/slm_eval_harness/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf", 
         "Gemma4-26B-A4B-Q4_K_M", 4096),
        # Gemma 4 26B A4B - Q8_0
        ("gemma4-26b-a4b-q8", "/root/slm_eval_harness/models/gemma-4-26B-A4B-it-Q8_0.gguf", 
         "Gemma4-26B-A4B-Q8_0", 4096),
    ]
    
    # BF16 models (split files - need special handling)
    bf16_models = [
        ("gemma4-26b-a4b-bf16", "/root/slm_eval_harness/models/BF16/gemma-4-26B-A4B-it-BF16-00001-of-00002.gguf", 
         "Gemma4-26B-A4B-BF16"),
    ]
    
    print("="*70)
    print("INFERENCE BENCHMARKING - SLM Evaluation Harness")
    print("="*70)
    print(f"Hardware: 32-core CPU, 125GB RAM")
    print(f"Metrics: TTFT, Throughput, Peak RAM")
    print(f"Target: 50-token generation")
    print("="*70)
    
    all_results = {}
    
    # Benchmark standard GGUF models (Q4_K_M, Q8_0)
    for model_key, model_path, display_name, n_ctx in models_to_benchmark:
        if os.path.exists(model_path):
            results = measure_inference_metrics(
                model_path=model_path,
                model_name=display_name,
                max_tokens=50,
                temperature=0.7,
                n_ctx=n_ctx,
                warmup_runs=1,
                measurement_runs=5,
            )
            if results:
                print_summary(results)
                all_results[model_key] = results
                
                # Save individual result
                output_file = f"/root/slm_eval_harness/reports/inference_metrics_{model_key}.json"
                save_results(results, output_file)
        else:
            print(f"\nSkipping {display_name}: Model not found at {model_path}")
    
    # Benchmark BF16 models (split GGUF files)
    for model_key, model_path, display_name in bf16_models:
        if os.path.exists(model_path):
            # BF16 needs larger context
            results = measure_inference_metrics(
                model_path=model_path,
                model_name=display_name,
                max_tokens=50,
                temperature=0.7,
                n_ctx=32768,  # BF16 requires larger context
                warmup_runs=1,
                measurement_runs=5,
            )
            if results:
                print_summary(results)
                all_results[model_key] = results
                
                # Save individual result
                output_file = f"/root/slm_eval_harness/reports/inference_metrics_{model_key}.json"
                save_results(results, output_file)
        else:
            print(f"\nSkipping {display_name}: Model not found at {model_path}")
    
    # Save combined results
    combined_output = "/root/slm_eval_harness/reports/inference_benchmarks_all.json"
    with open(combined_output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nAll results saved to: {combined_output}")
    
    return all_results


if __name__ == "__main__":
    main()
