#!/usr/bin/env python3
"""
Inference Metrics Measurement Script for Qwen3.6-35B-A3B-GGUF BF16

Measures:
- TTFT (Time To First Token): Time from prompt submission to first token generation
- Throughput (tokens/sec): Total tokens generated divided by total generation time

Hardware Context: 32-core CPU, 125GB RAM, no GPU
"""

import os
import sys
import time
import json
import statistics
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adapters.gguf_adapter import GGUFAdapter
from core.base import GenerationResult


# Test prompts of varying complexity for comprehensive measurement
TEST_PROMPTS = [
    # Short prompt - simple query
    "What is the capital of France?",
    
    # Medium prompt - coding task
    """Write a Python function to calculate the factorial of a number.

Example:
def factorial(n):
    # Your code here
    pass

print(factorial(5))  # Should output 120""",
    
    # Longer prompt - reasoning task
    """Explain the concept of recursion in programming. Include:
1. A definition
2. A simple example
3. When to use it vs iteration
4. Common pitfalls

Be concise but thorough.""",
    
    # Code completion prompt
    """Complete the following function:

def binary_search(arr, target):
    '''
    Perform binary search on a sorted array.
    Returns the index of target if found, -1 otherwise.
    '''
    left, right = 0, len(arr) - 1
    
    while left <= right:
        # TODO: Complete the implementation
        pass
    
    return -1""",
    
    # Multi-step reasoning
    """Solve this step by step:
If a train travels 120 km in 2 hours, and another train travels 
in the opposite direction at 80 km/h, how far apart will they 
be after 3 hours?""",
]


def measure_ttft_and_throughput(
    model_path: str,
    prompts: List[str],
    max_tokens: int = 256,
    temperature: float = 0.7,
    n_ctx: int = 4096,
    warmup_runs: int = 1,
    measurement_runs: int = 3,
) -> Dict[str, Any]:
    """
    Measure TTFT (Time To First Token) and throughput for a GGUF model.
    
    Args:
        model_path: Path to the GGUF model file (first shard for multi-part)
        prompts: List of test prompts
        max_tokens: Maximum tokens to generate per prompt
        temperature: Sampling temperature
        n_ctx: Context window size
        warmup_runs: Number of warmup runs before measurement
        measurement_runs: Number of measurement runs per prompt
    
    Returns:
        Dictionary with TTFT and throughput metrics
    """
    print(f"Loading model: {model_path}")
    print(f"Hardware: 32-core CPU, 125GB RAM, no GPU")
    print(f"Configuration: n_ctx={n_ctx}, max_tokens={max_tokens}")
    print("-" * 60)
    
    # Initialize adapter (CPU-only, no GPU layers)
    adapter = GGUFAdapter(
        model_name=model_path,
        device="cpu",
        n_ctx=n_ctx,
        n_gpu_layers=0,  # CPU only
        verbose=False,
    )
    
    print(f"Model loaded successfully!")
    print(f"Running {warmup_runs} warmup(s) followed by {measurement_runs} measurement(s) per prompt")
    print("-" * 60)
    
    # Storage for all measurements
    all_ttft_ms = []
    all_throughput = []
    all_total_time = []
    all_tokens_generated = []
    
    prompt_results = []
    
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\nPrompt {prompt_idx + 1}/{len(prompts)}:")
        print(f"  Length: {len(prompt)} chars")
        print(f"  Preview: {prompt[:60]}...")
        
        prompt_ttft = []
        prompt_throughput = []
        prompt_tokens = []
        prompt_times = []
        
        # Warmup runs
        for w in range(warmup_runs):
            print(f"    Warmup {w + 1}/{warmup_runs}...", end=" ", flush=True)
            _ = adapter.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            print("done")
        
        # Measurement runs
        for r in range(measurement_runs):
            print(f"    Measurement {r + 1}/{measurement_runs}...", end=" ", flush=True)
            
            # Measure TTFT and total time using streaming
            start_time = time.perf_counter()
            
            # Use streaming to capture first token time
            stream = adapter._model.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            
            # Process streaming output to capture TTFT
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
            ttft_ms = (first_token_time - start_time) * 1000  # Convert to milliseconds
            total_time_s = end_time - start_time
            
            # More accurate token count using model tokenizer
            if adapter._model:
                token_count = len(adapter._model.tokenize(full_text.encode('utf-8')))
            elif token_count == 0 and full_text:
                token_count = len(full_text) // 4  # Rough estimate fallback
            
            throughput = token_count / total_time_s if total_time_s > 0 else 0
            
            prompt_ttft.append(ttft_ms)
            prompt_throughput.append(throughput)
            prompt_tokens.append(token_count)
            prompt_times.append(total_time_s)
            
            print(f"TTFT={ttft_ms:.2f}ms, Throughput={throughput:.2f}tok/s, Tokens={token_count}")
        
        # Calculate statistics for this prompt
        avg_ttft = statistics.mean(prompt_ttft)
        std_ttft = statistics.stdev(prompt_ttft) if len(prompt_ttft) > 1 else 0
        avg_throughput = statistics.mean(prompt_throughput)
        std_throughput = statistics.stdev(prompt_throughput) if len(prompt_throughput) > 1 else 0
        avg_tokens = statistics.mean(prompt_tokens)
        
        prompt_results.append({
            "prompt_index": prompt_idx,
            "prompt_length": len(prompt),
            "avg_ttft_ms": round(avg_ttft, 2),
            "std_ttft_ms": round(std_ttft, 2),
            "avg_throughput": round(avg_throughput, 2),
            "std_throughput": round(std_throughput, 2),
            "avg_tokens_generated": round(avg_tokens, 1),
            "raw_measurements": {
                "ttft_ms": prompt_ttft,
                "throughput": prompt_throughput,
                "tokens": prompt_tokens,
                "total_time_s": prompt_times,
            }
        })
        
        all_ttft_ms.extend(prompt_ttft)
        all_throughput.extend(prompt_throughput)
        all_tokens_generated.extend(prompt_tokens)
        all_total_time.extend(prompt_times)
    
    # Calculate overall statistics
    overall_avg_ttft = statistics.mean(all_ttft_ms)
    overall_std_ttft = statistics.stdev(all_ttft_ms) if len(all_ttft_ms) > 1 else 0
    overall_avg_throughput = statistics.mean(all_throughput)
    overall_std_throughput = statistics.stdev(all_throughput) if len(all_throughput) > 1 else 0
    overall_total_tokens = sum(all_tokens_generated)
    overall_total_time_s = sum(all_total_time)
    
    results = {
        "metadata": {
            "model": "Qwen3.6-35B-A3B-GGUF BF16",
            "model_path": model_path,
            "timestamp": datetime.now().isoformat(),
            "hardware": {
                "cpu_cores": 32,
                "ram_gb": 125,
                "gpu": False,
            },
            "configuration": {
                "n_ctx": n_ctx,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "n_gpu_layers": 0,
            },
            "measurement_params": {
                "warmup_runs": warmup_runs,
                "measurement_runs": measurement_runs,
                "num_prompts": len(prompts),
            }
        },
        "overall_metrics": {
            "ttft_ms": {
                "mean": round(overall_avg_ttft, 2),
                "std": round(overall_std_ttft, 2),
                "min": round(min(all_ttft_ms), 2),
                "max": round(max(all_ttft_ms), 2),
            },
            "throughput_tokens_per_sec": {
                "mean": round(overall_avg_throughput, 2),
                "std": round(overall_std_throughput, 2),
                "min": round(min(all_throughput), 2),
                "max": round(max(all_throughput), 2),
            },
            "total_tokens_generated": overall_total_tokens,
            "total_time_seconds": round(overall_total_time_s, 2),
        },
        "per_prompt_results": prompt_results,
    }
    
    return results


def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def print_summary(results: Dict[str, Any]):
    """Print a formatted summary of results."""
    print("\n" + "=" * 60)
    print("INFERENCE METRICS SUMMARY - BF16")
    print("=" * 60)
    
    meta = results["metadata"]
    print(f"\nModel: {meta['model']}")
    print(f"Hardware: {meta['hardware']['cpu_cores']}-core CPU, {meta['hardware']['ram_gb']}GB RAM")
    print(f"Timestamp: {meta['timestamp']}")
    
    overall = results["overall_metrics"]
    print(f"\n--- OVERALL METRICS ---")
    print(f"\nTTFT (Time To First Token):")
    print(f"  Mean: {overall['ttft_ms']['mean']:.2f} ms")
    print(f"  Std:  {overall['ttft_ms']['std']:.2f} ms")
    print(f"  Range: {overall['ttft_ms']['min']:.2f} - {overall['ttft_ms']['max']:.2f} ms")
    
    print(f"\nThroughput:")
    print(f"  Mean: {overall['throughput_tokens_per_sec']['mean']:.2f} tokens/sec")
    print(f"  Std:  {overall['throughput_tokens_per_sec']['std']:.2f} tokens/sec")
    print(f"  Range: {overall['throughput_tokens_per_sec']['min']:.2f} - {overall['throughput_tokens_per_sec']['max']:.2f} tokens/sec")
    
    print(f"\nTotal Statistics:")
    print(f"  Total tokens generated: {overall['total_tokens_generated']}")
    print(f"  Total time: {overall['total_time_seconds']:.2f} seconds")
    
    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    # Model path - use the first BF16 shard (llama-cpp handles multi-shard)
    model_path = "/root/slm_eval_harness/BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf"
    
    # Output path
    output_path = "/root/slm_eval_harness/reports/inference_metrics_bf16.json"
    
    print("=" * 60)
    print("INFERENCE METRICS MEASUREMENT")
    print("Qwen3.6-35B-A3B-GGUF BF16 on CPU")
    print("=" * 60)
    
    # Run measurements
    results = measure_ttft_and_throughput(
        model_path=model_path,
        prompts=TEST_PROMPTS,
        max_tokens=256,
        temperature=0.7,
        n_ctx=4096,
        warmup_runs=1,
        measurement_runs=3,
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    save_results(results, output_path)
    
    return results


if __name__ == "__main__":
    main()
