#!/usr/bin/env /usr/bin/python3
"""
Visualization Script: Combine Inference Metrics with Benchmark Accuracies

Creates a bar chart showing:
- TTFT (Time To First Token) in ms
- Throughput (tokens/sec)
- HumanEval accuracy (%)
- HellaSwag accuracy (%)
- BFCL accuracy (%)

Hardware Context: 32-core CPU, 125GB RAM, no GPU
Model: Qwen3.6-35B-A3B-GGUF Q4_K_M
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_inference_metrics(path: str) -> dict:
    """Load inference metrics from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def create_combined_visualization(
    inference_metrics_path: str,
    output_path: str,
    humaneval_acc: float = 47.56,
    hellaswag_acc: float = 74.30,
    bfcl_acc: float = 46.00,
):
    """
    Create a bar chart combining inference metrics with benchmark accuracies.
    """
    # Load inference metrics
    metrics = load_inference_metrics(inference_metrics_path)
    
    # Extract key metrics
    ttft_ms = metrics['overall_metrics']['ttft_ms']['mean']
    throughput = metrics['overall_metrics']['throughput_tokens_per_sec']['mean']
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        'Qwen3.6-35B-A3B-GGUF Q4_K_M Performance Metrics\n'
        '32-core CPU | 125GB RAM | No GPU',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    # === Subplot 1: Inference Performance Metrics ===
    inference_labels = ['TTFT\n(ms)', 'Throughput\n(tok/s)']
    inference_values = [ttft_ms, throughput]
    inference_colors = ['#2E86AB', '#A23B72']
    
    bars1 = ax1.bar(inference_labels, inference_values, color=inference_colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax1.set_title('Inference Performance', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, value in zip(bars1, inference_values):
        height = bar.get_height()
        if value == ttft_ms:
            label = f'{value:.1f} ms'
        else:
            label = f'{value:.1f} tok/s'
        ax1.text(
            bar.get_x() + bar.get_width() / 2.,
            height + max(inference_values) * 0.02,
            label,
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    # Add TTFT range annotation
    ttft_min = metrics['overall_metrics']['ttft_ms']['min']
    ttft_max = metrics['overall_metrics']['ttft_ms']['max']
    ax1.annotate(
        f'Range: {ttft_min:.1f} - {ttft_max:.1f} ms',
        xy=(0, ttft_ms),
        xytext=(0.3, ttft_ms * 1.3),
        fontsize=8,
        color='#555555',
        arrowprops=dict(arrowstyle='->', color='#555555', lw=0.8)
    )
    
    # === Subplot 2: Benchmark Accuracies ===
    benchmark_labels = ['HumanEval', 'HellaSwag', 'BFCL']
    benchmark_values = [humaneval_acc, hellaswag_acc, bfcl_acc]
    benchmark_colors = ['#06A77D', '#F4A261', '#E76F51']
    
    bars2 = ax2.bar(benchmark_labels, benchmark_values, color=benchmark_colors, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Benchmark Accuracies', fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, value in zip(bars2, benchmark_values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 2,
            f'{value:.2f}%',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    # Add horizontal line at 50% for reference
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax2.text(2.15, 51, '50% baseline', fontsize=8, color='gray', va='bottom')
    
    # === Hardware Context Annotation ===
    hardware_text = (
        "Hardware Context:\n"
        "• 32-core CPU\n"
        "• 125GB RAM\n"
        "• No GPU (CPU-only inference)\n"
        "• llama-cpp-python 0.3.20"
    )
    fig.text(
        0.5, 0.02,
        hardware_text,
        ha='center',
        fontsize=9,
        style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    )
    
    # === Legend ===
    legend_elements = [
        mpatches.Patch(color='#2E86AB', label=f'TTFT: {ttft_ms:.1f} ms'),
        mpatches.Patch(color='#A23B72', label=f'Throughput: {throughput:.1f} tok/s'),
        mpatches.Patch(color='#06A77D', label=f'HumanEval: {humaneval_acc:.2f}%'),
        mpatches.Patch(color='#F4A261', label=f'HellaSwag: {hellaswag_acc:.2f}%'),
        mpatches.Patch(color='#E76F51', label=f'BFCL: {bfcl_acc:.2f}%'),
    ]
    fig.legend(
        handles=legend_elements,
        loc='upper right',
        bbox_to_anchor=(0.98, 0.92),
        fontsize=9,
        framealpha=0.9
    )
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to: {output_path}")
    
    # Also create a summary JSON
    summary = {
        "model": "Qwen3.6-35B-A3B-GGUF Q4_K_M",
        "timestamp": datetime.now().isoformat(),
        "hardware": {
            "cpu_cores": 32,
            "ram_gb": 125,
            "gpu": False
        },
        "inference_metrics": {
            "ttft_ms": round(ttft_ms, 2),
            "throughput_tokens_per_sec": round(throughput, 2)
        },
        "benchmark_accuracies": {
            "humaneval": humaneval_acc,
            "hellaswag": hellaswag_acc,
            "bfcl": bfcl_acc
        }
    }
    
    summary_path = output_path.replace('.png', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")
    
    plt.close()
    return summary


def main():
    """Main entry point."""
    # Paths
    inference_metrics_path = "/root/slm_eval_harness/reports/inference_metrics.json"
    output_path = "/root/slm_eval_harness/reports/qwen3.6_35b_inference_benchmarks.png"
    
    # Existing benchmark accuracies from task specification
    humaneval_acc = 47.56
    hellaswag_acc = 74.30
    bfcl_acc = 46.00
    
    print("=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)
    print(f"\nInference Metrics: {inference_metrics_path}")
    print(f"Output: {output_path}")
    print(f"\nBenchmark Accuracies:")
    print(f"  - HumanEval: {humaneval_acc}%")
    print(f"  - HellaSwag: {hellaswag_acc}%")
    print(f"  - BFCL: {bfcl_acc}%")
    
    # Create visualization
    summary = create_combined_visualization(
        inference_metrics_path=inference_metrics_path,
        output_path=output_path,
        humaneval_acc=humaneval_acc,
        hellaswag_acc=hellaswag_acc,
        bfcl_acc=bfcl_acc,
    )
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"\nFinal Metrics Summary:")
    print(f"  TTFT: {summary['inference_metrics']['ttft_ms']} ms")
    print(f"  Throughput: {summary['inference_metrics']['throughput_tokens_per_sec']} tokens/sec")
    print(f"  HumanEval: {summary['benchmark_accuracies']['humaneval']}%")
    print(f"  HellaSwag: {summary['benchmark_accuracies']['hellaswag']}%")
    print(f"  BFCL: {summary['benchmark_accuracies']['bfcl']}%")


if __name__ == "__main__":
    main()
