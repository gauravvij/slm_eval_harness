"""
HellaSwag test using OpenRouter API to verify implementation correctness.

This test:
1. Loads HellaSwag dataset using existing loader
2. Uses MC parser for answer extraction
3. Evaluates 20 samples via OpenRouter API
4. Reports accuracy and raw outputs to diagnose 0% accuracy issue

Usage:
    python tests/test_hellaswag_openrouter.py --model google/gemma-4-26b-it
    python tests/test_hellaswag_openrouter.py --model qwen/qwen-3.5-9b
"""

import argparse
import json
import sys
import os
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.openrouter_adapter import OpenRouterAdapter
from dataset_loaders.hellaswag_loader import HellaSwagLoader
from parsers.mc_parser import MultipleChoiceParser
from core.base import TaskConfig, TaskType, Sample, EvaluationResult


class SimpleHellaSwagTask:
    """Simple HellaSwag task wrapper for testing."""
    
    def __init__(self, config: TaskConfig):
        self.config = config
        self.name = config.name
        self.task_type = config.task_type
    
    def get_prompt_template(self, sample: Sample) -> str:
        """Return the prompt as-is (already formatted by loader)."""
        return sample.prompt
    
    def evaluate_prediction(self, prediction: Any, reference: Any, sample: Sample = None) -> tuple:
        """
        Evaluate prediction against reference.
        
        Returns:
            (score, passed) tuple
        """
        if prediction is None:
            return 0.0, False
        
        # Normalize both to uppercase strings
        pred_str = str(prediction).upper().strip()
        ref_str = str(reference).upper().strip()
        
        # Extract just the letter if there's extra text
        pred_letter = pred_str[0] if pred_str else ''
        ref_letter = ref_str[0] if ref_str else ''
        
        passed = pred_letter == ref_letter and pred_letter in 'ABCD'
        score = 1.0 if passed else 0.0
        
        return score, passed


def run_hellaswag_test(model_name: str, num_samples: int = 20) -> Dict[str, Any]:
    """
    Run HellaSwag test with OpenRouter.
    
    Args:
        model_name: OpenRouter model identifier (e.g., "google/gemma-4-26b-it")
        num_samples: Number of samples to evaluate
    
    Returns:
        Dictionary with results and diagnostics
    """
    print(f"\n{'='*70}")
    print(f"HellaSwag Test via OpenRouter")
    print(f"Model: {model_name}")
    print(f"Samples: {num_samples}")
    print(f"{'='*70}\n")
    
    # Create task config
    config = TaskConfig(
        name="hellaswag_openrouter_test",
        task_type=TaskType.MULTIPLE_CHOICE,
        dataset_path="Rowan/hellaswag",
        metric="multiple_choice",
        parser="mc",
        max_tokens=512,  # Increased from 32 to allow full completions
        temperature=0.0,
        top_p=1.0,
        num_samples=num_samples,
        subset=None,
        split="validation"
    )
    
    # Initialize components
    print("Loading dataset...")
    loader = HellaSwagLoader(config)
    samples = loader.load()
    
    # Limit to num_samples
    samples = samples[:num_samples]
    print(f"Loaded {len(samples)} samples\n")
    
    print("Initializing OpenRouter adapter...")
    try:
        adapter = OpenRouterAdapter(model_name=model_name)
        print(f"✅ Adapter initialized successfully\n")
    except Exception as e:
        print(f"❌ Failed to initialize adapter: {e}")
        return {"error": str(e)}
    
    parser = MultipleChoiceParser()
    task = SimpleHellaSwagTask(config)
    
    # Run evaluation
    results = []
    detailed_results = []
    
    print(f"Evaluating {len(samples)} samples...\n")
    
    for i, sample in enumerate(samples):
        print(f"Sample {i+1}/{len(samples)}: {sample.id}")
        
        try:
            # Get generation
            prompt = task.get_prompt_template(sample)
            generation = adapter.generate(
                prompt=prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
            )
            
            # Parse output
            parsed = parser.parse(generation.text)
            
            # Evaluate
            score, passed = task.evaluate_prediction(
                parsed.content,
                sample.reference,
                sample
            )
            
            # Store detailed result
            detailed_result = {
                "sample_id": sample.id,
                "reference": sample.reference,
                "prediction": parsed.content,
                "raw_output": generation.text,
                "finish_reason": generation.finish_reason,
                "tokens_generated": generation.tokens_generated,
                "passed": passed,
                "score": score,
            }
            detailed_results.append(detailed_result)
            results.append(detailed_result)
            
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status} | Ref: {sample.reference} | Pred: {parsed.content} | "
                  f"Finish: {generation.finish_reason}")
            
            # Show truncation warning
            if generation.finish_reason == 'length':
                print(f"  ⚠️  WARNING: Output truncated due to length limit!")
            
            # Show raw output preview (first 100 chars)
            raw_preview = generation.text[:100].replace('\n', ' ')
            print(f"  Raw: {raw_preview}...")
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            detailed_results.append({
                "sample_id": sample.id,
                "reference": sample.reference,
                "prediction": None,
                "raw_output": None,
                "error": str(e),
                "passed": False,
                "score": 0.0,
            })
        
        print()
    
    # Calculate metrics
    total = len(detailed_results)
    passed_count = sum(1 for r in detailed_results if r.get("passed", False))
    failed_count = total - passed_count
    accuracy = (passed_count / total * 100) if total > 0 else 0.0
    
    # Check for truncation
    truncated_count = sum(1 for r in detailed_results 
                          if r.get("finish_reason") == "length")
    
    # Summary
    print(f"{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Model: {model_name}")
    print(f"Total Samples: {total}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Truncated Outputs: {truncated_count}")
    print(f"{'='*70}\n")
    
    # Analysis
    print("DETAILED ANALYSIS:")
    print("-" * 70)
    
    if accuracy == 0.0:
        print("⚠️  0% ACCURACY DETECTED")
        print("\nPossible causes:")
        
        if truncated_count > 0:
            print(f"  1. Output truncation: {truncated_count}/{total} samples hit token limit")
            print("     → Model outputs are being cut off before completing the answer")
        
        # Check if predictions are empty or None
        empty_preds = sum(1 for r in detailed_results 
                         if not r.get("prediction"))
        if empty_preds > 0:
            print(f"  2. Empty predictions: {empty_preds}/{total} samples have no parsed answer")
        
        # Check raw outputs
        print("\n  Sample raw outputs (first 3 failures):")
        failures = [r for r in detailed_results if not r.get("passed", False)]
        for i, r in enumerate(failures[:3]):
            print(f"\n  Sample {r['sample_id']}:")
            print(f"    Reference: {r['reference']}")
            print(f"    Parsed: {r.get('prediction', 'N/A')}")
            print(f"    Raw: {r.get('raw_output', 'N/A')[:200]}...")
            print(f"    Finish reason: {r.get('finish_reason', 'N/A')}")
    
    else:
        print(f"✅ Accuracy: {accuracy:.2f}%")
        
        # Show some correct predictions
        correct = [r for r in detailed_results if r.get("passed", False)]
        if correct:
            print(f"\nSample correct predictions:")
            for r in correct[:3]:
                print(f"  {r['sample_id']}: Ref={r['reference']}, Pred={r['prediction']}")
        
        # Show some incorrect predictions
        incorrect = [r for r in detailed_results if not r.get("passed", False)]
        if incorrect:
            print(f"\nSample incorrect predictions:")
            for r in incorrect[:3]:
                print(f"  {r['sample_id']}: Ref={r['reference']}, Pred={r['prediction']}")
    
    print(f"\n{'='*70}\n")
    
    return {
        "model": model_name,
        "total_samples": total,
        "passed": passed_count,
        "failed": failed_count,
        "accuracy": accuracy,
        "truncated_count": truncated_count,
        "results": detailed_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test HellaSwag via OpenRouter API"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen/qwen3.5-9b",
        help="OpenRouter model identifier (default: qwen/qwen3.5-9b)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of samples to evaluate (default: 20)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results"
    )
    
    args = parser.parse_args()
    
    # Run test
    results = run_hellaswag_test(args.model, args.samples)
    
    # Save results if output path specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")
    
    # Return exit code based on success
    if "error" in results:
        sys.exit(1)
    
    # Exit with error if 0% accuracy (for CI/CD detection)
    if results.get("accuracy", 0) == 0.0:
        print("⚠️  Exiting with error code due to 0% accuracy")
        sys.exit(2)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
