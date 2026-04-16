"""Main evaluator with batch processing and progress tracking."""

import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from tqdm import tqdm

from core.base import (
    Task, TaskConfig, Sample, EvaluationResult,
    ModelAdapter, Parser, DatasetLoader
)
from core.checkpoint import CheckpointManager
from execution.sandbox import SandboxExecutor
from parsers.refusal_detector import RefusalDetector


@dataclass
class EvaluationConfig:
    """Configuration for evaluation run."""
    batch_size: int = 1
    max_workers: int = 1
    checkpoint_interval: int = 10
    enable_checkpoints: bool = True
    timeout: float = 30.0
    verbose: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
            "checkpoint_interval": self.checkpoint_interval,
            "enable_checkpoints": self.enable_checkpoints,
            "timeout": self.timeout,
            "verbose": self.verbose,
        }


class Evaluator:
    """
    Main evaluator for running benchmarks.
    
    Features:
    - Batch processing with progress tracking
    - Checkpoint/resume support
    - Error handling and recovery
    - Result aggregation
    """
    
    def __init__(
        self,
        task: Task,
        model_adapter: ModelAdapter,
        parser: Parser,
        dataset_loader: DatasetLoader,
        config: Optional[EvaluationConfig] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ):
        self.task = task
        self.model_adapter = model_adapter
        self.parser = parser
        self.dataset_loader = dataset_loader
        self.config = config or EvaluationConfig()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        
        self._refusal_detector = RefusalDetector()
        self._sandbox = SandboxExecutor(timeout=self.config.timeout)
    
    def evaluate(
        self,
        samples: Optional[List[Sample]] = None,
        resume_from_checkpoint: bool = True,
    ) -> List[EvaluationResult]:
        """
        Run evaluation on samples.
        
        Args:
            samples: Samples to evaluate (loads from dataset if None)
            resume_from_checkpoint: Whether to resume from checkpoint
        
        Returns:
            List of evaluation results
        """
        # Load samples if not provided
        if samples is None:
            samples = self.dataset_loader.load()
        
        if not samples:
            print("Warning: No samples to evaluate")
            return []
        
        # Check for checkpoint
        completed_indices = set()
        results = []
        
        if resume_from_checkpoint and self.config.enable_checkpoints:
            checkpoint = self.checkpoint_manager.load(
                self.task.config.name,
                self.model_adapter.model_name
            )
            if checkpoint:
                print(f"Resuming from checkpoint: {len(checkpoint.completed_indices)} samples completed")
                completed_indices = set(checkpoint.completed_indices)
                results = self._restore_results_from_checkpoint(checkpoint.results)
        
        # Filter out completed samples
        remaining_samples = [
            (i, s) for i, s in enumerate(samples)
            if i not in completed_indices
        ]
        
        if not remaining_samples:
            print("All samples already evaluated")
            return results
        
        # Run evaluation
        print(f"Evaluating {len(remaining_samples)} samples...")
        
        with tqdm(total=len(samples), initial=len(completed_indices)) as pbar:
            for idx, sample in remaining_samples:
                try:
                    result = self._evaluate_single(sample)
                    results.append(result)
                    completed_indices.add(idx)
                    
                    # Update progress
                    pbar.update(1)
                    pbar.set_postfix({
                        "passed": sum(1 for r in results if r.passed),
                        "failed": sum(1 for r in results if not r.passed),
                    })
                    
                    # Save checkpoint periodically
                    if (len(completed_indices) % self.config.checkpoint_interval == 0 and
                        self.config.enable_checkpoints):
                        self._save_checkpoint(completed_indices, results)
                        
                except Exception as e:
                    print(f"\nError evaluating sample {sample.id}: {e}")
                    # Create error result
                    error_result = EvaluationResult(
                        sample_id=sample.id,
                        parsed_result=None,
                        reference=sample.reference,
                        score=0.0,
                        passed=False,
                        execution_time=0.0,
                        error=str(e),
                    )
                    results.append(error_result)
                    completed_indices.add(idx)
                    pbar.update(1)
        
        # Final checkpoint save
        if self.config.enable_checkpoints:
            self._save_checkpoint(completed_indices, results)
        
        return results
    
    def _evaluate_single(self, sample: Sample) -> EvaluationResult:
        """Evaluate a single sample."""
        start_time = time.time()
        
        # Get prompt
        prompt = self.task.get_prompt_template(sample)
        
        # Generate
        generation_result = self.model_adapter.generate(
            prompt=prompt,
            max_tokens=self.task.config.max_tokens,
            temperature=self.task.config.temperature,
            top_p=self.task.config.top_p,
        )
        
        # Parse output
        parsed_result = self.parser.parse(generation_result.text)
        
        # Check for refusal
        if self._refusal_detector.is_refusal(generation_result.text):
            parsed_result.refusal_detected = True
        
        # Evaluate
        score, passed = self.task.evaluate_prediction(
            parsed_result.content,
            sample.reference,
            sample
        )
        
        execution_time = time.time() - start_time
        
        return EvaluationResult(
            sample_id=sample.id,
            parsed_result=parsed_result,
            reference=sample.reference,
            score=score,
            passed=passed,
            execution_time=execution_time,
            error=None,
            diagnostics={
                "generation_time": generation_result.generation_time,
                "tokens_generated": generation_result.tokens_generated,
                "finish_reason": generation_result.finish_reason,
            }
        )
    
    def _save_checkpoint(self, completed_indices: set, results: List[EvaluationResult]):
        """Save checkpoint."""
        try:
            self.checkpoint_manager.save(
                task_name=self.task.config.name,
                model_name=self.model_adapter.model_name,
                completed_indices=list(completed_indices),
                results=results,
                metadata={
                    "config": self.task.config.to_dict(),
                    "eval_config": self.config.to_dict(),
                }
            )
        except Exception as e:
            if self.config.verbose:
                print(f"Warning: Failed to save checkpoint: {e}")
    
    def _restore_results_from_checkpoint(
        self,
        checkpoint_results: List[Dict]
    ) -> List[EvaluationResult]:
        """Restore results from checkpoint data."""
        results = []
        for data in checkpoint_results:
            try:
                from .base import ParsedResult
                
                parsed_data = data.get("parsed_result", {})
                parsed_result = ParsedResult(
                    content=parsed_data.get("content"),
                    confidence=parsed_data.get("confidence", 0.0),
                    parser_used=parsed_data.get("parser_used", "unknown"),
                    raw_output=parsed_data.get("raw_output", ""),
                    refusal_detected=parsed_data.get("refusal_detected", False),
                    error=parsed_data.get("error"),
                )
                
                result = EvaluationResult(
                    sample_id=data["sample_id"],
                    parsed_result=parsed_result,
                    reference=data.get("reference"),
                    score=data.get("score", 0.0),
                    passed=data.get("passed", False),
                    execution_time=data.get("execution_time", 0.0),
                    error=data.get("error"),
                    diagnostics=data.get("diagnostics"),
                )
                results.append(result)
            except Exception as e:
                if self.config.verbose:
                    print(f"Warning: Failed to restore result: {e}")
                continue
        
        return results
    
    def compute_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Compute aggregate metrics from results."""
        if not results:
            return {"accuracy": 0.0, "total": 0.0, "passed": 0.0}
        
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        
        metrics = {
            "accuracy": passed / total if total > 0 else 0.0,
            "total": float(total),
            "passed": float(passed),
            "failed": float(total - passed),
            "avg_execution_time": sum(r.execution_time for r in results) / total if total > 0 else 0.0,
        }
        
        return metrics
    
    def get_summary(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Get evaluation summary."""
        metrics = self.compute_metrics(results)
        
        # Count refusals
        refusals = sum(
            1 for r in results
            if r.parsed_result and r.parsed_result.refusal_detected
        )
        
        # Count parsing errors
        parsing_errors = sum(
            1 for r in results
            if r.parsed_result and r.parsed_result.error
        )
        
        summary = {
            "task": self.task.config.name,
            "model": self.model_adapter.model_name,
            "metrics": metrics,
            "refusals": refusals,
            "parsing_errors": parsing_errors,
            "total_samples": len(results),
        }
        
        return summary
