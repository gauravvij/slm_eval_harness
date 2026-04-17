#!/usr/bin/env python3
"""Command-line interface for SLM Evaluation Harness."""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from core.config_validator import ConfigValidator
from core.registry import registry
from core.evaluator import Evaluator, EvaluationConfig
from core.checkpoint import CheckpointManager
from adapters.hf_adapter import HuggingFaceAdapter
from adapters.ollama_adapter import OllamaAdapter
from adapters.gguf_adapter import GGUFAdapter
from dataset_loaders.humaneval_loader import HumanEvalLoader
from dataset_loaders.hellaswag_loader import HellaSwagLoader
from dataset_loaders.bfcl_loader import BFCLLoader
from parsers.code_parser import CodeParser
from parsers.mc_parser import MultipleChoiceParser
from parsers.json_parser import JSONParser
from parsers.robust_parser import RobustParser
from metrics.pass_at_k import PassAtK
from metrics.exact_match import ExactMatch
from metrics.multiple_choice import MultipleChoiceAccuracy
from metrics.diagnostic import DiagnosticAnalyzer
from reporting.console import ConsoleReporter
from reporting.dashboard import DashboardReporter

console = Console()


def load_task_config(task_path: str) -> dict:
    """Load and validate task configuration."""
    import yaml
    
    with open(task_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate
    validator = ConfigValidator()
    is_valid, errors = validator.validate_dict(config)
    
    if not is_valid:
        console.print(f"[red]Task config validation failed:[/red]")
        for error in errors:
            console.print(f"  - {error}")
        sys.exit(1)
    
    return config


def create_adapter(model_type: str, model_name: str, model_args: dict):
    """Create model adapter."""
    if model_type == "hf":
        return HuggingFaceAdapter(model_name, **model_args)
    elif model_type == "ollama":
        return OllamaAdapter(model_name, **model_args)
    elif model_type == "gguf":
        return GGUFAdapter(model_name, **model_args)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_dataset_loader(config: dict):
    """Create dataset loader based on config."""
    loader_name = config['dataset']['loader']
    from core.base import TaskConfig
    task_config = TaskConfig.from_dict(config)
    
    if loader_name == "humaneval":
        return HumanEvalLoader(task_config)
    elif loader_name == "hellaswag":
        return HellaSwagLoader(task_config)
    elif loader_name == "bfcl":
        return BFCLLoader(task_config)
    else:
        raise ValueError(f"Unknown loader: {loader_name}")


def create_parser(config: dict):
    """Create parser based on config."""
    parser_name = config['parsing']['parser']
    
    if parser_name == "code":
        return CodeParser(
            language=config['parsing'].get('language'),
            extraction_mode=config['parsing'].get('extraction_mode', 'last_block')
        )
    elif parser_name == "mc":
        return MultipleChoiceParser()
    elif parser_name == "json":
        return JSONParser()
    elif parser_name == "robust":
        return RobustParser()
    else:
        raise ValueError(f"Unknown parser: {parser_name}")


def create_task(config: dict):
    """Create task from config."""
    from core.base import Task, TaskConfig, Sample
    from typing import Any, Tuple
    
    task_config = TaskConfig.from_dict(config)
    
    class GenericTask(Task):
        def __init__(self, task_config):
            super().__init__(task_config)
            self._sandbox = None
        
        def get_prompt_template(self, sample: Sample) -> str:
            template = config.get('prompt', {}).get('template', '{prompt}')
            return template.format(prompt=sample.prompt)
        
        def evaluate_prediction(self, prediction: Any, reference: Any, sample: Sample = None) -> Tuple[float, bool]:
            metric_name = config['evaluation']['metric']
            
            if metric_name == "pass_at_k":
                # For code, we need to execute using sandbox
                if sample is None:
                    # Fallback to string comparison if no sample metadata
                    passed = prediction == reference
                    return (1.0 if passed else 0.0, passed)
                
                # Import sandbox here to avoid circular imports
                from execution.sandbox import SandboxExecutor
                
                if self._sandbox is None:
                    self._sandbox = SandboxExecutor(timeout=5.0)
                
                # Get test code and entry point from sample metadata
                test_code = sample.metadata.get('test_code', '')
                entry_point = sample.metadata.get('entry_point', '')
                
                # Build the complete script: prediction (completion) + test code + check call
                code_to_execute = prediction if prediction else ""
                
                # HumanEval test code defines a check(candidate) function but doesn't call it
                # We need to add the call to actually run the tests
                if entry_point and 'def check(' in test_code:
                    test_code = test_code + f"\n\n# Run the check function\ncheck({entry_point})\n"
                
                # Execute in sandbox
                try:
                    result = self._sandbox.execute(
                        code=code_to_execute,
                        test_code=test_code,
                        entry_point=entry_point
                    )
                    passed = result.success
                    return (1.0 if passed else 0.0, passed)
                except Exception as e:
                    # Execution failed
                    return (0.0, False)
                    
            elif metric_name == "multiple_choice":
                # For MC, simple comparison
                pred_str = str(prediction).strip().upper() if prediction else ""
                ref_str = str(reference).strip().upper() if reference else ""
                passed = pred_str == ref_str
                return (1.0 if passed else 0.0, passed)
            elif metric_name == "json_validity":
                # For JSON, check validity
                import json
                try:
                    json.loads(prediction) if isinstance(prediction, str) else prediction
                    return (1.0, True)
                except:
                    return (0.0, False)
            elif metric_name == "function_call_match":
                # For function calling, compare predicted function calls against ground truth
                import json
                
                # Parse prediction
                try:
                    if isinstance(prediction, str):
                        pred_calls = json.loads(prediction)
                    else:
                        pred_calls = prediction
                    
                    # Ensure it's a list
                    if isinstance(pred_calls, dict):
                        pred_calls = [pred_calls]
                    if not isinstance(pred_calls, list):
                        return (0.0, False)
                except:
                    return (0.0, False)
                
                # Parse reference (ground truth)
                if not reference or not isinstance(reference, list):
                    return (0.0, False)
                
                # Ground truth format: [{'func_name': {'param1': [valid_values], ...}}]
                # Build a map of expected function calls
                expected_calls = {}
                for ref_item in reference:
                    if isinstance(ref_item, dict):
                        for func_name, params in ref_item.items():
                            expected_calls[func_name] = params
                
                if not expected_calls:
                    return (0.0, False)
                
                # Compare each predicted call against expected
                total_calls = len(pred_calls)
                correct_calls = 0
                
                for pred_call in pred_calls:
                    if not isinstance(pred_call, dict):
                        continue
                    
                    pred_func_name = pred_call.get('name', '')
                    pred_args = pred_call.get('arguments', {})
                    
                    # Check if function name matches
                    if pred_func_name not in expected_calls:
                        continue
                    
                    expected_params = expected_calls[pred_func_name]
                    
                    # Check each parameter
                    param_correct = True
                    for param_name, valid_values in expected_params.items():
                        if param_name not in pred_args:
                            param_correct = False
                            break
                        
                        pred_value = pred_args[param_name]
                        
                        # Handle different valid value formats
                        if isinstance(valid_values, list):
                            # Check if predicted value matches any valid value
                            if pred_value not in valid_values:
                                param_correct = False
                                break
                        elif pred_value != valid_values:
                            param_correct = False
                            break
                    
                    if param_correct:
                        correct_calls += 1
                
                # Score based on proportion of correct calls
                if total_calls > 0:
                    score = correct_calls / total_calls
                else:
                    score = 0.0
                
                # Passed if at least one call matches
                passed = correct_calls > 0
                return (score, passed)
            else:
                # Default exact match
                passed = prediction == reference
                return (1.0 if passed else 0.0, passed)
    
    return GenericTask(task_config)


def run_evaluation(args):
    """Run evaluation."""
    console.print(Panel.fit(
        f"[bold blue]SLM Evaluation Harness[/bold blue]\n"
        f"Model: {args.model_name}\n"
        f"Tasks: {', '.join(args.tasks)}"
    ))
    
    # Parse model args
    model_args = {}
    if args.model_args:
        for pair in args.model_args.split(','):
            if '=' in pair:
                k, v = pair.split('=', 1)
                model_args[k.strip()] = v.strip()
            else:
                model_args[pair.strip()] = True
    
    # Create adapter
    console.print(f"[yellow]Loading model: {args.model_name}...[/yellow]")
    try:
        adapter = create_adapter(args.model, args.model_name, model_args)
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        sys.exit(1)
    
    # Run tasks
    all_results = {}
    
    for task_file in args.tasks:
        task_path = Path(task_file)
        if not task_path.exists():
            # Try in tasks directory
            task_path = Path(__file__).parent / "tasks" / task_file
            if not task_path.exists():
                task_path = Path(__file__).parent / "tasks" / f"{task_file}.yaml"
        
        if not task_path.exists():
            console.print(f"[red]Task not found: {task_file}[/red]")
            continue
        
        console.print(f"\n[bold]Running task: {task_path.stem}[/bold]")
        
        # Load config
        config = load_task_config(str(task_path))
        
        # Create components
        dataset_loader = create_dataset_loader(config)
        parser = create_parser(config)
        task = create_task(config)
        
        # Create evaluator
        eval_config = EvaluationConfig(
            batch_size=args.batch_size,
            checkpoint_interval=args.checkpoint_interval,
            enable_checkpoints=not args.no_checkpoint,
            verbose=args.verbose,
        )
        
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=args.checkpoint_dir
        ) if not args.no_checkpoint else None
        
        evaluator = Evaluator(
            task=task,
            model_adapter=adapter,
            parser=parser,
            dataset_loader=dataset_loader,
            config=eval_config,
            checkpoint_manager=checkpoint_manager,
        )
        
        # Run evaluation
        try:
            results = evaluator.evaluate(resume_from_checkpoint=not args.no_checkpoint)
            
            # Compute metrics
            metrics = evaluator.compute_metrics(results)
            
            # Store results
            all_results[config['name']] = {
                'metrics': metrics,
                'results': [r.to_dict() if hasattr(r, 'to_dict') else str(r) for r in results],
            }
            
            # Print summary
            console.print(f"[green]Completed: {len(results)} samples[/green]")
            console.print(f"[green]Accuracy: {metrics['accuracy']:.2%}[/green]")
            
        except Exception as e:
            console.print(f"[red]Evaluation failed: {e}[/red]")
            import traceback
            if args.verbose:
                traceback.print_exc()
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        console.print(f"\n[green]Results saved to: {args.output}[/green]")
    
    # Print final summary
    console.print("\n[bold]Final Summary:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Task")
    table.add_column("Samples")
    table.add_column("Passed")
    table.add_column("Accuracy")
    
    for task_name, data in all_results.items():
        metrics = data['metrics']
        table.add_row(
            task_name,
            str(int(metrics['total'])),
            str(int(metrics['passed'])),
            f"{metrics['accuracy']:.2%}"
        )
    
    console.print(table)


def list_tasks(args):
    """List available tasks."""
    tasks_dir = Path(__file__).parent / "tasks"
    
    console.print("[bold]Available Tasks:[/bold]\n")
    
    for category in ["coding", "reasoning", "function_call"]:
        category_dir = tasks_dir / category
        if category_dir.exists():
            console.print(f"[bold cyan]{category.upper()}:[/bold cyan]")
            for task_file in sorted(category_dir.glob("*.yaml")):
                config = load_task_config(str(task_file))
                console.print(f"  - {task_file.stem}: {config.get('description', 'No description')[:60]}...")
            console.print()


def validate_tasks(args):
    """Validate task configurations."""
    validator = ConfigValidator()
    
    tasks_dir = Path(__file__).parent / "tasks"
    
    console.print("[bold]Validating Task Configurations...[/bold]\n")
    
    all_valid = True
    for task_file in tasks_dir.rglob("*.yaml"):
        is_valid, errors = validator.validate_file(str(task_file))
        
        if is_valid:
            console.print(f"[green]✓ {task_file.name}[/green]")
        else:
            console.print(f"[red]✗ {task_file.name}[/red]")
            for error in errors:
                console.print(f"    - {error}")
            all_valid = False
    
    if all_valid:
        console.print("\n[green]All tasks are valid![/green]")
    else:
        console.print("\n[red]Some tasks have validation errors.[/red]")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="SLM Evaluation Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run smoke test with Ollama
  python cli.py --model ollama --model_name gemma4:e4b --tasks humaneval_mini,hellaswag_mini,bfcl_tiny

  # Run full Core-3 suite
  python cli.py --model ollama --model_name gemma4:e4b --tasks humaneval,hellaswag,bfcl_mini

  # Run with HuggingFace model
  python cli.py --model hf --model_name meta-llama/Llama-3.2-1B-Instruct --tasks humaneval_mini

  # List available tasks
  python cli.py --list-tasks

  # Validate all task configs
  python cli.py --validate-tasks
        """
    )
    
    # Model arguments
    parser.add_argument(
        '--model', '-m',
        choices=['hf', 'ollama', 'gguf'],
        default='ollama',
        help='Model type (hf or ollama)'
    )
    parser.add_argument(
        '--model_name', '-n',
        required=True,
        help='Model name or path'
    )
    parser.add_argument(
        '--model_args',
        default='',
        help='Additional model arguments (key=value,key2=value2)'
    )
    
    # Task arguments
    parser.add_argument(
        '--tasks', '-t',
        required=False,
        help='Comma-separated list of task names or paths'
    )
    parser.add_argument(
        '--smoke-test', '-s',
        action='store_true',
        help='Run smoke test (humaneval_mini, hellaswag_mini, bfcl_tiny)'
    )
    
    # Evaluation arguments
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for generation'
    )
    parser.add_argument(
        '--checkpoint_interval',
        type=int,
        default=10,
        help='Save checkpoint every N samples'
    )
    parser.add_argument(
        '--checkpoint_dir',
        default='.checkpoints',
        help='Directory for checkpoints'
    )
    parser.add_argument(
        '--no-checkpoint',
        action='store_true',
        help='Disable checkpointing'
    )
    
    # Output arguments
    parser.add_argument(
        '--output', '-o',
        help='Output file for results (JSON)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    # Utility commands
    parser.add_argument(
        '--list-tasks',
        action='store_true',
        help='List available tasks'
    )
    parser.add_argument(
        '--validate-tasks',
        action='store_true',
        help='Validate all task configurations'
    )
    
    args = parser.parse_args()
    
    # Handle utility commands
    if args.list_tasks:
        list_tasks(args)
        return
    
    if args.validate_tasks:
        validate_tasks(args)
        return
    
    # Determine tasks to run
    if args.smoke_test:
        args.tasks = [
            "coding/humaneval_mini",
            "reasoning/hellaswag_mini",
            "function_call/bfcl_tiny"
        ]
    elif args.tasks:
        args.tasks = [t.strip() for t in args.tasks.split(',')]
    else:
        # Default to smoke test
        args.tasks = [
            "coding/humaneval_mini",
            "reasoning/hellaswag_mini",
            "function_call/bfcl_tiny"
        ]
    
    # Run evaluation
    run_evaluation(args)


if __name__ == '__main__':
    main()
