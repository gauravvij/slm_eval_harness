"""Console reporter with rich output and progress bars."""

import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeRemainingColumn
)
from rich.tree import Tree
from rich import box

from core.base import EvaluationResult


@dataclass
class ReportConfig:
    """Configuration for console reporting."""
    show_progress: bool = True
    show_details: bool = False
    show_errors: bool = True
    color: bool = True
    compact: bool = False


class ConsoleReporter:
    """
    Rich console reporter for evaluation results.
    
    Features:
    - Progress bars with rich formatting
    - Detailed result tables
    - Error highlighting
    - Summary panels
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self.console = Console(color_system="auto" if self.config.color else None)
        self._progress = None
    
    def print_header(self, title: str, subtitle: Optional[str] = None):
        """Print a formatted header."""
        content = f"[bold blue]{title}[/bold blue]"
        if subtitle:
            content += f"\n[dim]{subtitle}[/dim]"
        
        self.console.print(Panel.fit(content, border_style="blue"))
    
    def print_task_info(self, task_name: str, model_name: str, num_samples: int):
        """Print task information."""
        table = Table(show_header=False, box=box.SIMPLE)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Task", task_name)
        table.add_row("Model", model_name)
        table.add_row("Samples", str(num_samples))
        
        self.console.print(table)
    
    def create_progress(self, total: int, description: str = "Evaluating") -> Progress:
        """Create a rich progress bar."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            "•",
            TimeRemainingColumn(),
            console=self.console,
        )
        return progress
    
    def update_progress(self, progress: Progress, task_id: int, advance: int = 1, **kwargs):
        """Update progress bar."""
        progress.update(task_id, advance=advance, **kwargs)
    
    def print_result_summary(self, results: List[EvaluationResult]):
        """Print summary of evaluation results."""
        if not results:
            self.console.print("[yellow]No results to display[/yellow]")
            return
        
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        accuracy = passed / total if total > 0 else 0.0
        
        # Summary panel
        summary = f"""
[bold]Total Samples:[/bold] {total}
[bold]Passed:[/bold] [green]{passed}[/green]
[bold]Failed:[/bold] [red]{failed}[/red]
[bold]Accuracy:[/bold] {accuracy:.2%}
"""
        
        # Color based on performance
        if accuracy >= 0.8:
            border_style = "green"
        elif accuracy >= 0.5:
            border_style = "yellow"
        else:
            border_style = "red"
        
        self.console.print(Panel(summary, title="Results Summary", border_style=border_style))
    
    def print_detailed_results(self, results: List[EvaluationResult], limit: int = 10):
        """Print detailed results table."""
        if not results or not self.config.show_details:
            return
        
        table = Table(
            title="Detailed Results",
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED
        )
        
        table.add_column("Sample ID", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Score", justify="right")
        table.add_column("Time (s)", justify="right")
        table.add_column("Parser", style="dim")
        
        for result in results[:limit]:
            status = "[green]✓[/green]" if result.passed else "[red]✗[/red]"
            parser = result.parsed_result.parser_used if result.parsed_result else "N/A"
            
            table.add_row(
                result.sample_id[:30],
                status,
                f"{result.score:.2f}",
                f"{result.execution_time:.2f}",
                parser[:20]
            )
        
        if len(results) > limit:
            table.add_row(
                f"... and {len(results) - limit} more",
                "", "", "", "",
                style="dim"
            )
        
        self.console.print(table)
    
    def print_errors(self, results: List[EvaluationResult], limit: int = 5):
        """Print error details."""
        if not self.config.show_errors:
            return
        
        errors = [r for r in results if r.error or (r.parsed_result and r.parsed_result.error)]
        
        if not errors:
            return
        
        self.console.print(f"\n[bold red]Errors ({len(errors)}):[/bold red]")
        
        for i, result in enumerate(errors[:limit]):
            error_msg = result.error or (result.parsed_result.error if result.parsed_result else "Unknown")
            
            self.console.print(Panel(
                f"[bold]{result.sample_id}[/bold]\n{error_msg[:200]}",
                border_style="red"
            ))
        
        if len(errors) > limit:
            self.console.print(f"[dim]... and {len(errors) - limit} more errors[/dim]")
    
    def print_metrics(self, metrics: Dict[str, float]):
        """Print metrics table."""
        table = Table(title="Metrics", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        for key, value in metrics.items():
            # Format value based on type
            if isinstance(value, float):
                if "accuracy" in key.lower() or "rate" in key.lower():
                    formatted = f"{value:.2%}"
                else:
                    formatted = f"{value:.4f}"
            else:
                formatted = str(value)
            
            table.add_row(key, formatted)
        
        self.console.print(table)
    
    def print_comparison(self, results_dict: Dict[str, Dict[str, float]]):
        """Print comparison of multiple tasks."""
        if not results_dict:
            return
        
        # Get all metric keys
        all_metrics = set()
        for metrics in results_dict.values():
            all_metrics.update(metrics.keys())
        
        table = Table(title="Task Comparison", show_header=True, header_style="bold magenta")
        table.add_column("Task", style="cyan")
        
        for metric in sorted(all_metrics):
            table.add_column(metric, justify="right")
        
        for task_name, metrics in results_dict.items():
            row = [task_name]
            for metric in sorted(all_metrics):
                value = metrics.get(metric, "N/A")
                if isinstance(value, float):
                    if "accuracy" in metric.lower():
                        row.append(f"{value:.2%}")
                    else:
                        row.append(f"{value:.4f}")
                else:
                    row.append(str(value))
            table.add_row(*row)
        
        self.console.print(table)
    
    def print_tree(self, data: Dict[str, Any], title: Optional[str] = None):
        """Print data as a tree."""
        tree = Tree(f"[bold]{title or 'Results'}[/bold]")
        
        def add_branch(node, data):
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (dict, list)):
                        branch = node.add(f"[cyan]{key}[/cyan]")
                        add_branch(branch, value)
                    else:
                        node.add(f"[dim]{key}:[/dim] {value}")
            elif isinstance(data, list):
                for i, item in enumerate(data[:10]):  # Limit to 10 items
                    if isinstance(item, (dict, list)):
                        branch = node.add(f"[dim][{i}][/dim]")
                        add_branch(branch, item)
                    else:
                        node.add(f"[dim][{i}]:[/dim] {item}")
                if len(data) > 10:
                    node.add(f"[dim]... and {len(data) - 10} more[/dim]")
        
        add_branch(tree, data)
        self.console.print(tree)
    
    def print_success(self, message: str):
        """Print success message."""
        self.console.print(f"[green]✓ {message}[/green]")
    
    def print_warning(self, message: str):
        """Print warning message."""
        self.console.print(f"[yellow]⚠ {message}[/yellow]")
    
    def print_error(self, message: str):
        """Print error message."""
        self.console.print(f"[red]✗ {message}[/red]")
    
    def print_info(self, message: str):
        """Print info message."""
        self.console.print(f"[blue]ℹ {message}[/blue]")
    
    def print_divider(self, char: str = "─", width: int = 80):
        """Print a divider line."""
        self.console.print(char * width, style="dim")
