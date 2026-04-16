"""Dashboard reporter with ASCII summary and diagnostics."""

import json
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

from core.base import EvaluationResult
from metrics.diagnostic import DiagnosticReport


@dataclass
class DashboardConfig:
    """Configuration for dashboard reporting."""
    show_ascii_art: bool = True
    show_diagnostics: bool = True
    show_recommendations: bool = True
    max_width: int = 80


class DashboardReporter:
    """
    ASCII dashboard reporter with diagnostic analysis.
    
    Features:
    - ASCII art summary
    - Diagnostic breakdown
    - Recommendations
    - Export to various formats
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
    
    def generate(self, results: List[EvaluationResult], diagnostic: DiagnosticReport) -> str:
        """Generate full dashboard report."""
        lines = []
        
        if self.config.show_ascii_art:
            lines.extend(self._generate_header())
        
        lines.extend(self._generate_summary(results, diagnostic))
        lines.extend(self._generate_metrics(diagnostic))
        
        if self.config.show_diagnostics:
            lines.extend(self._generate_diagnostics(diagnostic))
        
        if self.config.show_recommendations:
            lines.extend(self._generate_recommendations(diagnostic))
        
        lines.extend(self._generate_footer())
        
        return "\n".join(lines)
    
    def _generate_header(self) -> List[str]:
        """Generate ASCII art header."""
        header = [
            "╔" + "═" * (self.config.max_width - 2) + "╗",
            "║" + " " * (self.config.max_width - 2) + "║",
            "║" + "SLM EVALUATION HARNESS".center(self.config.max_width - 2) + "║",
            "║" + "Results Dashboard".center(self.config.max_width - 2) + "║",
            "║" + " " * (self.config.max_width - 2) + "║",
            "╠" + "═" * (self.config.max_width - 2) + "╣",
        ]
        return header
    
    def _generate_summary(
        self,
        results: List[EvaluationResult],
        diagnostic: DiagnosticReport
    ) -> List[str]:
        """Generate summary section."""
        lines = [
            "║ SUMMARY".ljust(self.config.max_width - 1) + "║",
            "╠" + "─" * (self.config.max_width - 2) + "╣",
        ]
        
        # Key metrics
        metrics = [
            ("Total Samples", diagnostic.total_samples),
            ("Passed", diagnostic.passed),
            ("Failed", diagnostic.failed),
            ("Pass Rate", f"{diagnostic.pass_rate:.2%}"),
        ]
        
        for label, value in metrics:
            line = f"║  {label}: {value}".ljust(self.config.max_width - 1) + "║"
            lines.append(line)
        
        lines.append("╠" + "─" * (self.config.max_width - 2) + "╣")
        return lines
    
    def _generate_metrics(self, diagnostic: DiagnosticReport) -> List[str]:
        """Generate metrics section."""
        lines = [
            "║ ERROR ANALYSIS".ljust(self.config.max_width - 1) + "║",
            "╠" + "─" * (self.config.max_width - 2) + "╣",
        ]
        
        # Error patterns
        if diagnostic.error_patterns:
            lines.append(f"║  Error Patterns:".ljust(self.config.max_width - 1) + "║")
            for pattern, count in sorted(diagnostic.error_patterns.items(), key=lambda x: -x[1]):
                line = f"║    • {pattern}: {count}".ljust(self.config.max_width - 1) + "║"
                lines.append(line)
        
        # Error categories
        if diagnostic.error_categories:
            lines.append(f"║  Error Categories:".ljust(self.config.max_width - 1) + "║")
            for category, count in sorted(diagnostic.error_categories.items(), key=lambda x: -x[1]):
                line = f"║    • {category}: {count}".ljust(self.config.max_width - 1) + "║"
                lines.append(line)
        
        lines.append("╠" + "─" * (self.config.max_width - 2) + "╣")
        return lines
    
    def _generate_diagnostics(self, diagnostic: DiagnosticReport) -> List[str]:
        """Generate diagnostics section."""
        lines = [
            "║ CATEGORY PERFORMANCE".ljust(self.config.max_width - 1) + "║",
            "╠" + "─" * (self.config.max_width - 2) + "╣",
        ]
        
        if diagnostic.category_performance:
            for category, stats in diagnostic.category_performance.items():
                accuracy = stats.get("accuracy", 0.0)
                total = stats.get("total", 0)
                passed = stats.get("passed", 0)
                
                # Performance indicator
                if accuracy >= 0.8:
                    indicator = "✓"
                elif accuracy >= 0.5:
                    indicator = "~"
                else:
                    indicator = "✗"
                
                line = f"║  {indicator} {category}: {accuracy:.1%} ({passed}/{total})".ljust(self.config.max_width - 1) + "║"
                lines.append(line)
        else:
            lines.append(f"║  No category data available".ljust(self.config.max_width - 1) + "║")
        
        lines.append("╠" + "─" * (self.config.max_width - 2) + "╣")
        return lines
    
    def _generate_recommendations(self, diagnostic: DiagnosticReport) -> List[str]:
        """Generate recommendations section."""
        lines = [
            "║ RECOMMENDATIONS".ljust(self.config.max_width - 1) + "║",
            "╠" + "─" * (self.config.max_width - 2) + "╣",
        ]
        
        for i, rec in enumerate(diagnostic.recommendations, 1):
            # Wrap long recommendations
            wrapped = self._wrap_text(rec, self.config.max_width - 6)
            for j, line in enumerate(wrapped):
                prefix = f"{i}." if j == 0 else "  "
                formatted = f"║  {prefix} {line}".ljust(self.config.max_width - 1) + "║"
                lines.append(formatted)
        
        lines.append("╠" + "─" * (self.config.max_width - 2) + "╣")
        return lines
    
    def _generate_footer(self) -> List[str]:
        """Generate footer."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return [
            f"║ Generated: {timestamp}".ljust(self.config.max_width - 1) + "║",
            "╚" + "═" * (self.config.max_width - 2) + "╝",
        ]
    
    def _wrap_text(self, text: str, width: int) -> List[str]:
        """Wrap text to specified width."""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= width:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(" ".join(current_line))
        
        return lines
    
    def print_report(self, results: List[EvaluationResult], diagnostic: DiagnosticReport):
        """Print report to console."""
        report = self.generate(results, diagnostic)
        print(report)
    
    def save_report(
        self,
        results: List[EvaluationResult],
        diagnostic: DiagnosticReport,
        filepath: str
    ):
        """Save report to file."""
        report = self.generate(results, diagnostic)
        with open(filepath, 'w') as f:
            f.write(report)
    
    def export_json(
        self,
        results: List[EvaluationResult],
        diagnostic: DiagnosticReport,
        filepath: str
    ):
        """Export results to JSON."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "diagnostic": diagnostic.to_dict(),
            "results": [
                {
                    "sample_id": r.sample_id,
                    "passed": r.passed,
                    "score": r.score,
                    "execution_time": r.execution_time,
                    "error": r.error,
                }
                for r in results
            ],
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def export_markdown(
        self,
        results: List[EvaluationResult],
        diagnostic: DiagnosticReport,
        filepath: str
    ):
        """Export results to Markdown."""
        lines = [
            "# SLM Evaluation Results",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"- **Total Samples**: {diagnostic.total_samples}",
            f"- **Passed**: {diagnostic.passed}",
            f"- **Failed**: {diagnostic.failed}",
            f"- **Pass Rate**: {diagnostic.pass_rate:.2%}",
            "",
            "## Error Analysis",
            "",
        ]
        
        if diagnostic.error_patterns:
            lines.append("### Error Patterns")
            lines.append("")
            for pattern, count in sorted(diagnostic.error_patterns.items(), key=lambda x: -x[1]):
                lines.append(f"- {pattern}: {count}")
            lines.append("")
        
        if diagnostic.error_categories:
            lines.append("### Error Categories")
            lines.append("")
            for category, count in sorted(diagnostic.error_categories.items(), key=lambda x: -x[1]):
                lines.append(f"- {category}: {count}")
            lines.append("")
        
        if diagnostic.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for i, rec in enumerate(diagnostic.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")
        
        with open(filepath, 'w') as f:
            f.write("\n".join(lines))
