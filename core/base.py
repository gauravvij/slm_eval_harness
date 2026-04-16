"""Abstract base classes for the evaluation harness."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json


class TaskType(Enum):
    """Types of evaluation tasks."""
    CODING = "coding"
    REASONING = "reasoning"
    FUNCTION_CALL = "function_call"
    MULTIPLE_CHOICE = "multiple_choice"


@dataclass
class TaskConfig:
    """Configuration for a task."""
    name: str
    task_type: TaskType
    dataset_path: str
    metric: str
    parser: str
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    num_samples: Optional[int] = None
    subset: Optional[str] = None
    split: str = "test"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert TaskConfig to dictionary."""
        return {
            "name": self.name,
            "task_type": self.task_type.value if isinstance(self.task_type, TaskType) else self.task_type,
            "dataset_path": self.dataset_path,
            "metric": self.metric,
            "parser": self.parser,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_samples": self.num_samples,
            "subset": self.subset,
            "split": self.split,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskConfig":
        """Create TaskConfig from dictionary."""
        task_type = TaskType(data.get("task_type", "coding"))
        dataset = data.get("dataset", {})
        evaluation = data.get("evaluation", {})
        parsing = data.get("parsing", {})
        generation = data.get("generation", {})
        
        return cls(
            name=data["name"],
            task_type=task_type,
            dataset_path=dataset.get("path", ""),
            metric=evaluation.get("metric", "exact_match"),
            parser=parsing.get("parser", "robust"),
            max_tokens=generation.get("max_tokens", 512),
            temperature=generation.get("temperature", 0.0),
            top_p=generation.get("top_p", 1.0),
            num_samples=dataset.get("num_samples"),
            subset=dataset.get("subset"),
            split=dataset.get("split", "test"),
        )


@dataclass
class Sample:
    """A single evaluation sample."""
    id: str
    prompt: str
    reference: Any
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "reference": self.reference,
            "metadata": self.metadata,
        }


@dataclass
class GenerationResult:
    """Result from model generation."""
    text: str
    tokens_generated: int
    generation_time: float
    finish_reason: Optional[str] = None
    logprobs: Optional[List[float]] = None


@dataclass
class ParsedResult:
    """Result after parsing model output."""
    content: Any
    confidence: float
    parser_used: str
    raw_output: str
    refusal_detected: bool = False
    error: Optional[str] = None
    
    def is_valid(self) -> bool:
        """Check if result is valid (not a refusal and has content)."""
        return not self.refusal_detected and self.content is not None and self.error is None


@dataclass
class EvaluationResult:
    """Result of evaluating a single sample."""
    sample_id: str
    parsed_result: ParsedResult
    reference: Any
    score: float
    passed: bool
    execution_time: float
    error: Optional[str] = None
    diagnostics: Optional[Dict[str, Any]] = None


class Task(ABC):
    """Abstract base class for evaluation tasks."""
    
    def __init__(self, config: TaskConfig):
        self.config = config
        self.name = config.name
        self.task_type = config.task_type
    
    @abstractmethod
    def get_prompt_template(self, sample: Sample) -> str:
        """Get the prompt template for a sample."""
        pass
    
    @abstractmethod
    def evaluate_prediction(self, prediction: Any, reference: Any) -> Tuple[float, bool]:
        """Evaluate a prediction against reference. Returns (score, passed)."""
        pass


class ModelAdapter(ABC):
    """Abstract base class for model adapters."""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> GenerationResult:
        """Generate text from the model."""
        pass
    
    @abstractmethod
    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> List[GenerationResult]:
        """Generate text for a batch of prompts."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model is available."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        pass


class Parser(ABC):
    """Abstract base class for output parsers."""
    
    name: str = "base"
    
    @abstractmethod
    def parse(self, text: str, **kwargs) -> ParsedResult:
        """Parse model output."""
        pass
    
    def get_confidence(self, text: str, parsed: Any) -> float:
        """Calculate confidence score for parsing."""
        return 1.0


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""
    
    def __init__(self, config: TaskConfig):
        self.config = config
    
    @abstractmethod
    def load(self) -> List[Sample]:
        """Load the dataset and return samples."""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get the schema of the dataset."""
        pass


class Evaluator(ABC):
    """Abstract base class for evaluators."""
    
    def __init__(
        self,
        task: Task,
        model_adapter: ModelAdapter,
        parser: Parser,
        dataset_loader: DatasetLoader,
    ):
        self.task = task
        self.model_adapter = model_adapter
        self.parser = parser
        self.dataset_loader = dataset_loader
    
    @abstractmethod
    def evaluate(
        self,
        samples: Optional[List[Sample]] = None,
        checkpoint_path: Optional[str] = None,
    ) -> List[EvaluationResult]:
        """Run evaluation on samples."""
        pass
    
    @abstractmethod
    def compute_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Compute aggregate metrics from results."""
        pass
