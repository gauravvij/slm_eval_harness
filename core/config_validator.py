"""JSON Schema validation for task YAML configurations."""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from jsonschema import validate, ValidationError, Draft7Validator


# JSON Schema for task configuration validation
TASK_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["name", "task_type", "dataset", "evaluation"],
    "properties": {
        "name": {
            "type": "string",
            "description": "Unique name for the task"
        },
        "task_type": {
            "type": "string",
            "enum": ["coding", "reasoning", "function_call", "multiple_choice"],
            "description": "Type of evaluation task"
        },
        "description": {
            "type": "string",
            "description": "Human-readable description of the task"
        },
        "dataset": {
            "type": "object",
            "required": ["path", "loader"],
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to dataset (HuggingFace dataset ID or local path)"
                },
                "loader": {
                    "type": "string",
                    "enum": ["humaneval", "hellaswag", "bfcl", "gsm8k", "generic"],
                    "description": "Dataset loader to use"
                },
                "split": {
                    "type": "string",
                    "default": "test",
                    "description": "Dataset split to use"
                },
                "subset": {
                    "type": ["string", "null"],
                    "description": "Subset of the dataset (e.g., 'Python' for MBPP)"
                },
                "num_samples": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "description": "Number of samples to evaluate (null for all)"
                }
            }
        },
        "prompt": {
            "type": "object",
            "properties": {
                "template": {
                    "type": "string",
                    "description": "Prompt template with {placeholder} syntax"
                },
                "system_message": {
                    "type": ["string", "null"],
                    "description": "System message for chat models"
                },
                "few_shot": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "string"},
                            "output": {"type": "string"}
                        }
                    }
                }
            }
        },
        "generation": {
            "type": "object",
            "properties": {
                "max_tokens": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 512
                },
                "temperature": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 2,
                    "default": 0.0
                },
                "top_p": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 1.0
                },
                "stop_sequences": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "num_return_sequences": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1
                }
            }
        },
        "parsing": {
            "type": "object",
            "required": ["parser"],
            "properties": {
                "parser": {
                    "type": "string",
                    "enum": ["code", "mc", "json", "robust"],
                    "description": "Parser to use for model output"
                },
                "extraction_mode": {
                    "type": "string",
                    "enum": ["last_block", "first_block", "all"],
                    "default": "last_block"
                },
                "language": {
                    "type": ["string", "null"],
                    "description": "Programming language for code extraction"
                }
            }
        },
        "evaluation": {
            "type": "object",
            "required": ["metric"],
            "properties": {
                "metric": {
                    "type": "string",
                    "enum": ["pass_at_k", "exact_match", "multiple_choice", "json_validity", "composite"],
                    "description": "Metric to use for evaluation"
                },
                "k": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1,
                    "description": "K for pass@k metric"
                },
                "timeout": {
                    "type": "number",
                    "minimum": 1,
                    "default": 5.0,
                    "description": "Timeout for code execution in seconds"
                },
                "sandbox": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to use sandboxed execution"
                }
            }
        },
        "metadata": {
            "type": "object",
            "properties": {
                "version": {"type": "string"},
                "author": {"type": "string"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
    }
}


class ConfigValidator:
    """Validator for task configuration files."""
    
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """Initialize validator with schema."""
        self.schema = schema or TASK_SCHEMA
        self.validator = Draft7Validator(self.schema)
    
    def validate_file(self, filepath: str) -> Tuple[bool, List[str]]:
        """
        Validate a YAML configuration file.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        try:
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                return False, ["File is empty or invalid YAML"]
            
            return self.validate_dict(config)
            
        except yaml.YAMLError as e:
            return False, [f"YAML parsing error: {str(e)}"]
        except FileNotFoundError:
            return False, [f"File not found: {filepath}"]
        except Exception as e:
            return False, [f"Unexpected error: {str(e)}"]
    
    def validate_dict(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a configuration dictionary.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required fields
        required = self.schema.get("required", [])
        for field in required:
            if field not in config:
                errors.append(f"Missing required field: '{field}'")
        
        # Validate against schema
        try:
            validate(instance=config, schema=self.schema)
        except ValidationError as e:
            errors.append(f"Schema validation error: {e.message} at {list(e.path)}")
        
        # Custom validations
        errors.extend(self._custom_validations(config))
        
        return len(errors) == 0, errors
    
    def _custom_validations(self, config: Dict[str, Any]) -> List[str]:
        """Perform custom validations beyond schema."""
        errors = []
        
        # Validate task-specific requirements
        task_type = config.get("task_type")
        parsing = config.get("parsing", {})
        evaluation = config.get("evaluation", {})
        
        if task_type == "coding":
            if parsing.get("parser") not in ["code", "robust"]:
                errors.append(
                    f"Coding tasks should use 'code' or 'robust' parser, "
                    f"got '{parsing.get('parser')}'"
                )
            if evaluation.get("metric") not in ["pass_at_k", "exact_match"]:
                errors.append(
                    f"Coding tasks should use 'pass_at_k' or 'exact_match' metric, "
                    f"got '{evaluation.get('metric')}'"
                )
        
        elif task_type == "multiple_choice":
            if parsing.get("parser") not in ["mc", "robust"]:
                errors.append(
                    f"Multiple choice tasks should use 'mc' or 'robust' parser, "
                    f"got '{parsing.get('parser')}'"
                )
            if evaluation.get("metric") != "multiple_choice":
                errors.append(
                    f"Multiple choice tasks should use 'multiple_choice' metric, "
                    f"got '{evaluation.get('metric')}'"
                )
        
        elif task_type == "function_call":
            if parsing.get("parser") not in ["json", "robust"]:
                errors.append(
                    f"Function call tasks should use 'json' or 'robust' parser, "
                    f"got '{parsing.get('parser')}'"
                )
        
        # Validate generation parameters
        generation = config.get("generation", {})
        if generation.get("temperature", 0) > 0 and evaluation.get("metric") == "pass_at_k":
            # This is a warning-level issue, not an error
            pass
        
        return errors
    
    def validate_batch(self, directory: str) -> Dict[str, Tuple[bool, List[str]]]:
        """
        Validate all YAML files in a directory.
        
        Returns:
            Dictionary mapping filenames to (is_valid, errors)
        """
        results = {}
        path = Path(directory)
        
        if not path.exists():
            return {"error": (False, [f"Directory not found: {directory}"])}
        
        for yaml_file in path.glob("**/*.yaml"):
            rel_path = str(yaml_file.relative_to(path))
            is_valid, errors = self.validate_file(str(yaml_file))
            results[rel_path] = (is_valid, errors)
        
        return results
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the current schema."""
        return self.schema
    
    def validate_model_args(self, args_str: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate model arguments string.
        
        Args:
            args_str: Comma-separated key=value pairs (e.g., "pretrained=model,device=cuda")
        
        Returns:
            Tuple of (is_valid, parsed_args_dict)
        """
        try:
            args = {}
            for pair in args_str.split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    args[key.strip()] = value.strip()
                elif pair.strip():
                    args[pair.strip()] = True
            return True, args
        except Exception as e:
            return False, {"error": str(e)}


def validate_task_config(filepath: str) -> Tuple[bool, List[str]]:
    """Convenience function to validate a task config file."""
    validator = ConfigValidator()
    return validator.validate_file(filepath)
