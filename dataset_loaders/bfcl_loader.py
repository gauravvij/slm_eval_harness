"""BFCL (Berkeley Function Calling Leaderboard) dataset loader with JSON schema handling."""

import json
from typing import Any, Dict, List, Optional
from datasets import load_dataset

from .base_loader import BaseDatasetLoader
from core.base import Sample, TaskConfig
from core.registry import register_dataset_loader


@register_dataset_loader("bfcl")
class BFCLLoader(BaseDatasetLoader):
    """
    Loader for BFCL function calling benchmark.
    
    The BFCL dataset has multiple files with different schemas:
    - BFCL_v3_simple.json: Basic function calling
    - BFCL_v3_multiple.json: Multiple function calls
    - BFCL_v3_parallel.json: Parallel function calls
    - etc.
    
    Dataset fields (varies by file):
    - id: Unique identifier
    - question: User query (can be string or nested conversation format)
    - function: Function schema(s)
    - answer: Expected function call(s) (in possible_answer/ subdirectory)
    """
    
    # Map of subset names to file paths
    BFCL_SUBSETS = {
        "simple": "BFCL_v3_simple.json",
        "multiple": "BFCL_v3_multiple.json",
        "parallel": "BFCL_v3_parallel.json",
        "parallel_multiple": "BFCL_v3_parallel_multiple.json",
        "exec_simple": "BFCL_v3_exec_simple.json",
        "exec_multiple": "BFCL_v3_exec_multiple.json",
        "exec_parallel": "BFCL_v3_exec_parallel.json",
        "exec_parallel_multiple": "BFCL_v3_exec_parallel_multiple.json",
        "irrelevance": "BFCL_v3_irrelevance.json",
        "live_simple": "BFCL_v3_live_simple.json",
        "live_multiple": "BFCL_v3_live_multiple.json",
        "live_parallel": "BFCL_v3_live_parallel.json",
        "live_parallel_multiple": "BFCL_v3_live_parallel_multiple.json",
        "live_relevance": "BFCL_v3_live_relevance.json",
        "live_irrelevance": "BFCL_v3_live_irrelevance.json",
        "java": "BFCL_v3_java.json",
        "javascript": "BFCL_v3_javascript.json",
        "rest": "BFCL_v3_rest.json",
        "sql": "BFCL_v3_sql.json",
        "chatable": "BFCL_v3_chatable.json",
    }
    
    def _load_dataset(self):
        """Load BFCL dataset from HuggingFace with specific file handling.
        
        Loads both question data and ground truth from possible_answer/ subdirectory.
        """
        path = self.config.dataset_path
        subset = self.config.subset
        
        # Determine which subset to load
        if subset and subset in self.BFCL_SUBSETS:
            filename = self.BFCL_SUBSETS[subset]
        elif subset == "tiny" or subset == "mini":
            filename = self.BFCL_SUBSETS.get("simple", "BFCL_v3_simple.json")
        else:
            filename = self.BFCL_SUBSETS.get("simple", "BFCL_v3_simple.json")
        
        # Load question data
        data_url = f"hf://datasets/{path}/{filename}"
        try:
            question_dataset = load_dataset('json', data_files=data_url, split='train')
        except Exception as e:
            raise RuntimeError(f"Failed to load BFCL questions from {data_url}: {e}")
        
        # Load ground truth from possible_answer subdirectory
        answer_url = f"hf://datasets/{path}/possible_answer/{filename}"
        try:
            answer_dataset = load_dataset('json', data_files=answer_url, split='train')
            # Create a mapping from id to ground_truth
            answer_map = {item['id']: item.get('ground_truth', []) for item in answer_dataset}
        except Exception as e:
            # If possible_answer doesn't exist, use empty ground truth
            answer_map = {}
        
        # Merge question data with ground truth
        merged_data = []
        for item in question_dataset:
            item_id = item.get('id')
            if item_id in answer_map:
                item['ground_truth'] = answer_map[item_id]
            else:
                item['ground_truth'] = []
            merged_data.append(item)
        
        return merged_data
    
    def _get_field_mapping(self) -> Dict[str, str]:
        """Map standard fields to BFCL fields."""
        return {
            "prompt": "question",
            "question": "question",
            "reference": "answer",
            "function": "function",
            "functions": "function",
            "id": "id",
        }
    
    def _extract_sample(self, item: Dict[str, Any], index: int) -> Sample:
        """Extract a BFCL sample with flexible question format handling."""
        # Get ID
        sample_id = item.get("id", f"bfcl_{index}")
        
        # Get question - handle both simple string and nested conversation format
        question = item.get("question", "")
        
        # BFCL question can be:
        # 1. A simple string
        # 2. A nested list: [[{"role": "user", "content": "..."}]]
        # 3. A list of conversation turns
        if isinstance(question, list):
            # Extract text from nested conversation format
            if len(question) > 0:
                if isinstance(question[0], list) and len(question[0]) > 0:
                    # Format: [[{"role": "user", "content": "..."}]]
                    turn = question[0][0]
                    if isinstance(turn, dict):
                        question = turn.get("content", "")
                    else:
                        question = str(turn)
                elif isinstance(question[0], dict):
                    # Format: [{"role": "user", "content": "..."}]
                    question = question[0].get("content", "")
                else:
                    # Format: ["question text"]
                    question = str(question[0])
            else:
                question = ""
        
        # Get function schema(s)
        functions = item.get("function", [])
        if isinstance(functions, dict):
            functions = [functions]
        elif isinstance(functions, str):
            # Sometimes function is an empty string
            if not functions.strip():
                functions = []
            else:
                try:
                    functions = json.loads(functions)
                    if isinstance(functions, dict):
                        functions = [functions]
                except:
                    functions = []
        
        # Get expected answer (ground_truth from possible_answer/)
        answer = item.get("ground_truth", [])
        if isinstance(answer, dict):
            answer = [answer]
        
        # Build prompt
        prompt = self._build_prompt(question, functions)
        
        # Build metadata
        metadata = {
            "question": question,
            "functions": functions,
            "chat_history": item.get("chat_history", []),
            "function_call_type": item.get("function_call_type", "simple"),
        }
        
        return Sample(
            id=sample_id,
            prompt=prompt,
            reference=answer,
            metadata=metadata,
        )
    
    def _build_prompt(self, question: str, functions: List[Dict]) -> str:
        """
        Build the function calling prompt.
        
        Args:
            question: User query
            functions: List of function schemas
        
        Returns:
            Formatted prompt
        """
        # Format function schemas
        func_descriptions = []
        for func in functions:
            if isinstance(func, dict):
                name = func.get("name", "unknown")
                description = func.get("description", "")
                params = func.get("parameters", {})
                
                func_desc = f"Function: {name}\n"
                func_desc += f"Description: {description}\n"
                func_desc += f"Parameters: {json.dumps(params, indent=2)}\n"
                func_descriptions.append(func_desc)
        
        prompt = f"""You are a helpful assistant with access to functions. 

Available functions:
{chr(10).join(func_descriptions)}

User query: {question}

Respond with a JSON array of function calls in the format:
[{{"name": "function_name", "arguments": {{"arg1": "value1"}}}}]

Response:"""
        
        return prompt
    
    def _get_prompt(self, item: Dict[str, Any]) -> str:
        """Get the formatted prompt."""
        question = item.get("question", "")
        functions = item.get("function", [])
        if isinstance(functions, dict):
            functions = [functions]
        return self._build_prompt(question, functions)
    
    def _get_reference(self, item: Dict[str, Any]) -> List[Dict]:
        """Get the expected function calls."""
        answer = item.get("answer", [])
        if isinstance(answer, dict):
            return [answer]
        return answer if isinstance(answer, list) else []
    
    def get_functions(self, item: Dict[str, Any]) -> List[Dict]:
        """Get the function schemas."""
        functions = item.get("function", [])
        if isinstance(functions, dict):
            return [functions]
        return functions if isinstance(functions, list) else []
    
    def get_function_call_type(self, item: Dict[str, Any]) -> str:
        """Get the function call type (simple, multiple, parallel, etc.)."""
        return item.get("function_call_type", "simple")
    
    def validate_function_call(self, call: Dict, functions: List[Dict]) -> bool:
        """
        Validate a function call against the function schema.
        
        Args:
            call: Function call dict with 'name' and 'arguments'
            functions: List of available function schemas
        
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(call, dict):
            return False
        
        func_name = call.get("name")
        arguments = call.get("arguments", {})
        
        # Find matching function schema
        func_schema = None
        for func in functions:
            if func.get("name") == func_name:
                func_schema = func
                break
        
        if func_schema is None:
            return False
        
        # Validate parameters
        params = func_schema.get("parameters", {})
        required = params.get("required", [])
        properties = params.get("properties", {})
        
        # Check required parameters
        for req_param in required:
            if req_param not in arguments:
                return False
        
        # Check parameter types (basic validation)
        for arg_name, arg_value in arguments.items():
            if arg_name in properties:
                expected_type = properties[arg_name].get("type")
                if expected_type == "string" and not isinstance(arg_value, str):
                    return False
                elif expected_type == "integer" and not isinstance(arg_value, int):
                    return False
                elif expected_type == "number" and not isinstance(arg_value, (int, float)):
                    return False
                elif expected_type == "boolean" and not isinstance(arg_value, bool):
                    return False
        
        return True
