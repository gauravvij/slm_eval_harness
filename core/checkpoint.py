"""Checkpoint management with atomic writes and resume support."""

import json
import os
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import threading

from .base import EvaluationResult


@dataclass
class CheckpointState:
    """State saved in a checkpoint."""
    task_name: str
    model_name: str
    completed_indices: List[int]
    results: List[Dict[str, Any]]
    timestamp: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "model_name": self.model_name,
            "completed_indices": self.completed_indices,
            "results": self.results,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointState":
        return cls(
            task_name=data["task_name"],
            model_name=data["model_name"],
            completed_indices=data["completed_indices"],
            results=data["results"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {}),
        )


class CheckpointManager:
    """
    Manages checkpoints for resumable evaluation.
    
    Features:
    - Atomic writes (write to temp, then rename)
    - Automatic cleanup of old checkpoints
    - Thread-safe operations
    """
    
    def __init__(
        self,
        checkpoint_dir: str = ".checkpoints",
        max_checkpoints: int = 5,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self._lock = threading.Lock()
    
    def _get_checkpoint_path(self, task_name: str, model_name: str) -> Path:
        """Get the path for a checkpoint file."""
        # Sanitize names for filesystem
        safe_task = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_name)
        safe_model = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)
        return self.checkpoint_dir / f"{safe_task}_{safe_model}.json"
    
    def save(
        self,
        task_name: str,
        model_name: str,
        completed_indices: List[int],
        results: List[EvaluationResult],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a checkpoint atomically.
        
        Returns:
            Path to the saved checkpoint
        """
        with self._lock:
            checkpoint_path = self._get_checkpoint_path(task_name, model_name)
            
            # Convert results to dicts
            result_dicts = []
            for r in results:
                result_dicts.append({
                    "sample_id": r.sample_id,
                    "parsed_result": {
                        "content": r.parsed_result.content,
                        "confidence": r.parsed_result.confidence,
                        "parser_used": r.parsed_result.parser_used,
                        "raw_output": r.parsed_result.raw_output,
                        "refusal_detected": r.parsed_result.refusal_detected,
                        "error": r.parsed_result.error,
                    },
                    "reference": r.reference if hasattr(r, 'reference') else None,
                    "score": r.score,
                    "passed": r.passed,
                    "execution_time": r.execution_time,
                    "error": r.error,
                    "diagnostics": r.diagnostics,
                })
            
            state = CheckpointState(
                task_name=task_name,
                model_name=model_name,
                completed_indices=completed_indices,
                results=result_dicts,
                timestamp=datetime.now().isoformat(),
                metadata=metadata or {},
            )
            
            # Atomic write: write to temp file, then rename
            temp_fd, temp_path = tempfile.mkstemp(
                dir=self.checkpoint_dir,
                suffix=".tmp"
            )
            try:
                with os.fdopen(temp_fd, 'w') as f:
                    json.dump(state.to_dict(), f, indent=2)
                shutil.move(temp_path, checkpoint_path)
            except Exception:
                # Clean up temp file on error
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            return str(checkpoint_path)
    
    def load(
        self,
        task_name: str,
        model_name: str
    ) -> Optional[CheckpointState]:
        """
        Load a checkpoint if it exists.
        
        Returns:
            CheckpointState or None if no checkpoint exists
        """
        checkpoint_path = self._get_checkpoint_path(task_name, model_name)
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            return CheckpointState.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load checkpoint {checkpoint_path}: {e}")
            return None
    
    def exists(self, task_name: str, model_name: str) -> bool:
        """Check if a checkpoint exists."""
        checkpoint_path = self._get_checkpoint_path(task_name, model_name)
        return checkpoint_path.exists()
    
    def delete(self, task_name: str, model_name: str) -> bool:
        """
        Delete a checkpoint.
        
        Returns:
            True if checkpoint was deleted, False if it didn't exist
        """
        checkpoint_path = self._get_checkpoint_path(task_name, model_name)
        
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            return True
        return False
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                checkpoints.append({
                    "path": str(checkpoint_file),
                    "task_name": data.get("task_name", "unknown"),
                    "model_name": data.get("model_name", "unknown"),
                    "timestamp": data.get("timestamp", "unknown"),
                    "completed": len(data.get("completed_indices", [])),
                })
            except Exception:
                # Skip corrupted checkpoints
                pass
        
        return sorted(checkpoints, key=lambda x: x["timestamp"], reverse=True)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only the most recent ones."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for old_checkpoint in checkpoints[self.max_checkpoints:]:
            try:
                old_checkpoint.unlink()
            except OSError:
                pass
    
    def get_resume_info(self, task_name: str, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about resuming from a checkpoint.
        
        Returns:
            Dictionary with resume info or None if no checkpoint
        """
        state = self.load(task_name, model_name)
        
        if state is None:
            return None
        
        return {
            "can_resume": True,
            "task_name": state.task_name,
            "model_name": state.model_name,
            "completed_samples": len(state.completed_indices),
            "timestamp": state.timestamp,
            "metadata": state.metadata,
        }
    
    def clear_all(self):
        """Delete all checkpoints."""
        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            try:
                checkpoint_file.unlink()
            except OSError:
                pass
