"""
Utility functions for structured data extraction using LLMs.

Provides helper functions for loading data, saving results, and other common extraction tasks.
CLI functionality has been removed - use these functions directly in your Python code.
"""
from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List
from some.io import read_json, write_json, read_jsonl, write_jsonl


def configure_logging(verbosity: int) -> None:
    """Configure logging with appropriate verbosity level."""
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers to avoid duplicate log lines
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter("%(levelname)s: %(message)s")

    if verbosity > 0:
        try:
            from some.progress import TqdmLoggingHandler
            handler = TqdmLoggingHandler()
        except Exception:
            handler = logging.StreamHandler()
    else:
        handler = logging.StreamHandler()

    handler.setLevel(level)
    handler.setFormatter(formatter)
    root.addHandler(handler)


def load_data_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSON, JSONL, or text file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if path.suffix.lower() == '.jsonl':
        # Use the centralized read_jsonl function from io.py
        return read_jsonl(file_path)
    elif path.suffix.lower() == '.json':
        data = read_json(file_path)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            raise ValueError(f"JSON file must contain a list or dict, got {type(data)}")
    elif path.suffix.lower() == '.txt':
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        return [{"text": line} for line in lines]
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .json, .jsonl, or .txt")


def load_prompt_builder_class(module_path: str, class_name: str):
    """Dynamically load a prompt builder class from a module."""
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_path}': {e}")
    except AttributeError as e:
        raise AttributeError(f"Class '{class_name}' not found in module '{module_path}': {e}")


def save_results(results: List[Dict[str, Any]], output_path: str, format: str = "json") -> None:
    """Save extraction results to file."""
    if format.lower() == "jsonl":
        # Use the centralized write_jsonl function from io.py
        write_jsonl(output_path, results)
    else:  # json
        # Use the centralized write_json function from io.py
        write_json(output_path, results)


def save_evaluation_data(inputs: List[Dict[str, Any]], results: List[Dict[str, Any]], output_path: str) -> None:
    """Save inputs and results together for evaluation purposes."""
    evaluation_data = []

    for i, (input_data, result) in enumerate(zip(inputs, results)):
        eval_record = {
            "id": i,
            "input": {
                "messages": input_data.get("messages", []),
                "response_format": str(input_data.get("response_format", "")),
                "result_key": input_data.get("result_key", "")
            },
            "output": result.get(input_data.get("result_key", "result")) if not result.get("error") else None,
            "raw_result": result,
            "success": not bool(result.get("error")),
            "error": result.get("error")
        }
        evaluation_data.append(eval_record)

    # Use the centralized write_json function from io.py
    write_json(output_path, evaluation_data)

