"""
Utility functions for structured data extraction using LLMs.

Provides helper functions for loading data, saving results, and other common extraction tasks.
CLI functionality has been removed - use these functions directly in your Python code.
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from extraction.io import read_json


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
            from extraction.progress import TqdmLoggingHandler
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
        items = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
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


def save_results(results: List[Dict[str, Any]], output_path: str, format: str = "json", include_prompts: bool = False) -> None:
    """Save extraction results to file, optionally including prompts for evaluation."""
    output_path_obj = Path(output_path)
    if output_path_obj.parent != Path('.'):  # Only create parent dirs if not current directory
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "jsonl":
        with open(output_path_obj, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
    else:  # json
        with open(output_path_obj, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


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

    output_path_obj = Path(output_path)
    if output_path_obj.parent != Path('.'):
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path_obj, 'w', encoding='utf-8') as f:
        json.dump(evaluation_data, f, ensure_ascii=False, indent=2)





# CLI functionality has been removed - cmd_extract function is no longer available
# Use the utility functions in this module directly in your Python code







# CLI functionality has been removed - build_parser and main functions are no longer available
# Use the utility functions in this module directly in your Python code
