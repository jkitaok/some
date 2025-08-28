"""
Labeled Data Evaluation Utilities

Provides utility functions for working with labeled data in evaluation scenarios.
Includes data loading, validation, and metric calculation functions.
"""
from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import json

from some.io import read_json, read_jsonl


def load_labeled_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load labeled evaluation data from JSON or JSONL file.
    
    Expected format for each item:
    {
        "id": "unique_identifier",
        "original_text": "source text for extraction",
        "ground_truth": {...},  # The correct/expected extraction result
        "extraction_prompt": "prompt used for extraction (optional)",
        "expected_schema": {...},  # Schema definition (optional)
        "evaluation_context": "additional context (optional)"
    }
    
    Args:
        file_path: Path to the labeled data file (.json or .jsonl)
        
    Returns:
        List of labeled data items
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is unsupported or data is invalid
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Labeled data file not found: {file_path}")
    
    if path.suffix.lower() == '.jsonl':
        data = read_jsonl(file_path)
    elif path.suffix.lower() == '.json':
        data = read_json(file_path)
        if isinstance(data, dict):
            data = [data]
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .json or .jsonl")
    
    if not data:
        raise ValueError(f"No data found in file: {file_path}")
    
    # Validate data structure
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} is not a dictionary")
        
        required_fields = ["original_text", "ground_truth"]
        for field in required_fields:
            if field not in item:
                raise ValueError(f"Item {i} missing required field: {field}")
    
    return data


def validate_labeled_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize a labeled data item.
    
    Args:
        item: Labeled data item to validate
        
    Returns:
        Validated and normalized item
        
    Raises:
        ValueError: If the item is invalid
    """
    if not isinstance(item, dict):
        raise ValueError("Item must be a dictionary")
    
    required_fields = ["original_text", "ground_truth"]
    for field in required_fields:
        if field not in item:
            raise ValueError(f"Missing required field: {field}")
    
    # Ensure optional fields have default values
    normalized_item = {
        "id": item.get("id", f"item_{hash(str(item))%10000:04d}"),
        "original_text": item["original_text"],
        "ground_truth": item["ground_truth"],
        "extraction_prompt": item.get("extraction_prompt", ""),
        "expected_schema": item.get("expected_schema", {}),
        "evaluation_context": item.get("evaluation_context", "")
    }
    
    return normalized_item


def calculate_field_accuracy(extraction: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, bool]:
    """
    Calculate field-level accuracy between extraction and ground truth.
    
    Args:
        extraction: The extraction result
        ground_truth: The ground truth data
        
    Returns:
        Dictionary mapping field names to accuracy (True/False)
    """
    field_accuracy = {}
    
    # Check all fields in ground truth
    for field, gt_value in ground_truth.items():
        if field in extraction:
            extracted_value = extraction[field]
            # Handle different types of comparisons
            if isinstance(gt_value, (list, tuple)) and isinstance(extracted_value, (list, tuple)):
                # For lists, check if they contain the same elements (order-independent)
                field_accuracy[field] = set(gt_value) == set(extracted_value)
            else:
                # For other types, direct comparison
                field_accuracy[field] = gt_value == extracted_value
        else:
            # Field missing in extraction
            field_accuracy[field] = False
    
    return field_accuracy


def calculate_accuracy_metrics(extraction: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate comprehensive accuracy metrics.
    
    Args:
        extraction: The extraction result
        ground_truth: The ground truth data
        
    Returns:
        Dictionary containing accuracy metrics
    """
    field_accuracy = calculate_field_accuracy(extraction, ground_truth)
    
    # Overall accuracy (proportion of correct fields)
    total_fields = len(ground_truth)
    correct_fields = sum(field_accuracy.values())
    accuracy_score = correct_fields / total_fields if total_fields > 0 else 0.0
    
    # For list fields, calculate precision, recall, F1
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for field, gt_value in ground_truth.items():
        if isinstance(gt_value, (list, tuple)) and field in extraction:
            extracted_value = extraction[field]
            if isinstance(extracted_value, (list, tuple)):
                gt_set = set(gt_value)
                ext_set = set(extracted_value)
                
                if len(ext_set) > 0:
                    precision = len(gt_set & ext_set) / len(ext_set)
                    precision_scores.append(precision)
                
                if len(gt_set) > 0:
                    recall = len(gt_set & ext_set) / len(gt_set)
                    recall_scores.append(recall)
                
                if len(gt_set) > 0 and len(ext_set) > 0:
                    precision = len(gt_set & ext_set) / len(ext_set)
                    recall = len(gt_set & ext_set) / len(gt_set)
                    if precision + recall > 0:
                        f1 = 2 * (precision * recall) / (precision + recall)
                        f1_scores.append(f1)
    
    # Average metrics for list fields
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else None
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else None
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else None
    
    return {
        "accuracy_score": accuracy_score,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1_score": avg_f1
    }


def analyze_field_differences(extraction: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Union[List[str], int, str]]:
    """
    Analyze differences between extraction and ground truth at field level.

    Args:
        extraction: The extraction result
        ground_truth: The ground truth data

    Returns:
        Dictionary with lists of missing, extra, and incorrect fields, plus counts and string representations
    """
    missing_fields = []
    extra_fields = []
    incorrect_fields = []

    # Find missing fields (in ground truth but not in extraction)
    for field in ground_truth:
        if field not in extraction:
            missing_fields.append(field)

    # Find extra fields (in extraction but not in ground truth)
    for field in extraction:
        if field not in ground_truth:
            extra_fields.append(field)

    # Find incorrect fields (in both but with different values)
    for field in ground_truth:
        if field in extraction:
            gt_value = ground_truth[field]
            ext_value = extraction[field]

            if isinstance(gt_value, (list, tuple)) and isinstance(ext_value, (list, tuple)):
                if set(gt_value) != set(ext_value):
                    incorrect_fields.append(field)
            else:
                if gt_value != ext_value:
                    incorrect_fields.append(field)

    return {
        "missing_fields": missing_fields,
        "extra_fields": extra_fields,
        "incorrect_fields": incorrect_fields,
        "missing_fields_count": len(missing_fields),
        "extra_fields_count": len(extra_fields),
        "incorrect_fields_count": len(incorrect_fields),
        "missing_fields_list": ", ".join(missing_fields) if missing_fields else None,
        "extra_fields_list": ", ".join(extra_fields) if extra_fields else None,
        "incorrect_fields_list": ", ".join(incorrect_fields) if incorrect_fields else None
    }


def format_evaluation_record(
    original_item: Dict[str, Any],
    extraction_output: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Format a labeled data item and extraction output for evaluation.
    
    Args:
        original_item: Original labeled data item
        extraction_output: The extraction result to evaluate
        
    Returns:
        Formatted record ready for evaluation prompt
    """
    validated_item = validate_labeled_item(original_item)
    
    return {
        "original_text": validated_item["original_text"],
        "extraction_prompt": validated_item["extraction_prompt"],
        "expected_schema": validated_item["expected_schema"],
        "extraction_output": extraction_output,
        "ground_truth": validated_item["ground_truth"],
        "evaluation_context": validated_item["evaluation_context"]
    }
