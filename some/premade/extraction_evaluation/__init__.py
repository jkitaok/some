"""
Extraction Evaluation - Pre-made Example

This package demonstrates how to evaluate extraction results given a schema,
prompt, and output using the SOME library's building blocks.

Supports both manual evaluation and labeled data evaluation modes.

Key components:
- BasicEvaluation: Simple evaluation schema for manual evaluation
- LabeledEvaluation: Enhanced evaluation schema for labeled data comparison
- EvaluationPrompt: Prompt builder for manual evaluations
- LabeledEvaluationPrompt: Prompt builder for labeled data evaluations
- labeled_utils: Utilities for working with labeled data
- run_evaluation: Main execution script supporting both evaluation modes

Example usage:
    # Manual evaluation
    from some.premade.extraction_evaluation import main
    results = main()

    # Labeled data evaluation
    results = main(labeled_data_path="path/to/labeled_data.json")

    # Or run directly
    python -m some.premade.extraction_evaluation.run_evaluation
    python -m some.premade.extraction_evaluation.run_evaluation --labeled-data path/to/data.json
"""

from .schema import BasicEvaluation, LabeledEvaluation, EvaluationMode
from .prompt import EvaluationPrompt, LabeledEvaluationPrompt
from .labeled_utils import (
    load_labeled_data,
    validate_labeled_item,
    format_evaluation_record,
    calculate_accuracy_metrics,
    analyze_field_differences
)
from .run_evaluation import main, run_manual_evaluation, run_labeled_evaluation

__all__ = [
    # Schemas
    "BasicEvaluation",
    "LabeledEvaluation",
    "EvaluationMode",

    # Prompt builders
    "EvaluationPrompt",
    "LabeledEvaluationPrompt",

    # Labeled data utilities
    "load_labeled_data",
    "validate_labeled_item",
    "format_evaluation_record",
    "calculate_accuracy_metrics",
    "analyze_field_differences",

    # Main functions
    "main",
    "run_manual_evaluation",
    "run_labeled_evaluation"
]
