"""
Extraction Evaluation Schema

Defines the structured output format for evaluating extraction results.
Uses a simple evaluation schema similar to the generic extraction example.
Supports both manual evaluation and labeled data comparison.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum

class EvaluationMode(str, Enum):
    """Evaluation mode for different types of evaluation."""
    MANUAL = "manual"  # Manual evaluation without ground truth
    LABELED = "labeled"  # Evaluation against labeled ground truth data

class BasicEvaluation(BaseModel):
    """
    Simple evaluation of an extraction result.

    This schema provides a straightforward assessment of extraction quality
    focusing on correctness and formatting compliance.
    """
    correct: bool
    formatted: bool
    reasoning: Optional[str] = None
    confidence: Optional[float] = None

class LabeledEvaluation(BaseModel):
    """
    Enhanced evaluation schema for labeled data comparison.

    This schema provides detailed metrics when comparing extraction results
    against ground truth labeled data.
    """
    # Overall accuracy metrics
    exact_match: bool = Field(description="Does the extraction exactly match the ground truth?")
    partial_match: bool = Field(description="Does the extraction partially match the ground truth?")
    formatted: bool = Field(description="Does the output follow the expected schema properly?")

    # Quantitative metrics
    accuracy_score: float = Field(ge=0.0, le=1.0, description="Overall accuracy score (0.0-1.0)")

    # Field analysis (simplified for OpenAI compatibility)
    missing_fields_count: int = Field(ge=0, description="Number of fields missing from extraction")
    extra_fields_count: int = Field(ge=0, description="Number of extra fields in extraction")
    incorrect_fields_count: int = Field(ge=0, description="Number of fields with incorrect values")

    # Optional detailed metrics
    precision: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Precision score for multi-value fields")
    recall: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Recall score for multi-value fields")
    f1_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="F1 score for multi-value fields")

    # Qualitative assessment
    reasoning: Optional[str] = Field(default=None, description="Detailed explanation of the evaluation")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence in the evaluation")

    # Ground truth comparison details
    ground_truth_summary: Optional[str] = Field(default=None, description="Summary of what the ground truth contained")
    extraction_summary: Optional[str] = Field(default=None, description="Summary of what was extracted")

    # Field details as strings (OpenAI compatible)
    missing_fields_list: Optional[str] = Field(default=None, description="Comma-separated list of missing fields")
    extra_fields_list: Optional[str] = Field(default=None, description="Comma-separated list of extra fields")
    incorrect_fields_list: Optional[str] = Field(default=None, description="Comma-separated list of incorrect fields")
