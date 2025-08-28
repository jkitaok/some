"""
Extraction Evaluation Prompt Builder

Defines the prompt template for evaluating extraction results against schemas.
Uses the BasePromptBuilder interface from the SOME library.
Supports both manual evaluation and labeled data comparison.
"""
from __future__ import annotations

from typing import Any, Dict
import json

from .schema import BasicEvaluation, LabeledEvaluation, EvaluationMode
from some.prompting import BasePromptBuilder

class EvaluationPrompt(BasePromptBuilder):
    """
    Prompt builder for evaluating extraction results.

    This prompt evaluates extractions on two key criteria:
    - Correct: Is the extracted information factually accurate?
    - Formatted: Does the output follow the expected schema properly?
    """
    
    def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a prompt for evaluating an extraction result.
        
        Args:
            item: Dictionary containing:
                - 'original_text': The source text that was processed
                - 'extraction_prompt': The prompt used for extraction
                - 'expected_schema': The schema definition (as dict or string)
                - 'extraction_output': The actual extraction result
                - 'evaluation_context': Optional additional context
                
        Returns:
            Dictionary with prompt configuration for the language model
        """
        original_text = item["original_text"]
        extraction_prompt = item["extraction_prompt"]
        expected_schema = item["expected_schema"]
        extraction_output = item["extraction_output"]
        evaluation_context = item.get("evaluation_context", "")
        
        # Format schema for display
        if isinstance(expected_schema, dict):
            schema_display = json.dumps(expected_schema, indent=2)
        else:
            schema_display = str(expected_schema)
        
        # Format extraction output for display
        if isinstance(extraction_output, dict):
            output_display = json.dumps(extraction_output, indent=2)
        else:
            output_display = str(extraction_output)
        
        prompt_text = f"""Evaluate the extraction result on two key criteria:
1. Is the extracted information factually accurate based on the input text?
2. Does the output follow the expected format/schema properly?

Respond with your evaluation in the specified JSON format.

**ORIGINAL SOURCE TEXT:**
{original_text}

**EXTRACTION PROMPT USED:**
{extraction_prompt}

**EXPECTED SCHEMA:**
{schema_display}

**ACTUAL EXTRACTION OUTPUT:**
{output_display}

{f"**ADDITIONAL CONTEXT:**\n{evaluation_context}\n" if evaluation_context else ""}"""

        return {
            "prompt_text": prompt_text,
            "response_format": BasicEvaluation,
            "result_key": "evaluation_result",
        }


class LabeledEvaluationPrompt(BasePromptBuilder):
    """
    Enhanced prompt builder for evaluating extraction results against labeled ground truth data.

    This prompt provides detailed comparison between extraction results and ground truth,
    including field-level accuracy, missing/extra fields, and quantitative metrics.
    """

    def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a prompt for evaluating an extraction result against ground truth.

        Args:
            item: Dictionary containing:
                - 'original_text': The source text that was processed
                - 'extraction_prompt': The prompt used for extraction
                - 'expected_schema': The schema definition (as dict or string)
                - 'extraction_output': The actual extraction result
                - 'ground_truth': The labeled ground truth data
                - 'evaluation_context': Optional additional context

        Returns:
            Dictionary with prompt configuration for the language model
        """
        original_text = item["original_text"]
        extraction_prompt = item["extraction_prompt"]
        expected_schema = item["expected_schema"]
        extraction_output = item["extraction_output"]
        ground_truth = item["ground_truth"]
        evaluation_context = item.get("evaluation_context", "")

        # Format schema for display
        if isinstance(expected_schema, dict):
            schema_display = json.dumps(expected_schema, indent=2)
        else:
            schema_display = str(expected_schema)

        # Format extraction output for display
        if isinstance(extraction_output, dict):
            output_display = json.dumps(extraction_output, indent=2)
        else:
            output_display = str(extraction_output)

        # Format ground truth for display
        if isinstance(ground_truth, dict):
            ground_truth_display = json.dumps(ground_truth, indent=2)
        else:
            ground_truth_display = str(ground_truth)

        prompt_text = f"""Evaluate the extraction result by comparing it against the provided ground truth labeled data.

Perform a detailed analysis including:
1. **Exact Match**: Does the extraction exactly match the ground truth?
2. **Partial Match**: Does the extraction partially match the ground truth?
3. **Field-Level Analysis**: Compare each field individually
4. **Schema Compliance**: Does the output follow the expected format?
5. **Quantitative Metrics**: Calculate accuracy, precision, recall, and F1 scores where applicable

Respond with your evaluation in the specified JSON format.

**ORIGINAL SOURCE TEXT:**
{original_text}

**EXTRACTION PROMPT USED:**
{extraction_prompt}

**EXPECTED SCHEMA:**
{schema_display}

**GROUND TRUTH (LABELED DATA):**
{ground_truth_display}

**ACTUAL EXTRACTION OUTPUT:**
{output_display}

{f"**ADDITIONAL CONTEXT:**\n{evaluation_context}\n" if evaluation_context else ""}

**EVALUATION INSTRUCTIONS:**
- Compare each field in the extraction output against the corresponding field in the ground truth
- Identify missing fields (present in ground truth but not in extraction)
- Identify extra fields (present in extraction but not in ground truth)
- Identify incorrect fields (present in both but with different values)
- Calculate an overall accuracy score based on the proportion of correct fields
- For list/array fields, calculate precision, recall, and F1 scores if applicable
- Provide detailed reasoning for your evaluation
- Assign a confidence score to your evaluation"""

        return {
            "prompt_text": prompt_text,
            "response_format": LabeledEvaluation,
            "result_key": "labeled_evaluation_result",
        }
