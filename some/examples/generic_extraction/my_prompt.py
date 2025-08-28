from __future__ import annotations

from typing import Any, Dict

from .my_schema import ProductSpec, BasicEvaluation
from some.prompting import BasePromptBuilder

class ProductPrompt(BasePromptBuilder):
    def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
        text = item["text"]
        return {
            "prompt_text": f"Extract ProductSpec as JSON from this text and adhere strictly to the schema.\n{text}",
            "response_format": ProductSpec,
            "result_key": "product",
        }
    
class EvaluationPrompt(BasePromptBuilder):
    def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
        input_prompt = item["input_prompt"]
        expected_format = item["expected_format"]
        extraction_output = item["extraction_output"]

        prompt = f"""Evaluate the extraction result on two key criteria: 1. Is the extracted information factually accurate based on the input text? 2. Does the output follow the expected format/schema properly? Respond with your evaluation in the specified JSON format.

        **Original Input Text:**
        {input_prompt}

        **Expected Format:**
        {expected_format}

        **Extraction Result:**
        {extraction_output}
        """

        return {
            "prompt_text": prompt,
            "response_format": BasicEvaluation,
            "result_key": "evaluation",
        }
