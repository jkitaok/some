from __future__ import annotations

from typing import Any, Dict

from .my_schema import ProductSpec
from some.prompting import BasePromptBuilder

class ProductPrompt(BasePromptBuilder):
    def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
        text = item["text"]
        return {
            "prompt_text": f"Extract ProductSpec as JSON from this text and adhere strictly to the schema.\n{text}",
            "response_format": ProductSpec,
            "result_key": "product",
        }

