from __future__ import annotations

from typing import Any, Dict


class BasePromptBuilder:
    """Generic interface for building model-ready inputs for an item.

    Returns a dict payload that may include:
      - messages: List[{"role": ..., "content": ...}]
      - response_format: Optional Pydantic model class for structured outputs
      - result_key: Optional key name to store structured output under in results
    """

    def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError



