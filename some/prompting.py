from __future__ import annotations

from typing import Any, Dict


class BasePromptBuilder:
    """Generic interface for building model-agnostic prompt inputs for an item.

    Returns a dict payload that includes:
      - prompt_text: str - The formatted text prompt
      - image_path: Optional[str] - Local path to image file if needed
      - response_format: Optional Pydantic model class for structured outputs
      - result_key: Optional key name to store structured output under in results

    Language model subclasses will convert this generic format to their specific
    message format using their build_messages() method.
    """

    def build(self, item: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError



