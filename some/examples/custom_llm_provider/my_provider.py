from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from some import BaseLanguageModel


class MyLanguageModel(BaseLanguageModel):
    """Skeleton provider showing the minimal contract to integrate a custom LLM.

    Fill in your own client setup and call in `_generate_single` or directly in `generate`.
    Batching behavior is up to you; you can reuse the pattern from built-in providers.
    """

    def __init__(self, *, model: Optional[str] = None, **kwargs) -> None:
        super().__init__(model=model)
        # TODO: Initialize your client here (e.g., HTTP session, SDK client, etc.)

    def _generate_single(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: Replace with actual model call.
        # Respect response_format if you can parse into Pydantic model.
        result_key = payload.get("result_key", "result")
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            result_key: {"demo": True},
        }

    def generate(
        self,
        inputs: List[Dict[str, Any]],
        *,
        max_workers: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        if not inputs:
            return [], 0
        # Minimal sequential implementation; you can parallelize like built-ins
        results = []
        for item in inputs:
            try:
                results.append(self._generate_single(item))
            except Exception as e:
                result_key = item.get("result_key", "result")
                results.append({
                    "input_tokens": 0,
                    "output_tokens": 0,
                    result_key: None,
                    "error": str(e),
                })
        return results, 1
