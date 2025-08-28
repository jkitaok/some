from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import time

from some import BaseLanguageModel, register_language_model


class CustomLanguageModel(BaseLanguageModel):
    """Simple custom language model for testing."""

    def __init__(self, *, model: Optional[str] = None, **kwargs) -> None:
        super().__init__(model=model or "simple-custom")

    def generate(
        self,
        inputs: List[Dict[str, Any]],
        *,
        max_workers: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], int, float]:
        """Generate simple mock responses."""
        start_time = time.time()
        results = []

        for item in inputs:
            response = self._generate_single(item)
            results.append(response)
            # Simulate some processing time
            time.sleep(0.01)

        total_inference_time = time.time() - start_time
        return results, len(inputs), total_inference_time
    
    def _generate_single(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        result_key = payload.get("result_key", "result")
        return {
            "input_tokens": 10,
            "output_tokens": 20,
            result_key: {},
        }


# Register the custom language model
register_language_model("custom", lambda **kw: CustomLanguageModel(**kw))
